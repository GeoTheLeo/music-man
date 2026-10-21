// Music Man on Azure Container Apps: three services sharing one PostgreSQL.
//
//   mcp      MCP server (Streamable HTTP) for Copilot Studio and other MCP
//            clients; X-API-Key required; one replica (one guardrail run)
//   webhook  HMAC-signed, idempotent fraud-alert trigger
//   review   Streamlit approval app; behind Microsoft Entra sign-in, and not
//            exposed at all unless an Entra app registration is supplied
//
// All three run the same image with different commands, pull it from ACR
// with a managed identity (no registry passwords), keep secrets as Container
// Apps secrets, and scale to zero when idle. infra/deploy.sh drives it.

targetScope = 'resourceGroup'

@description('Short prefix for resource names.')
param prefix string = 'musicman'

param location string = resourceGroup().location

@description('false on the first pass: create the registry and database, then build and push the image.')
param deployApps bool = true

@description('Image tag in the registry, e.g. music-man:20261021-1')
param image string = 'music-man:latest'

@secure()
param postgresPassword string

@secure()
param mcpApiKey string

@secure()
param webhookSecret string

@secure()
@description('Optional: lets webhook-triggered investigations call Claude.')
param anthropicApiKey string = ''

@description('Entra app registration (client id) for the review app sign-in. Empty: the review app gets no public ingress.')
param reviewClientId string = ''

@secure()
param reviewClientSecret string = ''

@description('propose_hold writes nothing when true.')
param dryRun bool = false

var suffix = uniqueString(resourceGroup().id)
var postgresUser = 'musicman'
var postgresDb = 'musicman'
var reviewAuthEnabled = !empty(reviewClientId)

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs-${suffix}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${prefix}acr${suffix}'
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource pullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-pull-${suffix}'
  location: location
}

// AcrPull for the apps' identity.
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, pullIdentity.id, 'acrpull')
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: pullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = {
  name: '${prefix}-pg-${suffix}'
  location: location
  sku: { name: 'Standard_B1ms', tier: 'Burstable' }
  properties: {
    version: '16'
    administratorLogin: postgresUser
    administratorLoginPassword: postgresPassword
    storage: { storageSizeGB: 32 }
    backup: { backupRetentionDays: 7, geoRedundantBackup: 'Disabled' }
    highAvailability: { mode: 'Disabled' }
    network: { publicNetworkAccess: 'Enabled' }
  }
}

resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = {
  parent: postgres
  name: postgresDb
}

// Container Apps on the Consumption plan have no fixed outbound IP, so the
// server admits Azure-hosted clients (0.0.0.0 rule) and relies on its
// password and TLS. VNet integration with a private endpoint is the next step.
resource allowAzure 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = {
  parent: postgres
  name: 'AllowAzureServices'
  properties: { startIpAddress: '0.0.0.0', endIpAddress: '0.0.0.0' }
}

resource appsEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env-${suffix}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var databaseUrl = 'postgresql://${postgresUser}:${uriComponent(postgresPassword)}@${postgres.properties.fullyQualifiedDomainName}:5432/${postgresDb}?sslmode=require'
var imageRef = '${registry.properties.loginServer}/${image}'
var mcpName = '${prefix}-mcp'
var mcpHost = '${mcpName}.${appsEnv.properties.defaultDomain}'

// Separate from `common`: identity must be computable at deployment start.
var appIdentity = {
  type: 'UserAssigned'
  userAssignedIdentities: { '${pullIdentity.id}': {} }
}

var common = {
  registries: [{ server: registry.properties.loginServer, identity: pullIdentity.id }]
  env: [
    { name: 'MUSIC_MAN_DATABASE_URL', secretRef: 'database-url' }
    { name: 'MUSIC_MAN_DRY_RUN', value: dryRun ? '1' : '' }
  ]
  scale: { minReplicas: 0, maxReplicas: 1 }
}

resource mcp 'Microsoft.App/containerApps@2024-03-01' = if (deployApps) {
  name: mcpName
  location: location
  identity: appIdentity
  dependsOn: [acrPull, database, allowAzure]
  properties: {
    managedEnvironmentId: appsEnv.id
    configuration: {
      ingress: { external: true, targetPort: 8765, transport: 'auto' }
      registries: common.registries
      secrets: [
        { name: 'database-url', value: databaseUrl }
        { name: 'mcp-api-key', value: mcpApiKey }
      ]
    }
    template: {
      containers: [
        {
          name: 'mcp'
          image: imageRef
          command: ['python', '-m', 'music_man.mcp_server', '--http', '8765', '--host', '0.0.0.0']
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: concat(common.env, [
            { name: 'MUSIC_MAN_MCP_API_KEY', secretRef: 'mcp-api-key' }
            { name: 'MUSIC_MAN_MCP_ALLOWED_HOSTS', value: mcpHost }
          ])
          probes: [
            { type: 'Liveness', httpGet: { path: '/healthz', port: 8765 } }
            { type: 'Readiness', httpGet: { path: '/healthz', port: 8765 } }
          ]
        }
      ]
      // One replica: policy checks are remembered per process (one guardrail run).
      scale: common.scale
    }
  }
}

resource webhook 'Microsoft.App/containerApps@2024-03-01' = if (deployApps) {
  name: '${prefix}-webhook'
  location: location
  identity: appIdentity
  dependsOn: [acrPull, database, allowAzure]
  properties: {
    managedEnvironmentId: appsEnv.id
    configuration: {
      ingress: { external: true, targetPort: 8800, transport: 'auto' }
      registries: common.registries
      secrets: concat(
        [
          { name: 'database-url', value: databaseUrl }
          { name: 'webhook-secret', value: webhookSecret }
        ],
        empty(anthropicApiKey) ? [] : [{ name: 'anthropic-api-key', value: anthropicApiKey }]
      )
    }
    template: {
      containers: [
        {
          name: 'webhook'
          image: imageRef
          command: ['uvicorn', 'music_man.webhook.app:app', '--host', '0.0.0.0', '--port', '8800']
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: concat(
            common.env,
            [{ name: 'MUSIC_MAN_WEBHOOK_SECRET', secretRef: 'webhook-secret' }],
            empty(anthropicApiKey) ? [] : [{ name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }]
          )
        }
      ]
      scale: common.scale
    }
  }
}

resource review 'Microsoft.App/containerApps@2024-03-01' = if (deployApps) {
  name: '${prefix}-review'
  location: location
  identity: appIdentity
  dependsOn: [acrPull, database, allowAzure]
  properties: {
    managedEnvironmentId: appsEnv.id
    configuration: {
      // No sign-in configured means no public endpoint: approvals are never open to the internet.
      ingress: { external: reviewAuthEnabled, targetPort: 8501, transport: 'auto' }
      registries: common.registries
      secrets: concat(
        [{ name: 'database-url', value: databaseUrl }],
        reviewAuthEnabled ? [{ name: 'microsoft-provider-authentication-secret', value: reviewClientSecret }] : []
      )
    }
    template: {
      containers: [
        {
          name: 'review'
          image: imageRef
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: common.env
        }
      ]
      scale: common.scale
    }
  }
}

// Built-in authentication ("Easy Auth"): only users of this Entra tenant reach the review app.
resource reviewAuth 'Microsoft.App/containerApps/authConfigs@2024-03-01' = if (deployApps && reviewAuthEnabled) {
  parent: review
  name: 'current'
  properties: {
    platform: { enabled: true }
    globalValidation: { unauthenticatedClientAction: 'RedirectToLoginPage', redirectToProvider: 'azureactivedirectory' }
    identityProviders: {
      azureActiveDirectory: {
        enabled: true
        registration: {
          clientId: reviewClientId
          clientSecretSettingName: 'microsoft-provider-authentication-secret'
          openIdIssuer: '${environment().authentication.loginEndpoint}${tenant().tenantId}/v2.0'
        }
      }
    }
  }
}

output registry string = registry.properties.loginServer
output registryName string = registry.name
output mcpUrl string = deployApps ? 'https://${mcpHost}/mcp' : ''
output webhookUrl string = deployApps ? 'https://${webhook!.properties.configuration.ingress.fqdn}/events' : ''
output reviewUrl string = deployApps && reviewAuthEnabled ? 'https://${review!.properties.configuration.ingress.fqdn}' : ''
output postgresHost string = postgres.properties.fullyQualifiedDomainName
