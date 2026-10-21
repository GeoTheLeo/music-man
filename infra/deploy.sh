#!/usr/bin/env bash
# Deploy Music Man to Azure Container Apps (see infra/main.bicep).
#
#   az login
#   bash infra/deploy.sh            # first run creates everything; re-runs update it
#
# Secrets (Postgres password, MCP API key, webhook secret) are generated on
# the first run and kept in infra/.secrets.env (git-ignored), so re-runs
# reuse them. Set ANTHROPIC_API_KEY in the environment to let
# webhook-triggered investigations call Claude.
#
# Order: (1) registry, database, and environment; (2) an Entra app
# registration for the review app's sign-in, whose redirect URI needs the
# environment's domain; (3) build and push the image locally (ACR Tasks
# are often unavailable on free-credit subscriptions); (4) the three apps.

set -euo pipefail
cd "$(dirname "$0")/.."

RG="${RG:-musicman-rg}"
# Default region: PostgreSQL Flexible Server provisioning is restricted for some
# subscriptions in Germany West Central and West Europe; Sweden Central is EU and nearby.
LOCATION="${LOCATION:-swedencentral}"
PREFIX="${PREFIX:-musicman}"
SECRETS=infra/.secrets.env
TAG="music-man:$(date -u +%Y%m%d-%H%M%S)"

if [[ ! -f "$SECRETS" ]]; then
  gen() { python -c "import secrets; print(secrets.token_urlsafe(32))"; }
  {
    echo "POSTGRES_PASSWORD=$(gen)"
    echo "MCP_API_KEY=$(gen)"
    echo "WEBHOOK_SECRET=$(gen)"
  } > "$SECRETS"
  echo "Generated secrets in $SECRETS (keep it; it is git-ignored)."
fi
# shellcheck disable=SC1090
source "$SECRETS"

deploy() {
  az deployment group create -g "$RG" -n "music-man-$1" -f infra/main.bicep -o none \
    -p prefix="$PREFIX" deployApps="$2" image="$TAG" \
       postgresPassword="$POSTGRES_PASSWORD" mcpApiKey="$MCP_API_KEY" webhookSecret="$WEBHOOK_SECRET" \
       anthropicApiKey="${ANTHROPIC_API_KEY:-}" \
       reviewClientId="${REVIEW_CLIENT_ID:-}" reviewClientSecret="${REVIEW_CLIENT_SECRET:-}"
}
output() { az deployment group show -g "$RG" -n "music-man-$1" --query "properties.outputs.$2.value" -o tsv; }

echo "== 1/4 resource group, registry, database, environment ($LOCATION)"
az group create -n "$RG" -l "$LOCATION" -o none
deploy base false
REGISTRY=$(output base registryName)
DOMAIN=$(az containerapp env list -g "$RG" --query "[0].properties.defaultDomain" -o tsv)

echo "== 2/4 Entra sign-in for the review app"
REVIEW_HOST="$PREFIX-review.$DOMAIN"
APP_ID=$(az ad app list --display-name "$PREFIX-review" --query "[0].appId" -o tsv)
if [[ -z "$APP_ID" ]]; then
  APP_ID=$(az ad app create --display-name "$PREFIX-review" --sign-in-audience AzureADMyOrg \
    --web-redirect-uris "https://$REVIEW_HOST/.auth/login/aad/callback" \
    --enable-id-token-issuance true --query appId -o tsv)
fi
if [[ -z "${REVIEW_CLIENT_SECRET:-}" ]]; then
  REVIEW_CLIENT_SECRET=$(az ad app credential reset --id "$APP_ID" --display-name deploy --years 1 --query password -o tsv)
  echo "REVIEW_CLIENT_SECRET=$REVIEW_CLIENT_SECRET" >> "$SECRETS"
fi
REVIEW_CLIENT_ID="$APP_ID"

echo "== 3/4 build and push $TAG"
az acr login -n "$REGISTRY"
docker build -t "$REGISTRY.azurecr.io/$TAG" .
docker push "$REGISTRY.azurecr.io/$TAG"

echo "== 4/4 container apps"
deploy apps true

echo
echo "MCP server:  $(output apps mcpUrl)   (header X-API-Key: see MCP_API_KEY in $SECRETS)"
echo "Webhook:     $(output apps webhookUrl)"
echo "Review app:  $(output apps reviewUrl)   (sign in with your Entra work account)"
