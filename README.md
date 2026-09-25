# Music Man — Music-Streaming Royalty-Fraud Lakehouse

An agent that investigates suspicious music-streaming patterns (bot-farm
royalty fraud) and proposes withholding a payout — a human always approves
before anything actually happens. Built on PySpark + Delta Lake + MLflow,
running entirely locally, portable to a real Databricks workspace.

Companion project to [NorthStar's Agentic Intervention Copilot](../northstar-platform):
same human-in-the-loop spine (agent proposes, never executes), different
stack — this one proves out lakehouse-scale data engineering instead of a
React/Next.js frontend.

## Results (a real run, not a mockup)

Verified against the actual public catalog and a full pipeline run:

- **89,740 real tracks, 31,428 real artists** seeded from a public Kaggle Spotify dataset
- **2.1M synthetic play events** processed through the full bronze → silver → gold medallion pipeline on local PySpark + Delta Lake
- **40 injected fraud cases** across 4 distinct bot-farm signatures — the trained classifier caught **100% of them (recall 1.0)** on a held-out test split, at **67% precision**. The false positives are genuinely popular artists whose scale overlaps with the fraud signature — which is exactly the argument for why a human reviews every hold before it takes effect, not a flaw to hide.

**A real excerpt from a live agent run** — it didn't just call tools in sequence, it caught a problem in my own synthetic-data generator mid-investigation:

> *"device_concentration_ratio is 0.0668 here and 0.0667–0.0669 for every other top candidate inspected (Cachureos, King 810, Feid). That near-identical value across unrelated artists looks like a pipeline artifact rather than an independent signal, so it should NOT be weighted as corroborating evidence."*

Unprompted, it also noticed the model's anomaly scores were saturated at 1.0 across 39 artists, refused to rank candidates by score alone, tie-broke on financial exposure instead, and flagged that the shared signal fingerprint across cases pointed to one coordinated bot operator rather than 39 unrelated incidents — then declined to re-propose a hold that already existed for one artist/period, and picked the next-best candidate instead.

## One-time setup

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
```

### 2. Kaggle account (for the seed catalog)

The artist/track catalog is seeded from a real public dataset, not
generated from scratch:

1. Sign up at [kaggle.com](https://www.kaggle.com).
2. Profile icon → Settings → API → **Create New Token** (downloads `kaggle.json`).
3. Place it at `C:\Users\<you>\.kaggle\kaggle.json`.

### 3. Anthropic API key (for the agent)

```bash
copy .env.example .env
```

Fill in `ANTHROPIC_API_KEY=sk-ant-...` in `.env`.

### 4. Windows only: bundled 64-bit JDK + Hadoop native libraries

PySpark needs a 64-bit JVM and Windows-specific Hadoop native binaries
(`winutils.exe`, `hadoop.dll`) that don't ship with PySpark itself. These
live in `tools/` (git-ignored — too large to commit, and trivially
re-downloadable):

```bash
mkdir tools
cd tools

# 64-bit JDK 17 (Eclipse Temurin)
curl -sL -o jdk17.zip "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse?project=jdk"
powershell -Command "Expand-Archive -Path jdk17.zip -DestinationPath . -Force"
del jdk17.zip

# Hadoop 3.3.6 winutils.exe + hadoop.dll (matches PySpark 3.5.x's bundled Hadoop client)
mkdir hadoop-3.3.6\bin
curl -sL -o hadoop-3.3.6\bin\winutils.exe "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/winutils.exe"
curl -sL -o hadoop-3.3.6\bin\hadoop.dll "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/hadoop.dll"
```

`src/music_man/spark_session.py` points `JAVA_HOME`/`HADOOP_HOME` at these
automatically — no system-wide install or environment variables needed.
**Why a bundled JDK at all:** if your system's `java` is a 32-bit JVM (check
with `java -version` — look for "Client VM" instead of "64-Bit Server VM"),
it cannot load the 64-bit `hadoop.dll` above, and a 32-bit JVM's ~1.5GB heap
cap would cripple this project's actual goal (processing millions of
events) regardless. If your system Java is already a real 64-bit JDK 11+,
you can skip the JDK download and only fetch the Hadoop binaries.

**Also on Windows:** the first time you run anything Spark-related, allow
`.venv\Scripts\python.exe` through Windows Firewall (Windows Security →
Firewall & network protection → Allow an app → Browse to that exact path,
check Private + Public). Without it, the JVM↔Python-worker loopback socket
gets reset every time (`java.net.SocketException: Connection reset`).

## Running it

```bash
# 1. Generate the synthetic dataset (organic events + injected fraud cases)
#    and write it to the bronze layer.
.venv\Scripts\python -m music_man.pipeline.bronze

# 2. Clean, sessionize, and join -> silver.
.venv\Scripts\python -m music_man.pipeline.silver

# 3. Aggregate to per-artist/day fraud-signal features -> gold.
.venv\Scripts\python -m music_man.pipeline.gold

# 4. Train the fraud classifier (MLflow-tracked) and write anomaly_score
#    back into the gold table + a Parquet snapshot for the agent.
.venv\Scripts\python -m music_man.ml.train_model

# 5. Run the agent - plans, investigates, proposes a hold (never executes).
.venv\Scripts\python -m music_man.agent.run_agent "Find the most suspicious artist and propose a hold if warranted."

# 6. Review and approve/reject proposed holds.
.venv\Scripts\streamlit run src/music_man/review_app/app.py
```

Inspect MLflow runs with `.venv\Scripts\mlflow ui` (reads `./mlruns`).

## Testing

```bash
.venv\Scripts\pytest
```

`test_fraud_patterns.py` needs nothing external. `test_pipeline.py` spins
up a local Spark session and runs the real pipeline against a tiny
synthetic catalog (monkeypatched data directories - never touches your
real `data/{bronze,silver,gold}`).

## Architecture

```
src/music_man/
  paths.py           canonical filesystem paths (single source of truth)
  spark_session.py   local SparkSession factory (bundled JDK/Hadoop on Windows)
  generator/    catalog.py (Kaggle seed), users.py (Faker), events.py
                (organic + fraud event stream), fraud_patterns.py (the 4
                injectable fraud signatures)
  pipeline/     bronze.py -> silver.py -> gold.py (medallion architecture,
                PySpark + Delta Lake)
  ml/           train_model.py (supervised fraud classifier, MLflow-tracked)
  agent/        tools.py (Claude tools), run_agent.py (plan-then-execute),
                queue.py (SQLite approval queue)
  review_app/   app.py (Streamlit approval UI)
```

**The four injected fraud patterns** (`generator/fraud_patterns.py`), all
real, documented streaming-fraud signatures:

1. **Burst velocity** — an implausible play spike in a few minutes.
2. **Device/IP concentration** — a handful of devices account for most plays.
3. **Duration regularity** — bots replay a track for a near-identical
   duration every time; organic listening varies far more.
4. **Geo-impossible** — the same user/device streams from two far-apart
   locations minutes apart.

**Human-in-the-loop:** `propose_hold` is the agent's only side-effecting
tool, and its side effect is only queuing a pending row in
`data/queue.db` — it never withholds a real payment. The Streamlit
reviewer's Approve button is the only place anything "real" happens: it
flips the row's status and appends a line to `data/audit_log.csv`.

## Porting to real Databricks

- Swap the local Delta paths in `paths.py` for Unity Catalog managed table
  names; `pipeline/*.py` code is otherwise unchanged (same PySpark/Delta APIs).
- Upload `notebooks/` (thin wrappers importing from `src/music_man/*`).
- Point `agent/tools.py`'s snapshot read at a Databricks SQL Warehouse query
  instead of the local Parquet file.
- `ml/train_model.py`'s MLflow calls work as-is against a Databricks-hosted
  MLflow tracking server - just change the tracking URI.
