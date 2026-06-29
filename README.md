# 🛰️ EO Policy Skills Dashboard

A Streamlit web application that analyses policy documents for evidence of **space industry and Earth Observation downstream skills needs**.

Built as a production-ready refactor of the `pilot_policy_hybrid_pipeline_v4.1` Jupyter notebook.

---

## What It Does

Point the app at a folder of PDF policy documents → it automatically:

1. Extracts and cleans text (per-page, with page number tracking)
2. Splits into sentences for precise excerpt extraction
3. Runs **keyword search** against the EO skills taxonomy
4. Runs **semantic search** (SBERT offline, or OpenAI via env var)
5. Classifies each excerpt: `VALID EVIDENCE` / `WEAK EVIDENCE` / `NOT RELEVANT`
6. Lets you **human-validate** each excerpt in the dashboard
7. Exports to **CSV**, **Excel**, **JSON**, and **Markdown report**
8. Auto-saves exports to a persistent server-side outputs directory

### Skills Themes Detected

| Theme | Sample Keywords |
|---|---|
| EO Downstream Skills | earth observation, Copernicus, satellite imagery, remote sensing |
| Space Industry Skills | space economy, aerospace, satellite operations |
| Geospatial & GIS | GIS, spatial analysis, geoinformatics, QGIS |
| Copernicus & EU Space Services | Copernicus, C3S, CAMS, Galileo, EGNOS |
| Digital Skills for Space | AI for EO, cloud computing, data science, Python |
| Skills Gaps & Workforce Development | upskilling, reskilling, capacity building |
| Policy Support for Downstream Apps | smart specialisation, S3, value-added services |

---

## Why Folder Path Instead of Browser Upload

| | Browser upload | Folder path |
|---|---|---|
| File limit | ~20 reliable | Unlimited |
| Timeout risk | High (websocket) | None |
| Memory impact | All files in RAM | One file at a time |
| Large corpora (130 PDFs) | ❌ Crashes | ✅ Works |
| Requires server setup | No | Yes (volume mount) |

For any corpus larger than ~20 PDFs, the folder path mode is the only reliable option.
Browser upload is fine for quick tests with a handful of documents.

---

## Project Structure

```
eo_policy_dashboard/
├── app.py                          # Main Streamlit entry point
├── requirements.txt
├── .env.example                    # Environment variable template
├── railway.toml                    # Railway deployment config
├── Procfile                        # Start command
│
├── core/
│   ├── pdf_extractor.py            # PDF → DocumentContent (pdfminer + PyPDF2 fallback)
│   ├── chunker.py                  # Document → sentences
│   ├── search_engine.py            # Keyword + SBERT + OpenAI pipeline
│   ├── pipeline.py                 # Pipeline orchestrator
│   ├── cache_manager.py            # Two-layer cache + directory management
│   ├── taxonomy_loader.py          # YAML taxonomy loader
│   └── exporters.py                # CSV / Excel / PDF / JSON / Markdown
│
├── pages/
│   ├── results_table.py            # Paginated evidence card view
│   ├── charts.py                   # Plotly charts
│   └── human_validation.py        # Manual labelling UI
│
├── config/
│   └── taxonomy.yaml               # EO skills taxonomy (EN + NL + EL + PT + IT)
│
└── .streamlit/
    └── config.toml                 # Theme, upload limits
```

---

## Local Development

```bash
git clone https://github.com/abocin/eo-policy-dashboard.git
cd eo-policy-dashboard

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: set environment variables
cp .env.example .env
# edit .env with your values

streamlit run app.py
```

The app opens at http://localhost:8501.

---

## Environment Variables

| Variable | Required | Example | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | No | `sk-proj-...` | Enables OpenAI text-embedding-3-small |
| `PDF_FOLDER` | No | `/data/pdfs` | Pre-fills the folder path input |
| `CACHE_DIR` | No | `/data/cache` | Override embedding cache directory |

Copy `.env.example` to `.env` for local use. On Railway, set these in the **Variables** tab.

---

## Railway Deployment (Production)

### Why Railway

Railway auto-detects Python via Railpack — no Dockerfile needed. It supports persistent volumes, which are required for both the PDF folder and the embedding cache.

### Step 1 — Push to GitHub and create a Railway project

```bash
git push origin main
```

In the [Railway dashboard](https://railway.app):
**New project → Deploy from GitHub repo → select `eo-policy-dashboard`**

### Step 2 — Add a persistent volume

A volume is required for two things:
- Storing your PDF corpus (`/data/pdfs`)
- Persisting embedding vectors across redeploys (`/data/cache`)

**In Railway dashboard:**
1. Your service → **Volumes** tab → **Add Volume**
2. Set mount path: `/data`
3. Click **Create**

### Step 3 — Set environment variables

In **Variables** tab:

| Variable | Value |
|---|---|
| `PDF_FOLDER` | `/data/pdfs` |
| `CACHE_DIR` | `/data/cache` |
| `OPENAI_API_KEY` | `sk-proj-...` (if using OpenAI) |

### Step 4 — Copy PDFs into the Railway volume

Railway volumes are not accessible via SFTP or FTP directly. Use one of these methods:

#### Method A — Railway CLI (recommended)

```bash
# Install Railway CLI
npm install -g @railway/cli
# or: brew install railway

railway login
railway link   # select your project

# Open a shell in the running container
railway shell

# Inside the container shell:
mkdir -p /data/pdfs
# Then use the upload method below to get files there
```

#### Method B — Zip upload via Streamlit (one-time)

Add a temporary "admin" page to your app that accepts a zip file and extracts it:

```python
# In a temporary admin script (delete after use)
import streamlit as st, zipfile, io, pathlib
f = st.file_uploader("Upload PDFs as zip", type="zip")
if f:
    z = zipfile.ZipFile(io.BytesIO(f.read()))
    pathlib.Path("/data/pdfs").mkdir(parents=True, exist_ok=True)
    z.extractall("/data/pdfs")
    st.success(f"Extracted {len(z.namelist())} files")
```

Deploy this temporarily, upload your zip, then remove it.

#### Method C — GitHub release asset

1. Create a GitHub release and attach a `pdfs.zip` as a release asset
2. In your app startup or a Railway cron job, download and extract it:

```python
import urllib.request, zipfile, io, pathlib
url = "https://github.com/abocin/eo-policy-dashboard/releases/download/v1.0/pdfs.zip"
pathlib.Path("/data/pdfs").mkdir(parents=True, exist_ok=True)
data = urllib.request.urlopen(url).read()
zipfile.ZipFile(io.BytesIO(data)).extractall("/data/pdfs")
```

#### Method D — S3 / Google Drive sync

Store PDFs in S3 or Google Drive and add a sync step to your startup:

```bash
# With rclone (add to Procfile before streamlit command)
rclone copy gdrive:your-folder /data/pdfs && streamlit run app.py ...
```

#### Method E — Direct HTTP download

If your PDFs are accessible via URL, download them with Python in a startup script:

```python
import urllib.request, pathlib
urls = ["https://example.com/policy1.pdf", ...]
out = pathlib.Path("/data/pdfs")
out.mkdir(parents=True, exist_ok=True)
for url in urls:
    name = url.split("/")[-1]
    urllib.request.urlretrieve(url, out / name)
```

### Step 5 — Deploy

Railway redeploys automatically on every `git push`. The Procfile sets the start command:

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## Embedding Cache

The first time you run analysis on a corpus, sentence embeddings are computed (via OpenAI API or SBERT) and saved as `.npy` files under `$CACHE_DIR/embeddings/`.

On subsequent runs (or redeploys) the cached vectors are loaded from disk — **no API calls, ~10 seconds instead of ~5 minutes, $0 cost**.

Cache directory layout:
```
/data/cache/
  embeddings/     ← per-document, per-theme sentence vectors
  outputs/        ← auto-saved CSV/Excel/JSON exports
  uploads/        ← optional browser-upload staging
```

The sidebar shows cache status (persistent/ephemeral), number of cached documents, and disk usage. A "Clear embedding cache" button is provided for when you change the taxonomy or update documents.

---

## Memory and Performance

| Mode | RAM usage | Speed | Cost |
|---|---|---|---|
| OpenAI-only (`use_sbert: false`) | ~200 MB | ~5 min / 30 PDFs (first run) | ~$0.05 |
| OpenAI + disk cache | ~200 MB | ~10 sec (subsequent runs) | $0 |
| SBERT-only (offline) | ~700 MB | ~10 min / 30 PDFs | $0 |
| Keyword-only | ~100 MB | ~1 min / 30 PDFs | $0 |

For Railway's default 512 MB RAM: use OpenAI-only mode (`use_sbert: false` in taxonomy.yaml).
For Railway's 2–8 GB RAM (paid plan): SBERT mode works fine.

The pipeline processes **one document at a time** — peak memory is always 1 doc, not the full corpus.

---

## Troubleshooting

### Folder not found
```
❌ Folder not found: /data/pdfs
```
- Volume is not mounted — check Railway dashboard → Volumes
- Wrong path — confirm the mount point matches what you entered
- Fix: set `PDF_FOLDER=/data/pdfs` as env var so the sidebar pre-fills correctly

### Permission denied
```
❌ Permission denied reading: /data/pdfs
```
- The Railway volume may not be writable by the app user
- Fix: check volume mount permissions in Railway dashboard

### PDFs not visible after deployment
- The volume was freshly mounted — it is empty until you copy files into it
- See Step 4 above for methods to copy PDFs into `/data/pdfs`

### Volume not mounted correctly
- `cache_stats()` in the sidebar shows "🟡 Ephemeral" instead of "🟢 Persistent"
- Fix: verify the volume is mounted at `/data` in Railway dashboard, and set `CACHE_DIR=/data/cache`

### App restarts and cache disappears
- Without a volume, the container filesystem is ephemeral — everything is lost on restart
- Fix: add a Railway volume (Step 2 above) and set `CACHE_DIR=/data/cache`

### Memory errors during analysis
```
❌ Out of memory processing the corpus.
```
- You are using SBERT mode on a low-RAM plan
- Fix: set `use_sbert: false` in `config/taxonomy.yaml` (OpenAI-only mode uses ~200 MB)
- Alternative: upgrade Railway plan to 2+ GB RAM

### Streamlit timeout during long runs
- Large corpora (50+ docs) can take 10–30 minutes on first run
- Fix: `tcpProxyTimeout = 3600` is already set in `railway.toml`
- The embedding cache means subsequent runs are fast (~10 sec per doc)

### Invalid or scanned PDFs
- Scanned PDFs (image-only, no text layer) will extract 0 sentences
- The app logs a warning and skips them — they appear in the document count but have 0 excerpts
- Fix: run OCR (e.g. `ocrmypdf`) on scanned documents before adding to the corpus

### OpenAI API errors
```
AuthenticationError: Incorrect API key
```
- Check the `OPENAI_API_KEY` environment variable in Railway Variables tab
- The key must start with `sk-`

---

## Changelog

| Version | Description |
|---|---|
| 1.0.0 | Initial Streamlit dashboard — full pipeline from notebook v4.1 |
| 1.1.0 | Multilingual taxonomy — NL + EL keywords (289 total); paginated evidence cards; PDF + Excel evidence reports |
| 1.2.0 | OpenAI-only mode (`use_sbert: false`) — ~200 MB RAM, no torch; PT + IT keywords; Railway deployment |
| 1.3.0 | Disk embedding cache — OpenAI vectors persisted as `.npy` files; Railway volume support; 7× fewer API calls per document |
| 1.4.0 | Folder path ingestion — read PDFs directly from server disk; auto-save outputs to `/data/outputs`; production Railway setup; troubleshooting guide |
