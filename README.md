# 🛰️ EO Policy Skills Dashboard

A Streamlit web application that analyses policy documents for evidence of **space industry and Earth Observation downstream skills needs**.

Built as a production-ready refactor of the `pilot_policy_hybrid_pipeline_v4.1` Jupyter notebook, adapted for policy-to-skills evidence mining rather than pilot-to-policy matching.

---

## What It Does

Upload PDF policy documents → the app automatically:

1. Extracts and cleans text (per-page, with page number tracking)
2. Splits into sentences for precise excerpt extraction
3. Runs **keyword search** against the EO skills taxonomy
4. Runs **semantic search** (SBERT, offline, no API needed)
5. Refines results with a **CrossEncoder reranker**
6. Optionally re-ranks with **OpenAI embeddings** (API key optional, via env var only)
7. Classifies each excerpt: `VALID EVIDENCE` / `WEAK EVIDENCE` / `NOT RELEVANT`
8. Lets you **human-validate** each excerpt in the dashboard
9. Exports to **CSV** (Power BI), **Excel**, **JSON** (D3/network), and **Markdown report**

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

## Project Structure

```
eo_policy_dashboard/
├── app.py                          # Main Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
│
├── core/                           # All business logic (no Streamlit)
│   ├── pdf_extractor.py            # PDF → DocumentContent (pdfminer + PyPDF2 fallback)
│   ├── chunker.py                  # Document → sentences + sliding-window chunks
│   ├── search_engine.py            # Keyword + SBERT + CrossEncoder + OpenAI pipeline
│   ├── pipeline.py                 # Pipeline orchestrator (called from app.py)
│   ├── taxonomy_loader.py          # Load & validate taxonomy YAML
│   ├── cache_manager.py            # Session + disk caching
│   └── exporters.py                # CSV / Excel / JSON / Markdown exports
│
├── pages/                          # Streamlit UI components
│   ├── results_table.py            # Filtered evidence card view
│   ├── charts.py                   # Plotly visualisations (5 chart types)
│   └── human_validation.py         # Per-excerpt labelling interface
│
├── config/
│   └── taxonomy.yaml               # EO skills search taxonomy (editable)
│
├── tests/
│   └── test_pipeline.py            # pytest unit tests (no Streamlit required)
│
└── .streamlit/
    ├── config.toml                 # Dark theme + upload size settings
    └── secrets.toml.example        # Template — copy to secrets.toml, never commit
```

---

## Quick Start — Local

### 1. Clone / copy the project

```bash
git clone <your-repo-url>
cd eo_policy_dashboard
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run** will download the SBERT model (~85 MB) and CrossEncoder (~85 MB) from Hugging Face. Subsequent runs use the local cache.

### 4. (Optional) Set your OpenAI key

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."

# Or create .streamlit/secrets.toml (copy from .streamlit/secrets.toml.example)
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Deployment

### Streamlit Community Cloud (free tier)

1. Push your code to a **public** GitHub repository  
   (make sure `.streamlit/secrets.toml` is in `.gitignore`)

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**

3. Select your repo, branch, and set the main file to `app.py`

4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."   # optional
   ```

5. Click **Deploy** — the app is live in ~2 minutes

> **Note on model size:** Streamlit Community Cloud has a 1 GB RAM limit. The default SBERT model (`all-MiniLM-L6-v2`) is well within this. The CrossEncoder is lightweight too. Do not use `all-mpnet-base-v2` on the free tier.

---

### Railway (recommended for production)

Railway auto-detects Python via Railpack — no Nixpacks or Dockerfile needed.

#### 1. Push to GitHub and create a Railway project

```bash
git push origin main
```

In the [Railway dashboard](https://railway.app): **New project → Deploy from GitHub repo** → select `eo-policy-dashboard`.

#### 2. Set environment variables

In **Variables** (Railway dashboard):

| Variable | Value | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | Required for OpenAI-only mode |
| `CACHE_DIR` | `/data` | Set this **after** adding a volume (see below) |

#### 3. (Recommended) Add a persistent volume for embedding cache

Without a volume, embedding vectors are recomputed from scratch on every redeploy (~5 min, ~$0.05). With a volume, cached embeddings survive redeploys — cache hits take ~10 seconds and cost $0.

**Steps:**

1. In Railway dashboard → your service → **Volumes** tab → **Add Volume**
2. Set mount path: `/data`
3. Go to **Variables** and add: `CACHE_DIR = /data`
4. Redeploy the service

The cache manager resolves the directory automatically:
- `CACHE_DIR` env var → highest priority (Railway volume)
- `/data` if it exists and is writable → Railway default volume path
- `.cache` → local ephemeral fallback

Embedding files are stored as `{file_hash}_{theme_slug}.npy` under `$CACHE_DIR/embeddings/`. The sidebar shows cache status (persistent vs. ephemeral), number of cached documents, and total disk size.

#### 4. Deploy

Railway deploys automatically on every `git push`. The Procfile configures the start command:

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

### Hugging Face Spaces (free, GPU optional)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
   - SDK: **Streamlit**
   - Hardware: CPU Basic (free) or upgrade for faster inference

2. In your Space's **Files**, upload all project files maintaining the folder structure

3. Add a `packages.txt` file if you need system-level dependencies (typically not needed)

4. Set your OpenAI key under **Settings → Repository secrets** as `OPENAI_API_KEY`

5. The Space builds automatically and is accessible at `https://huggingface.co/spaces/<your-username>/<space-name>`

---

### Render (free or paid)

1. Create a `render.yaml` in the project root:

```yaml
services:
  - type: web
    name: eo-policy-dashboard
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: OPENAI_API_KEY
        sync: false    # set manually in Render dashboard
```

2. Push to GitHub and connect the repo in [render.com](https://render.com)

3. Set `OPENAI_API_KEY` in Render's environment variables

---

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]
```

```bash
docker build -t eo-policy-dashboard .
docker run -p 8501:8501 -e OPENAI_API_KEY="sk-..." eo-policy-dashboard
```

---

## Configuration

### Editing the taxonomy

Edit `config/taxonomy.yaml` to:
- Add new skill themes
- Add or remove keywords
- Adjust semantic search queries
- Tune scoring thresholds

You can also upload a custom YAML directly in the sidebar at runtime — no restart needed.

### Tuning thresholds

| Parameter | Default | Effect |
|---|---|---|
| `valid_match` | 0.50 | Final score ≥ this → VALID EVIDENCE |
| `weak_match` | 0.35 | Final score ≥ this → WEAK EVIDENCE |
| `top_k_sentences` | 5 | Max sentences returned per theme per doc |

The sidebar sliders override these at runtime.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover: text cleaning, sentence splitting, chunking, keyword search, classification thresholds, taxonomy loading, and export formatting. All tests run without GPU or API keys.

---

## Security Notes

- **API keys are never hard-coded.** The OpenAI key is read only from `os.environ["OPENAI_API_KEY"]` or Streamlit secrets. The sidebar input stores it temporarily in the session environment.
- **PDFs are processed locally.** No document content is sent to any external service unless you opt in to OpenAI re-ranking.
- The old notebook contained a hard-coded key (`sk-proj-rk4...`) — this has been removed. Rotate that key immediately in your OpenAI dashboard.
- `.streamlit/secrets.toml` is in `.gitignore`. Never commit it.

---

## Power BI Integration

See the "Power BI Integration" section at the bottom of this README for a full step-by-step guide.

### Quick steps:
1. Export **CSV (Power BI)** from the Export tab
2. In Power BI Desktop: **Get Data → Text/CSV** → select the exported file
3. Recommended measures:
   - `Valid Evidence Count = COUNTROWS(FILTER(Table, [Validation Category] = "VALID EVIDENCE"))`
   - `Avg Score by Theme = AVERAGEX(FILTER(Table, [Theme] = SELECTEDVALUE(Theme[Theme])), [Final Score])`
4. Use **Document**, **Theme**, and **Validation Category** as slicers
5. Build a matrix visual with Document on rows and Theme on columns, values = count of excerpts

---

## Changelog

| Version | Description |
|---|---|
| 1.0.0 | Initial Streamlit dashboard — full pipeline from notebook v4.1 |
| — | Added EO skills taxonomy replacing pilot domain map |
| — | Added human validation UI with batch labelling |
| — | Added Plotly heatmap, score distribution, theme stacked bar |
| — | Added Excel multi-sheet export + D3 network JSON |
| — | Removed hard-coded API key |
| 1.1.0 | Multilingual taxonomy — NL + EL keywords (289 total); paginated evidence cards; PDF + Excel evidence reports |
| 1.2.0 | OpenAI-only mode (`use_sbert: false`) — ~200MB RAM, no torch; PT + IT keywords; Railway deployment |
| 1.3.0 | Disk embedding cache — OpenAI vectors persisted as `.npy` files keyed by file hash; Railway volume mount support (`CACHE_DIR` env var); cache stats in sidebar; clear cache button |
