# Example Test Workflow — 5 Policy PDFs

This document walks through a complete end-to-end test of the dashboard using
five representative policy documents covering EU space, skills, and regional
development themes.

---

## Recommended Test Documents

| # | Document | Source | Why it's useful |
|---|---|---|---|
| 1 | EU Space Strategy for Security and Defence | European Commission, 2023 | Direct space policy — should have high EO Skills + Space Industry scores |
| 2 | European Skills Agenda 2023 Progress Report | DG EMPL | Skills gaps and workforce development focus — tests the "Skills Gaps" theme |
| 3 | Copernicus User Uptake Strategy | ESA / EUMETSAT | Copernicus-specific — tests the "Copernicus & EU Space Services" theme |
| 4 | Smart Specialisation Strategy — Nordrhein-Westfalen | Regional government | Regional S3 policy — should have moderate "Policy Support for Downstream" scores |
| 5 | Digital Europe Programme Work Programme 2023-24 | European Commission | Digital skills focus — tests "Digital Skills for Space" theme |

Download PDFs from official EU sources (EUR-Lex, ESA website, regional government portals).

---

## Step-by-Step Walkthrough

### Step 1 — Launch the dashboard

```bash
cd eo_policy_dashboard
streamlit run app.py
```

Open `http://localhost:8501`.

---

### Step 2 — Upload the five PDFs

In the sidebar, click **"Upload one or more PDFs"** and select all five files.

You should see them listed in the uploader widget.

---

### Step 3 — Configure thresholds

Set:
- **Valid evidence threshold**: 0.50 (default)
- **Weak evidence threshold**: 0.35 (default)

Leave the OpenAI key blank for this offline test.

---

### Step 4 — Run the analysis

Click **🔍 Run Analysis**.

Expected progress messages:
```
Extracting text from EU_Space_Strategy_2023.pdf…
Extracting text from European_Skills_Agenda_2023.pdf…
…
Running semantic search across ~8,000 sentences…
```

This takes approximately **2–5 minutes** on first run (SBERT model loads ~30s,
then sentence encoding for 5 documents).

On re-run the documents are cached — only the search re-runs.

---

### Step 5 — Review summary metrics

After analysis, check the top metric bar:

| Metric | Expected range |
|---|---|
| Documents | 5 |
| Total Excerpts | 50–300 (depends on doc length) |
| Valid Evidence | 20–80 |
| Weak Evidence | 10–50 |
| Themes Detected | 5–7 |

---

### Step 6 — Explore the Results tab

Use filters to explore:
- Filter **Document** → "Copernicus User Uptake Strategy.pdf"
- Filter **Theme** → "Copernicus & EU Space Services"
- Set **Minimum score** → 0.40

You should see excerpts about Copernicus services, downstream users,
and user uptake training.

---

### Step 7 — Review the Charts tab

Check:
- **Heatmap**: EU Space Strategy and Copernicus docs should show the darkest
  cells for EO Downstream Skills and Space Industry Skills
- **Score distribution**: a healthy run has most scores between 0.30–0.65,
  with a tail above 0.50 (valid evidence)
- **Themes per document**: Digital Europe PDF should have the highest
  "Digital Skills for Space" count

---

### Step 8 — Human validation

Go to the **Human Validation** tab.

- Filter to **"VALID EVIDENCE only"**
- Review the top 10 excerpts — check whether they genuinely reference EO or space skills needs
- Use the radio buttons to relabel any that are false positives as "Not relevant"
- Use **batch labelling** to mark all excerpts from the Skills Agenda PDF as "Valid evidence" if they clearly discuss workforce gaps

---

### Step 9 — Export

Go to the **Export** tab:

1. Download **CSV (Power BI)** — open in Excel to verify all columns are present
2. Download **Excel Workbook** — check the "Theme Summary" sheet for a pivot view
3. Download **D3 / Network JSON** — open in a text editor and verify the `network_data.nodes` array contains both document and theme nodes

---

### Step 10 — Taxonomy tuning

If you find the Copernicus theme is missing relevant excerpts:

1. Open `config/taxonomy.yaml`
2. Under the `Copernicus & EU Space Services` theme, add keywords like:
   ```yaml
   - Copernicus Land Monitoring
   - Copernicus Climate Change Service
   - Global Monitoring for Environment
   ```
3. Save the file and click **🔄 Clear & Reset** in the sidebar
4. Re-upload and re-run

---

## Expected Issues and Fixes

| Issue | Likely cause | Fix |
|---|---|---|
| "No text extracted" for a PDF | Scanned / image-only PDF | Use an OCR tool (e.g. Adobe Acrobat, `ocrmypdf`) before uploading |
| Very few results (<5 excerpts) | Thresholds too high or generic doc | Lower weak threshold to 0.25; check taxonomy keywords cover the document's vocabulary |
| SBERT model download stalls | Slow internet or HF rate limit | Pre-download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` |
| App crashes on large PDFs (>100 pages) | RAM limit on free cloud tier | Split the PDF before uploading, or increase Streamlit Cloud instance RAM |
