# Power BI Integration Guide

This guide explains how to connect the EO Policy Skills Dashboard export to
Power BI for advanced reporting and sharing.

---

## Step 1 — Export the CSV

In the dashboard **Export** tab, click **Download CSV (Power BI)**.

The file has these columns:

| Column | Type | Notes |
|---|---|---|
| Document | Text | PDF filename |
| Page | Integer | Source page number |
| Theme | Text | EO skills theme label |
| Excerpt | Text | Full sentence from the document |
| Keyword Hit | Boolean | Was this triggered by a keyword? |
| Matched Keyword | Text | The triggering keyword (if keyword hit) |
| SBERT Score | Decimal | Cosine similarity (0–1) |
| CrossEncoder Score | Decimal | CrossEncoder reranker score |
| OpenAI Score | Decimal | Optional OpenAI re-rank score (0 if not used) |
| Final Score | Decimal | Weighted blend score (primary sorting column) |
| Validation Category | Text | VALID EVIDENCE / WEAK EVIDENCE / NOT RELEVANT |
| Human Label | Text | Your manual label (if any) |

---

## Step 2 — Load into Power BI Desktop

1. Open Power BI Desktop
2. **Home → Get Data → Text/CSV**
3. Select your exported CSV file
4. In the preview: set **Delimiter** to Comma, **Data type detection** to "Based on first 200 rows"
5. Click **Load** (or **Transform Data** if you want to clean first)

---

## Step 3 — Recommended Data Model

### Calculated measures (DAX)

```dax
-- Count of valid evidence excerpts
Valid Count =
CALCULATE(
    COUNTROWS('eo_policy_evidence'),
    'eo_policy_evidence'[Validation Category] = "VALID EVIDENCE"
)

-- Average final score (valid + weak only)
Avg Relevant Score =
CALCULATE(
    AVERAGEX('eo_policy_evidence', 'eo_policy_evidence'[Final Score]),
    'eo_policy_evidence'[Validation Category] IN {"VALID EVIDENCE", "WEAK EVIDENCE"}
)

-- Coverage: % of documents with at least one valid match per theme
Theme Coverage % =
DIVIDE(
    CALCULATE(
        DISTINCTCOUNT('eo_policy_evidence'[Document]),
        'eo_policy_evidence'[Validation Category] = "VALID EVIDENCE"
    ),
    DISTINCTCOUNT('eo_policy_evidence'[Document])
) * 100

-- Human-validated match rate
Human Validation Rate =
DIVIDE(
    COUNTROWS(FILTER('eo_policy_evidence', 'eo_policy_evidence'[Human Label] <> "")),
    COUNTROWS('eo_policy_evidence')
) * 100
```

---

## Step 4 — Recommended Visuals

### Dashboard page 1: Overview

| Visual | Fields |
|---|---|
| Card | Valid Count |
| Card | Avg Relevant Score |
| Donut chart | Validation Category (count) |
| Bar chart | Document (axis), Valid Count (value) |

### Dashboard page 2: Theme analysis

| Visual | Fields |
|---|---|
| Matrix | Document (rows), Theme (columns), Valid Count (values) — this replicates the heatmap |
| Stacked bar | Document (axis), Theme (legend), COUNTROWS (value) |
| Scatter | Final Score (x), CrossEncoder Score (y), Theme (colour) |

### Dashboard page 3: Evidence browser

| Visual | Fields |
|---|---|
| Table | Document, Page, Theme, Final Score, Validation Category, Excerpt |
| Slicers | Document, Theme, Validation Category, Human Label |
| Text filter | Excerpt (use Q&A or a search slicer) |

---

## Step 5 — Scheduled refresh (if publishing to Power BI Service)

If you publish the report to Power BI Service and want it to update
when you run new analyses:

1. Store the CSV in a **OneDrive for Business** or **SharePoint** folder
2. In Power BI Desktop, use **Get Data → SharePoint folder** instead of local CSV
3. After publishing, configure scheduled refresh in the Power BI Service dataset settings
4. Re-export from the dashboard to the same SharePoint location after each analysis run

---

## Step 6 — D3 Network JSON in Power BI

Power BI does not natively render D3 graphs, but you can:

1. Use the **HTML Viewer** custom visual (from AppSource) to embed a D3 network diagram
2. Or export the JSON and use it in an Observable notebook / separate web page
3. Or use the **Charticulator** custom visual for custom network layouts

The JSON structure is:
```json
{
  "network_data": {
    "nodes": [
      {"id": "policy_doc.pdf", "group": "document"},
      {"id": "EO Downstream Skills", "group": "theme"}
    ],
    "links": [
      {"source": "policy_doc.pdf", "target": "EO Downstream Skills", "value": 0.72}
    ]
  }
}
```

This is directly compatible with D3 force-directed graph examples at
[observablehq.com](https://observablehq.com/@d3/force-directed-graph).
