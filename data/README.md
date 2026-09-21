# `data/` — local development only

**Your live corpora are NOT here.** On Railway they live on the persistent
volume mounted at `/data`, which is separate storage and is not part of this
repository. This folder is `/app/data` inside the container — a different place.

That distinction matters: **committing PDFs here does not add them to the
deployed corpus.** They would be baked into the container image, ignored by the
app, and would bloat every build.

## Adding documents to a live corpus

Use the **🗂️ PDF File Manager** page in the running app:

- **🐙 Import from GitHub** — paste a repo folder or release URL and the server
  downloads every PDF (or a zip of PDFs) straight onto the volume. Release
  assets allow up to 2 GB each, versus a 25 MB cap on web uploads into a repo.
- **🌐 Add PDFs from URL** — individual PDF links, or one `.zip` that the server
  unpacks into the target folder.
- **📥 Copy PDFs between folders** — move documents between corpora.

## Corpus folders

Any subfolder of the volume is a corpus and is picked up automatically, so
`/data/pdfs`, `/data/destine`, `/data/trends` and any folder you create in the
app all appear in the folder selector with a live document count. Create them
from the sidebar or the File Manager, not from this repo.

Set `CORPUS_ROOT` to override the volume root (default `/data`) when running
locally.

## Local use

For local development, point the app at a folder on your own machine:

```bash
CORPUS_ROOT=./data streamlit run app.py
```

Then drop PDFs into `data/pdfs/`. Anything you put in this folder is ignored by
git except this README.
