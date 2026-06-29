"""
Temporary admin page — upload a zip of PDFs directly into the Railway volume.
DELETE THIS FILE after use. Do not leave it in production.
"""
import io
import pathlib
import zipfile

import streamlit as st

st.set_page_config(page_title="Admin: Upload PDFs", page_icon="🔒")
st.title("🔒 Admin: Upload PDF Zip to Volume")
st.warning("Delete this page (`pages/99_Admin_Upload.py`) after use.")

TARGET = pathlib.Path("/data/pdfs")
TARGET.mkdir(parents=True, exist_ok=True)

st.markdown(f"**Target folder:** `{TARGET}`")

existing = list(TARGET.glob("*.pdf"))
st.caption(f"{len(existing)} PDF(s) already in folder")

uploaded_zip = st.file_uploader("Upload a .zip file containing PDFs", type=["zip"])

if uploaded_zip:
    zf = zipfile.ZipFile(io.BytesIO(uploaded_zip.read()))
    pdf_names = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
    st.info(f"Found {len(pdf_names)} PDF(s) in zip")

    if st.button(f"Extract {len(pdf_names)} PDFs to {TARGET}", type="primary"):
        extracted = 0
        for name in pdf_names:
            # Strip any folder path inside the zip
            safe_name = pathlib.Path(name).name
            dest = TARGET / safe_name
            dest.write_bytes(zf.read(name))
            extracted += 1
        st.success(f"✅ Extracted {extracted} PDF(s) to `{TARGET}`")
        st.caption("You can now go back to the main app and use the folder path `/data/pdfs`.")
        if st.button("List files"):
            for p in sorted(TARGET.glob("*.pdf")):
                st.caption(f"• {p.name}")
