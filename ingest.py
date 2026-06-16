import os
import glob
import uuid
import io

import lancedb
import fitz  # pymupdf
import voyageai
from PIL import Image

vo = voyageai.Client()  # reads VOYAGE_API_KEY
DB = lancedb.connect("./semicon_db")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}

def load_text(path):
    if path.endswith(".pdf"):
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_pdf_images(path):
    doc = fitz.open(path)
    for page_num, page in enumerate(doc):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:  # CMYK or similar — convert to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            yield page_num, Image.open(io.BytesIO(pix.tobytes("png")))

def chunk(text, size=1200, overlap=200):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]

def embed(inputs, input_type="document"):
    # voyage-multimodal-3 accepts strings and PIL Images in the same call
    out = []
    for i in range(0, len(inputs), 128):
        batch = inputs[i:i+128]
        out += vo.multimodal_embed(batch, model="voyage-multimodal-3", input_type=input_type).embeddings
    return out

rows = []
for path in glob.glob("./docs/**/*", recursive=True):
    if not os.path.isfile(path):
        continue

    ext = os.path.splitext(path)[1].lower()
    source = os.path.basename(path)

    # text chunks from non-image files
    if ext not in IMAGE_EXTS:
        text = load_text(path)
        cks = chunk(text)
        if cks:
            vecs = embed(cks)
            for c, v in zip(cks, vecs):
                rows.append({
                    "id": str(uuid.uuid4()),
                    "vector": v,
                    "text": c,
                    "source": source,
                    "path": path,
                    "content_type": "text",
                    "page": -1,
                })

    # standalone image files
    if ext in IMAGE_EXTS:
        img = Image.open(path)
        vec = embed([img])[0]
        rows.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "text": "",
            "source": source,
            "path": path,
            "content_type": "image",
            "page": -1,
        })

    # images embedded inside PDFs
    if ext == ".pdf":
        for page_num, img in extract_pdf_images(path):
            vec = embed([img])[0]
            rows.append({
                "id": str(uuid.uuid4()),
                "vector": vec,
                "text": "",
                "source": source,
                "path": path,
                "content_type": "image",
                "page": page_num,
            })

DB.drop_table("chunks", ignore_missing=True)
tbl = DB.create_table("chunks", data=rows)
print(f"Indexed {len(rows)} chunks from your docs.")
