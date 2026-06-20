import os
import glob
import uuid
import io
import base64

import lancedb
import fitz  # pymupdf
import voyageai
import anthropic
from PIL import Image

vo = voyageai.Client()          # reads VOYAGE_API_KEY
claude = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
DB = lancedb.connect("./semicon_db")

EMBED_MODEL = "voyage-multimodal-3"   # embeds strings AND images in one space
VISION_MODEL = "claude-opus-4-8"      # extracts text from images at ingest

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}
# Allowlist of things safe to read as UTF-8 text. Anything else (e.g. raw
# OneNote .one files) is skipped instead of being read as garbage bytes.
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm", ".rtf"}

VISION_PROMPT = """You are extracting content from an image for a semiconductor research knowledge base.

1. Transcribe ALL visible text verbatim — every value in tables, axis labels, legends,
   footnotes, and annotations. Preserve tables as markdown tables with columns and units intact.
2. Then add a short technical description of any diagrams, charts, schematics, or roadmaps:
   what they show, the axes, the trend, and the key transitions.

Do NOT summarize away numbers — keep every data point. Mark anything you cannot read as [illegible]."""


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


def extract_image_text(img):
    """Use Claude vision to transcribe + describe an image. Returns text (or '')."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode()
    try:
        msg = claude.messages.create(
            model=VISION_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
        )
        return next((b.text for b in msg.content if b.type == "text"), "")
    except Exception as e:
        print(f"    vision extraction failed: {e}")
        return ""


def chunk(text, size=1200, overlap=200):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed(inputs, input_type="document"):
    # voyage-multimodal-3 accepts strings and PIL Images in the same call
    out = []
    for i in range(0, len(inputs), 128):
        batch = inputs[i:i + 128]
        out += vo.multimodal_embed(batch, model=EMBED_MODEL, input_type=input_type).embeddings
    return out


def add_text_rows(text, source, path, content_type, page):
    """Chunk + embed text and append rows. Used for documents and image transcripts."""
    cks = chunk(text)
    if not cks:
        return
    vecs = embed(cks)
    for c, v in zip(cks, vecs):
        rows.append({
            "id": str(uuid.uuid4()),
            "vector": v,
            "text": c,
            "source": source,
            "path": path,
            "content_type": content_type,
            "page": page,
        })


def add_image_row(img, source, path, page):
    """Embed the image natively (visual-similarity row, no text)."""
    rows.append({
        "id": str(uuid.uuid4()),
        "vector": embed([img])[0],
        "text": "",
        "source": source,
        "path": path,
        "content_type": "image",
        "page": page,
    })


rows = []
for path in glob.glob("./docs/**/*", recursive=True):
    if not os.path.isfile(path):
        continue

    ext = os.path.splitext(path)[1].lower()
    source = os.path.basename(path)

    if ext == ".pdf":
        # OneNote pages exported to PDF land here too.
        add_text_rows(load_text(path), source, path, "text", -1)
        for page_num, img in extract_pdf_images(path):
            add_image_row(img, source, path, page_num)               # visual row
            print(f"  vision-extracting image on {source} p{page_num} ...")
            add_text_rows(extract_image_text(img), source, path, "image_text", page_num)

    elif ext in IMAGE_EXTS:
        img = Image.open(path)
        add_image_row(img, source, path, -1)                         # visual row
        print(f"  vision-extracting {source} ...")
        add_text_rows(extract_image_text(img), source, path, "image_text", -1)

    elif ext in TEXT_EXTS:
        add_text_rows(load_text(path), source, path, "text", -1)

    else:
        # Unknown/binary — e.g. raw .one OneNote files. Export to PDF first.
        print(f"  SKIP (unsupported type {ext}): {source} — export to PDF first")

if not rows:
    raise SystemExit("No chunks produced. Put PDFs, images, or text files in ./docs.")

DB.drop_table("chunks", ignore_missing=True)
tbl = DB.create_table("chunks", data=rows)
n_img = sum(1 for r in rows if r["content_type"] == "image")
n_imgtext = sum(1 for r in rows if r["content_type"] == "image_text")
print(f"Indexed {len(rows)} chunks: {n_img} image vectors, {n_imgtext} image-transcript chunks, "
      f"{len(rows) - n_img - n_imgtext} text chunks.")
