import base64
import lancedb
import voyageai
import anthropic
import fitz

vo = voyageai.Client()
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
tbl = lancedb.connect("./semicon_db").open_table("chunks")

EMBED_MODEL = "voyage-multimodal-3"
ANSWER_MODEL = "claude-opus-4-8"
MAX_IMAGES = 4  # cap re-attached source images per answer (token/cost control)

SYSTEM = """You are a semiconductor industry research assistant.
Answer using ONLY the provided context chunks. Be technically precise and quantitative.
Cite sources inline as [source: filename]. If the context lacks the answer, say so explicitly
rather than guessing. Avoid generic AI-flavored language."""

def retrieve(question, k=8):
    qvec = vo.multimodal_embed([question], model=EMBED_MODEL, input_type="query").embeddings[0]
    return tbl.search(qvec).limit(k).to_list()

def _image_bytes(hit):
    """Return (bytes, media_type) for an image hit, re-rendering from source."""
    path = hit["path"]
    if hit["page"] >= 0:  # image came from a PDF page — render that page
        page = fitz.open(path)[hit["page"]]
        return page.get_pixmap(dpi=150).tobytes("png"), "image/png"
    ext = path.rsplit(".", 1)[-1].lower()
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
             "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return f.read(), media

def ask(question):
    hits = retrieve(question)
    # Text context: document chunks AND image transcripts.
    text_hits = [h for h in hits if h["content_type"] in ("text", "image_text")]
    # Images to re-attach: native image hits AND transcript hits (dedup by source image).
    image_hits = [h for h in hits if h["content_type"] in ("image", "image_text")]

    content = []

    if text_hits:
        text_context = "\n\n---\n\n".join(
            f"[source: {h['source']}]\n{h['text']}" for h in text_hits
        )
        content.append({"type": "text", "text": f"Context (text):\n{text_context}"})

    seen, attached = set(), 0
    for h in image_hits:
        if attached >= MAX_IMAGES:
            break
        key = (h["path"], h["page"])
        if key in seen:
            continue
        seen.add(key)
        try:
            img_bytes, media_type = _image_bytes(h)
        except Exception as e:
            print(f"(skipped image {h['source']}: {e})")
            continue
        b64 = base64.standard_b64encode(img_bytes).decode()
        content.append({"type": "text", "text": f"[image from: {h['source']}, page {h['page']}]"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })
        attached += 1

    content.append({"type": "text", "text": f"Question: {question}"})

    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return next((b.text for b in msg.content if b.type == "text"), "")

if __name__ == "__main__":
    import sys
    print(ask(" ".join(sys.argv[1:])))
