import base64
import lancedb
import voyageai
import anthropic
import fitz

vo = voyageai.Client()
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
tbl = lancedb.connect("./semicon_db").open_table("chunks")



def retrieve(question, k=8):
    qvec = vo.multimodal_embed([question], model="voyage-multimodal-3", input_type="query").embeddings[0]
    hits = tbl.search(qvec).limit(k).to_list()
    return hits

def _image_bytes(hit):
    """Return (bytes, media_type) for an image hit, re-rendering from source."""
    path = hit["path"]
    if hit["page"] >= 0:  # image embedded in a PDF — render that page
        page = fitz.open(path)[hit["page"]]
        return page.get_pixmap(dpi=150).tobytes("png"), "image/png"
    ext = path.rsplit(".", 1)[-1].lower()
    media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
             "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return f.read(), media

def ask(question):
    hits = retrieve(question)
    text_hits  = [h for h in hits if h["content_type"] == "text"]
    image_hits = [h for h in hits if h["content_type"] == "image"]

    content = []

    if text_hits:
        text_context = "\n\n---\n\n".join(
            f"[source: {h['source']}]\n{h['text']}" for h in text_hits
        )
        content.append({"type": "text", "text": f"Context (text):\n{text_context}"})

    for h in image_hits:
        img_bytes, media_type = _image_bytes(h)
        b64 = base64.standard_b64encode(img_bytes).decode()
        content.append({"type": "text", "text": f"[image from: {h['source']}, page {h['page']}]"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })

    content.append({"type": "text", "text": f"Question: {question}"})

    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return msg.content[0].text

if __name__ == "__main__":
    import sys
    print(ask(" ".join(sys.argv[1:])))
