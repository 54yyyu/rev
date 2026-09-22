"""Images in a state: where they are found, what the text says instead, and
what reaches the template. No model and no server; the served model's
tokenizer is fetched (templates only) if it is not cached."""
from rev.prompt import render, split_images

fail = 0
def check(cond, what):
    global fail
    if not cond:
        print("  FAIL:", what); fail += 1

RED = "data:image/png;base64,iVBORw0KGgo="
part = lambda url: {"type": "image_url", "image_url": {"url": url}}

# Anywhere in a structured state, in document order; the text keeps a name for each.
s, urls = split_images({"ticket": "Screen is blank", "shots": [part(RED), part("https://x/y.png")]})
check(urls == [RED, "https://x/y.png"], f"urls {urls}")
check(s == {"ticket": "Screen is blank", "shots": ["Picture 1", "Picture 2"]}, f"state {s}")

# A chat message's content parts become one text.
s, urls = split_images([{"type": "text", "text": "Which colour?"}, part(RED)])
check(s == "Which colour?\n\nPicture 1" and urls == [RED], f"parts {s!r}")

# The bare string form of image_url is accepted too.
check(split_images({"type": "image_url", "image_url": RED})[1] == [RED], "string image_url")

# No images: the state is untouched, including a list of plain parts.
for st in ("plain", {"a": [1, 2]}, [{"type": "text", "text": "hi"}]):
    check(split_images(st) == (st, []), f"untouched {st!r}")

# A path would be read from the model server's own disk.
for bad in ("/etc/passwd", "file:///etc/passwd", "data:text/html,x"):
    try:
        split_images(part(bad)); check(False, f"accepted {bad}")
    except ValueError:
        pass

try:
    from rev.remote import DEFAULT_TOKENIZER, load_tokenizer
    tok = load_tokenizer(DEFAULT_TOKENIZER)
except Exception as e:                                   # noqa: BLE001
    print(f"SKIP render checks: no tokenizer ({e})")
else:
    labelled = [("A", "red"), ("B", "blue")]
    with_img = render(tok, "Picture 1", "Colour?", labelled, images=2)
    check(with_img.count("<|image_pad|>") == 2, "one placeholder per image")
    check("Picture 1: <|vision_start|>" in with_img and "Picture 2: " in with_img, "numbered images")
    check(with_img.rstrip().endswith("</think>"), "thinking closed")
    # Without images the prompt is byte-for-byte what it was.
    text = render(tok, "x", "Colour?", labelled)
    check("<|vision_start|>" not in text and text == tok.apply_chat_template(
        [{"role": "system", "content": __import__("rev.prompt").prompt.SYSTEM},
         {"role": "user", "content": __import__("json").dumps(
             {"evidence": "x", "criterion": "Colour?", "options": [
                 {"letter": "A", "description": "red"}, {"letter": "B", "description": "blue"}]},
             ensure_ascii=False)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False), "text prompt unchanged")

print("FAILURES:", fail)
raise SystemExit(1 if fail else 0)
