"""Copy the speaker notes from slides.md into speaker_notes.md, one section per slide, with a time per slide.

Run it from the repository folder after changing the slides: uv run python presentation/make_notes.py
"""

import re
from pathlib import Path

HERE = Path(__file__).parent
SECONDS = [15, 25, 30, 25, 30, 20, 30, 25, 30, 25, 30, 40, 15, 0]  # per slide; the last one is a backup slide

text = (HERE / "slides.md").read_text(encoding="utf-8")
slides = re.split(r"\n---\n", text.split("\n---\n", 1)[1])  # drop the front matter

lines = ["# Speaker Notes: Which Flower Is It?", "", "Presented by Amir · about 5 minutes", ""]
elapsed = 0
for number, (slide, seconds) in enumerate(zip(slides, SECONDS), start=1):
    title = re.search(r"^#{1,2} (.+)$", slide, re.MULTILINE).group(1)
    notes = [
        note.strip()
        for note in re.findall(r"<!--(.*?)-->", slide, re.DOTALL)
        if not note.strip().startswith("_")  # skip Marp settings such as _paginate
    ]
    elapsed += seconds
    timing = f"{seconds} s, until {elapsed // 60}:{elapsed % 60:02d}" if seconds else "only if asked"
    lines += [f"## {number}. {title}", f"*{timing}*", ""] + notes + [""]

(HERE / "speaker_notes.md").write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {len(slides)} slides, total {elapsed // 60}:{elapsed % 60:02d}")
