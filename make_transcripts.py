import re
import shutil
import tempfile
from pathlib import Path

from qa_engine import DocQA

SAMPLES = ["photosynthesis.txt", "machine_learning.txt", "solar_energy.txt"]
QUESTIONS = [
    ("Grounded", "Why do leaves look green?"),
    ("Grounded", "How does a solar panel make electricity?"),
    ("Grounded", "What is overfitting?"),
    ("Grounded", "definition of photosynthesis"),
    ("Unsupported", "Who won the 2018 World Cup?"),
    ("Unsupported", "What is the capital of Australia?"),
]

tmp = Path(tempfile.mkdtemp())
for name in SAMPLES:
    shutil.copy(Path("corpus") / name, tmp / name)
qa = DocQA(str(tmp))

lines = []
for kind, q in QUESTIONS:
    r = qa.ask(q)
    lines += ["### " + kind + ": " + q, "", "**Answer:** " + r["answer"], ""]
    if r["grounded"]:
        s = r["source"]
        lines += ["**Source:** `" + s["doc"] + "`, chunk " + str(s["id"]) +
                  " (score " + format(r["score"], ".2f") + ")", "",
                  "> " + s["passage"], ""]
    else:
        lines += ["No supporting passage. Best match score " +
                  format(r["score"], ".2f") +
                  " is below the threshold, so nothing is generated.", ""]

block = ("<!-- TRANSCRIPTS START -->\n" + "\n".join(lines) +
         "\n<!-- TRANSCRIPTS END -->")
readme = Path("README.md").read_text(encoding="utf-8")
readme = re.sub(r"<!-- TRANSCRIPTS START -->.*?<!-- TRANSCRIPTS END -->",
                lambda m: block, readme, flags=re.S)
Path("README.md").write_text(readme, encoding="utf-8")
print("README.md updated with", len(QUESTIONS), "transcripts")