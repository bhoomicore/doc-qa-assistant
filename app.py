import html
import streamlit as st
import json
import wave
from pathlib import Path

import numpy as np
from vosk import KaldiRecognizer, Model, SetLogLevel
from collections import Counter
from qa_engine import DocQA, tokenize, split_sentences

st.set_page_config(page_title="NeonDoc Quest", page_icon="⚡", layout="wide")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700&family=Outfit:wght@400;500&family=Press+Start+2P&display=swap');
.stApp{background-color:#f3ff1f;background-image:linear-gradient(rgba(26,15,61,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(26,15,61,.07) 1px,transparent 1px);background-size:32px 32px;font-family:'Outfit',sans-serif}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:900px;padding-top:2rem}
[data-testid="stSidebar"]{background:#15102e;border-right:3px solid #1a0f3d}
[data-testid="stSidebar"] *{color:#ece9ff}
.title{font:700 clamp(30px,6vw,46px) 'Chakra Petch',sans-serif;color:#1a0f3d;text-shadow:3px 3px 0 #ff2bd6;margin-bottom:14px}
.hud{display:flex;align-items:center;gap:16px;flex-wrap:wrap;background:#15102e;border:2px solid #00f0ff;border-radius:12px;padding:12px 16px;box-shadow:0 0 0 3px #1a0f3d,0 0 22px rgba(0,240,255,.5);margin-bottom:16px}
.lv{font:14px 'Press Start 2P',monospace;color:#b6ff3b;text-shadow:0 0 8px #b6ff3b}
.xp{flex:1;min-width:150px}
.xpt{font-size:13px;color:#a39dcf;display:flex;justify-content:space-between}
.bar{height:12px;border-radius:6px;background:#241b4d;overflow:hidden;margin-top:3px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#00f0ff,#ff2bd6);box-shadow:0 0 12px #ff2bd6}
.stat{font:700 15px 'Chakra Petch',sans-serif;color:#f3ff1f;text-shadow:0 0 8px rgba(243,255,31,.7)}
.box{background:#15102e;border:2px solid #1a0f3d;border-radius:12px;padding:16px;box-shadow:0 0 0 2px #2f2560,0 6px 0 #1a0f3d;margin-top:16px}
.stApp .box p{color:#ece9ff;margin:0}
.stApp .box p.hint{color:#a39dcf;font-size:14px;margin-top:4px}
.h{font:700 17px 'Chakra Petch',sans-serif;color:#ff2bd6;text-shadow:0 0 8px rgba(255,43,214,.6);margin:14px 0 8px}
.doc{display:flex;justify-content:space-between;gap:8px;padding:7px 10px;border:1px solid #2f2560;border-radius:6px;margin-bottom:7px;font-size:14px}
.quest{display:flex;gap:10px;align-items:flex-start;font-size:14px;margin-bottom:8px;opacity:.7}
.quest b{flex:none;width:20px;height:20px;border:2px solid #2f2560;border-radius:5px;display:grid;place-items:center;font-size:13px}
.quest.done{opacity:1}
.quest.done b{background:#b6ff3b;border-color:#b6ff3b;color:#1a0f3d;box-shadow:0 0 10px #b6ff3b}
.badges{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.bd{text-align:center;font-size:12px;filter:grayscale(1);opacity:.4}
.bd span{display:block;font-size:26px}
.bd.on{filter:none;opacity:1}
.ans{border-left:5px solid #b6ff3b}
.ans.nf{border-left-color:#ff4d6d}
.tag{font:700 14px 'Chakra Petch',sans-serif;color:#b6ff3b;margin-bottom:6px;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.nf .tag{color:#ff4d6d}
.gain{background:#f3ff1f;color:#1a0f3d;border-radius:999px;padding:1px 10px;box-shadow:0 0 12px #f3ff1f}
.stApp .box p.big{font-size:18px}
.src{margin-top:14px;border:1px solid #ff2bd6;border-radius:8px;padding:14px;box-shadow:0 0 14px rgba(255,43,214,.25)}
.src .sh{font:700 15px 'Chakra Petch',sans-serif;color:#ff2bd6;margin-bottom:8px;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.src .sh small{color:#a39dcf;font:400 13px 'Outfit',sans-serif}
.stApp .box .src p{color:#d4cff7}
mark{background:rgba(0,240,255,.18);color:#00f0ff;border-radius:3px;padding:0 2px}
.best{background:rgba(182,255,59,.14);box-shadow:inset 0 -2px 0 #b6ff3b}
.rh{font:700 15px 'Chakra Petch',sans-serif;color:#00f0ff;margin-bottom:8px}
.row{display:grid;grid-template-columns:130px 1fr 44px;gap:10px;align-items:center;font-size:13px;color:#a39dcf;margin-bottom:6px}
.row .bar{height:6px;margin:0}
[data-testid="stForm"]{border:none;padding:0}
.stTextInput [data-baseweb="input"]{background:#0d0a22;border:2px solid #00f0ff;border-radius:8px;box-shadow:0 0 14px rgba(0,240,255,.35)}
.stTextInput input{background:transparent;color:#ece9ff}
.stButton button{background:#15102e;color:#ece9ff;border:1px solid #2f2560;border-radius:999px;font-size:14px;padding:4px 14px}
.stButton button p{color:inherit}
.stButton button:hover{border-color:#f3ff1f;color:#f3ff1f;background:#15102e}
div[data-testid="stFormSubmitButton"] button{background:#f3ff1f;color:#1a0f3d;border:2px solid #1a0f3d;border-radius:8px;font:700 16px 'Chakra Petch',sans-serif;box-shadow:0 4px 0 #ff2bd6;width:100%}
div[data-testid="stFormSubmitButton"] button:hover{background:#f3ff1f;color:#1a0f3d;border-color:#1a0f3d}
[data-testid="stFileUploaderDropzone"]{background:#0d0a22;border:2px dashed #00f0ff;border-radius:8px}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *{background:#f3ff1f;color:#1a0f3d;font-weight:700}
[data-testid="stSidebar"] .stButton button{padding:4px 8px;border-radius:8px;border-color:#ff4d6d;background:#15102e}
[data-testid="stAudioInput"]{border:2px solid #1a0f3d;border-radius:8px;box-shadow:0 4px 0 #ff2bd6}
[data-testid="stAudioInput"] [data-testid="stWidgetLabel"]{display:inline-flex;width:auto;background:#15102e;border:1px solid #00f0ff;border-radius:999px;padding:2px 14px;box-shadow:0 0 12px rgba(0,240,255,.5);margin-bottom:6px}
[data-testid="stAudioInput"] [data-testid="stWidgetLabel"] *{color:#00f0ff !important;font-family:'Chakra Petch',sans-serif;font-weight:700}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


from pathlib import Path
from pypdf import PdfReader

CORPUS = Path("corpus")
CORPUS.mkdir(exist_ok=True)


def pdf_to_text(file):
    reader = PdfReader(file)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages).replace("-\n", "")


def delete_doc(name):
    (CORPUS / name).unlink(missing_ok=True)
    st.cache_resource.clear()
    st.session_state.last = None
    st.session_state.setdefault("toasts", []).append("Deleted " + name)


if "up_key" not in st.session_state:
    st.session_state.up_key = 0

vault_box = st.sidebar.container()
upload_box = st.sidebar.container()

with upload_box:
    st.markdown('<div class="h">Add documents</div>', unsafe_allow_html=True)
    new_files = st.file_uploader(
        "Upload .txt or .pdf files", type=["txt", "pdf"],
        accept_multiple_files=True,
        key="up" + str(st.session_state.up_key), label_visibility="collapsed")

if new_files:
    added, skipped = [], []
    for f in new_files:
        name = Path(f.name).name
        try:
            if name.lower().endswith(".pdf"):
                text = pdf_to_text(f)
            else:
                text = f.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        text = text.strip()
        if text:
            (CORPUS / (Path(name).stem + ".txt")).write_text(text, encoding="utf-8")
            added.append(name)
        else:
            skipped.append(name)
    st.session_state.up_key += 1
    st.cache_resource.clear()
    toasts = st.session_state.setdefault("toasts", [])
    if added:
        toasts.append("Added " + str(len(added)) + " document(s) to the vault")
    for name in skipped:
        toasts.append("Could not read " + name + " (scanned PDF?)")
    st.rerun()


@st.cache_resource
def load_engine():
    return DocQA("corpus")


qa = load_engine() if any(CORPUS.glob("*.txt")) else None
counts = Counter(c["doc"] for c in qa.chunks) if qa else Counter()

SetLogLevel(-1)


@st.cache_resource
def load_vosk():
    folder = next(Path("models").glob("vosk-model*"), None)
    return Model(str(folder)) if folder else None


def transcribe(audio):
    model = load_vosk()
    if model is None:
        return None
    audio.seek(0)
    with wave.open(audio, "rb") as wf:
        channels, rate = wf.getnchannels(), wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if channels > 1:
        frames = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)[:, 0].tobytes()
    rec = KaldiRecognizer(model, rate)
    rec.AcceptWaveform(frames)
    return json.loads(rec.FinalResult()).get("text", "").strip()

# ---------- game rules ----------
QUESTS = [
    ("Find 3 pieces of evidence", lambda s: s.found >= 3),
    ("Dodge 1 hallucination trap", lambda s: s.traps >= 1),
    ("Reach a 3 combo", lambda s: s.max_combo >= 3),
]
BADGES = [
    ("first", "⭐", "First evidence", lambda s: s.found >= 1),
    ("trap", "🛡️", "Trap dodger", lambda s: s.traps >= 1),
    ("combo", "🔥", "Combo x3", lambda s: s.max_combo >= 3),
    ("lv2", "🏆", "Level 2", lambda s: s.xp >= 50),
]
SAMPLES = (qa.suggestions if qa else []) + ["Who won the 2018 World Cup?"]

defaults = dict(xp=0, combo=0, max_combo=0, found=0, traps=0,
                badges=set(), quests=set(), last=None, toasts=[], q="")
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def handle(question):
    question = question.strip()
    if not question:
        return
    s = st.session_state
    old_level = s.xp // 50
    res = qa.ask(question)
    if res["grounded"]:
        s.combo += 1
        s.max_combo = max(s.max_combo, s.combo)
        s.found += 1
        gain = 10 + 5 * (s.combo - 1)
    else:
        s.combo = 0
        s.traps += 1
        gain = 15
    s.xp += gain
    for name, test in QUESTS:
        if name not in s.quests and test(s):
            s.quests.add(name)
            s.xp += 20
            s.toasts.append("Quest complete: " + name + " +20 XP")
    for key, icon, name, test in BADGES:
        if key not in s.badges and test(s):
            s.badges.add(key)
            s.toasts.append("Badge unlocked: " + name)
    if s.xp // 50 > old_level:
        s.toasts.append("Level up!")
    s.last = {"q": question, "res": res, "gain": gain}
    s.q = question


def highlight(text, q_terms, best):
    out = []
    for sent in split_sentences(text):
        words = []
        for w in sent.split(" "):
            safe = html.escape(w)
            hit = set(tokenize(w)) & q_terms
            words.append("<mark>" + safe + "</mark>" if hit else safe)
        h = " ".join(words)
        out.append('<span class="best">' + h + "</span>" if sent in best else h)
    return " ".join(out)


def result_html(last):
    res, gain = last["res"], last["gain"]
    rows = ""
    for x in res["retrieved"]:
        name = x["doc"].replace(".txt", "") + " #" + str(x["id"])
        pct = int(x["coverage"] * 100)
        rows += ('<div class="row"><span>' + html.escape(name) + '</span>'
                 '<div class="bar"><i style="width:' + str(pct) + '%"></i></div>'
                 '<span>' + format(x["coverage"], ".2f") + '</span></div>')
    ret = '<div class="box"><div class="rh">Top 3 retrieved chunks</div>' + rows + "</div>"
    gain_tag = '<span class="gain">+' + str(gain) + " XP</span>"
    if not res["grounded"]:
        card = ('<div class="box ans nf"><div class="tag"><span>Trap dodged</span>' + gain_tag + "</div>"
                '<p class="big">Information not found in provided document.</p>'
                '<p class="hint">Best match scored ' + format(res["score"], ".2f") +
                ", under the threshold, so nothing is made up.</p></div>")
    else:
        src = res["source"]
        passage = highlight(src["passage"], set(tokenize(last["q"])), res["answer"])
        card = ('<div class="box ans"><div class="tag"><span>Evidence found · score ' +
                format(res["score"], ".2f") + "</span>" + gain_tag + "</div>"
                '<p class="big">' + html.escape(res["answer"]) + "</p>"
                '<div class="src"><div class="sh">Supporting passage<small>' +
                html.escape(src["doc"]) + " · chunk " + str(src["id"]) + "</small></div>"
                "<p>" + passage + "</p></div></div>")
    return card + ret


s = st.session_state

# ---------- sidebar ----------
docs_html = "".join(
    '<div class="doc"><span>' + html.escape(d) + "</span><span>" + str(n) + " chunks</span></div>"
    for d, n in counts.items())
quest_html = ""
for name, _ in QUESTS:
    done = name in s.quests
    quest_html += ('<div class="quest' + (" done" if done else "") + '"><b>' +
                   ("✓" if done else "") + "</b><span>" + name + "</span></div>")
badge_html = ""
for key, icon, name, _ in BADGES:
    on = key in s.badges
    badge_html += '<div class="bd' + (" on" if on else "") + '"><span>' + icon + "</span>" + name + "</div>"
with vault_box:
    st.markdown('<div class="h">Document vault</div>', unsafe_allow_html=True)
    for d, n in counts.items():
        c1, c2 = st.columns([5, 1])
        c1.markdown('<div class="doc"><span>' + html.escape(d) + "</span><span>" +
                    str(n) + " chunks</span></div>", unsafe_allow_html=True)
        c2.button("🗑", key="del_" + d, on_click=delete_doc, args=(d,),
                  help="Delete " + d)
with st.sidebar:
    st.markdown('<div class="h">Quests</div>' + quest_html +
                '<div class="h">Badges</div><div class="badges">' + badge_html + "</div>",
                unsafe_allow_html=True)
    
# ---------- main ----------
level, prog = s.xp // 50 + 1, s.xp % 50
st.markdown(
    '<div class="title">NeonDoc Quest</div>'
    '<div class="hud"><span class="lv">LV ' + str(level) + "</span>"
    '<div class="xp"><div class="xpt"><span>XP</span><span>' + str(prog) + " / 50</span></div>"
    '<div class="bar"><i style="width:' + str(prog * 2) + '%"></i></div></div>'
    '<span class="stat">Combo ' + str(s.combo) + "</span>"
    '<span class="stat">Evidence ' + str(s.found) + "</span></div>",
    unsafe_allow_html=True)

if qa is None:
    st.info("The vault is empty. Add a .txt or .pdf document in the sidebar to start.")
    st.stop()
    
def on_voice():
    audio = st.session_state.get("voice")
    if audio is None:
        return
    try:
        text = transcribe(audio)
    except Exception:
        text = ""
    if text is None:
        st.session_state.toasts.append("Voice model not found in the models folder.")
    elif not text:
        st.session_state.toasts.append("Could not hear a question. Try again.")
    else:
        handle(text)


with st.form("ask_form", border=False):
    c1, c2 = st.columns([5, 1])
    c1.text_input("Question", key="q", placeholder="Ask the documents a question",
                  label_visibility="collapsed")
    c2.form_submit_button("Ask", on_click=lambda: handle(st.session_state.q))

st.audio_input("🎤 Ask by voice (works offline)", key="voice", on_change=on_voice)
    
cols = st.columns(3)
for i, text in enumerate(SAMPLES):
    cols[i % 3].button(text, key="sample" + str(i), on_click=handle, args=(text,))

if s.last:
    st.markdown(result_html(s.last), unsafe_allow_html=True)
else:
    st.markdown("Pick a sample question or type your own. Find evidence, dodge traps, level up.")

for t in s.toasts:
    st.toast(t, icon="⚡")
if any(t.startswith("Level up") for t in s.toasts):
    st.balloons()
s.toasts = []