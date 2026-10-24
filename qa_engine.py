import difflib
import re
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stemmer = PorterStemmer()
STOP = set(ENGLISH_STOP_WORDS) | {"does", "did"}
THRESHOLD = 0.45
NOT_FOUND = "Information not found in provided document."
END = ".!?:"
BULLET = re.compile("[\uf000-\uf0ff\u2022\u2023\u2043\u25a0\u25a1\u25aa\u25cf\u25e6\u27a2\u27a4]")
DEF_WORDS = re.compile(r"defin|meaning|what is|what are|what do you mean")
DEF_SENT = re.compile(
    r"defined as|refers? to|is called|are called|known as|means|is the|is a |are the")
INTENT = {"definition", "define", "defination", "defintion", "meaning", "mean",
          "means", "explain", "explanation", "describe", "description", "tell",
          "give", "brief", "briefly", "detail", "details", "about", "please",
          "write", "note", "notes", "short"}
SMALL = {"of", "and", "the", "in", "for", "to", "a", "an", "on", "vs", "with",
         "or", "by", "at", "as"}
PRONOUNS = {"it", "this", "that", "there", "these", "those", "they", "he", "she",
            "which", "what", "when", "where", "who", "how", "also", "here"}
DEF_Q = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:The\s+|A\s+|An\s+)?"
    r"([A-Z][A-Za-z0-9\-]*(?:\s[A-Za-z0-9\-]+){0,3}?)\s(?:is|are)\s"
    r"(?:defined as|a|an|the|called|known as)\b")


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stemmer.stem(w) for w in words if w not in STOP]


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def lower_words(text):
    return " ".join(w if (w.isupper() and len(w) > 1) else w.lower()
                    for w in text.split())


def heading_text(sentence):
    """Return the heading text if this 'sentence' is really a heading."""
    s = re.sub(r"^\d+(?:\.\d+)*\s+", "", sentence.strip().rstrip("."))
    words = s.split()
    if not 1 <= len(words) <= 7 or len(s) < 4 or not words[0][0].isupper():
        return None
    for w in words:
        if re.search(r"\d", w):
            return None
        if not (w[0].isupper() or w.lower() in SMALL):
            return None
    return s


def clean_document(text):
    """Turn raw PDF/TXT text into clean sentences that end with punctuation."""
    paras, blocks, state = [], [], {"cur": ""}

    def close_block():
        cur = state["cur"].strip()
        if cur:
            blocks.append(cur if cur[-1] in END else cur.rstrip(",;") + ".")
        state["cur"] = ""

    def close_para():
        close_block()
        if blocks:
            paras.append(" ".join(blocks))
            blocks.clear()

    for raw in text.replace("\u00ad", "").splitlines():
        line = raw.strip()
        cur = state["cur"]
        if not line:
            if not cur or cur[-1] in END:
                close_para()
            continue
        if BULLET.match(line):
            close_block()
            state["cur"] = BULLET.sub(" ", line).strip()
            continue
        is_head = ((not cur or cur[-1] in END) and len(line.split()) <= 10
                   and line[-1] not in END
                   and (line[0].isdigit() or line[0].isupper()))
        if is_head:
            close_block()
            state["cur"] = line
            close_block()
            continue
        state["cur"] = (cur + " " + line).strip()
    close_para()
    return "\n\n".join(paras)


def chunk_text(text, max_words=80, min_words=25, hard=140):
    """Sentence-based chunks of about max_words words, with 1-sentence overlap."""
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        units = []
        for sent in split_sentences(para):
            w = sent.split()
            units += [" ".join(w[i:i + hard]) for i in range(0, len(w), hard)]
        current, count = [], 0
        for u in units:
            n = len(u.split())
            if current and count + n > max_words:
                chunks.append(" ".join(current))
                last = current[-1]
                if len(last.split()) <= 30:
                    current, count = [last], len(last.split())
                else:
                    current, count = [], 0
            current.append(u)
            count += n
        if current:
            chunks.append(" ".join(current))

    merged, carry = [], ""
    for c in chunks:
        c = (carry + " " + c).strip()
        if len(c.split()) < min_words:
            carry = c
        else:
            merged.append(c)
            carry = ""
    if carry:
        merged.append(carry)
    return merged


class DocQA:
    def __init__(self, corpus_dir="corpus"):
        self.chunks = []
        for path in sorted(Path(corpus_dir).glob("*.txt")):
            text = clean_document(path.read_text(encoding="utf-8"))
            for n, piece in enumerate(chunk_text(text), 1):
                self.chunks.append({"doc": path.name, "id": n, "text": piece})
        if not self.chunks:
            raise ValueError("No .txt files found in " + corpus_dir)

        texts = [c["text"] for c in self.chunks]
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                   lowercase=False)
        self.matrix = self.vec.fit_transform(texts)
        self.idf = dict(zip(self.vec.get_feature_names_out(), self.vec.idf_))
        self.max_idf = float(max(self.vec.idf_))
        self.chunk_terms = [set(tokenize(t)) for t in texts]
        self.chunk_bigrams = []
        for t in texts:
            toks = tokenize(t)
            self.chunk_bigrams.append(set(zip(toks, toks[1:])))
        self.chunk_sents = [split_sentences(t) for t in texts]
        self.sent_terms = [[set(tokenize(s)) for s in ss] for ss in self.chunk_sents]
        self.vocab = set(w for t in texts for w in re.findall(r"[a-z]{4,}", t.lower()))
        self.surface = {}
        for w in self.vocab:
            if len(w) >= 5:
                self.surface.setdefault(stemmer.stem(w), w)
        self.suggestions = self.suggest_questions()

    def clean_question(self, question):
        """Fix typos using words from the documents and drop filler words."""
        fixed = []
        for w in re.findall(r"[a-z0-9]+", question.lower()):
            if len(w) >= 6 and w not in self.vocab and w not in INTENT:
                m = difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.85)
                if m:
                    w = m[0]
            fixed.append(w)
        return " ".join(w for w in fixed if w not in INTENT)

    def coverage(self, q_terms, chunk_terms):
        total = sum(self.idf.get(t, self.max_idf) for t in q_terms)
        hit = sum(self.idf.get(t, 0) for t in q_terms if t in chunk_terms)
        return hit / total if total else 0.0

    def phrase_score(self, q_tokens, i):
        pairs = set(zip(q_tokens, q_tokens[1:]))
        if not pairs:
            return 0.0
        return len(pairs & self.chunk_bigrams[i]) / len(pairs)

    def ask(self, question, top_k=3):
        wants_def = bool(DEF_WORDS.search(question.lower()))
        clean = self.clean_question(question)
        q_tokens = tokenize(clean)
        q_terms = set(q_tokens)
        if not q_terms:
            return {"grounded": False, "answer": NOT_FOUND, "source": None,
                    "score": 0.0, "retrieved": []}

        sims = cosine_similarity(self.vec.transform([clean]), self.matrix)[0]
        need = max(1, round(0.6 * len(q_terms)))

        ranked = []
        for i in range(len(self.chunks)):
            cov = self.coverage(q_terms, self.chunk_terms[i])
            phrase = self.phrase_score(q_tokens, i)
            bonus = 0.0
            if wants_def:
                for s, st in zip(self.chunk_sents[i], self.sent_terms[i]):
                    if DEF_SENT.search(s.lower()) and len(q_terms & st) >= need:
                        bonus = 0.25
                        break
            rank = 0.5 * cov + 0.3 * float(sims[i]) + 0.2 * phrase + bonus
            ranked.append((rank, cov, float(sims[i]), i))
        ranked.sort(reverse=True)

        retrieved = []
        for rank, cov, sim, i in ranked[:top_k]:
            c = dict(self.chunks[i])
            c["index"] = i
            c["cosine"] = round(sim, 3)
            c["coverage"] = round(cov, 3)
            retrieved.append(c)

        top = retrieved[0]
        if top["coverage"] < THRESHOLD:
            return {"grounded": False, "answer": NOT_FOUND, "source": None,
                    "score": top["coverage"], "retrieved": retrieved}

        i = top["index"]
        own = len(self.chunk_sents[i])
        sents = list(self.chunk_sents[i])
        if i + 1 < len(self.chunks) and self.chunks[i + 1]["doc"] == top["doc"]:
            for s in self.chunk_sents[i + 1]:
                if s not in sents:
                    sents.append(s)

        def score(j):
            s = sents[j]
            hit = len(q_terms & set(tokenize(s)))
            return hit + (1.0 if wants_def and DEF_SENT.search(s.lower()) else 0.0)

        k = max(range(own), key=score)
        picked, words = [], 0
        for s in sents[k:]:
            picked.append(s)
            words += len(s.split())
            if len(picked) >= 5 or words >= 70:
                break
        answer = " ".join(picked)
        if answer[-1] not in ".!?:":
            answer += "."

        top["passage"] = " ".join(sents[:max(own, k + len(picked))])
        return {"grounded": True, "answer": answer, "source": top,
                "score": top["coverage"], "retrieved": retrieved}

    def suggest_questions(self, n=6):
        """Build starter questions from the documents; keep only answerable ones."""
        by_doc, count = {}, Counter()
        for c, sents in zip(self.chunks, self.chunk_sents):
            for s in sents:
                h = heading_text(s)
                if h:
                    count[h] += 1
                    item = ("Explain " + lower_words(h), h, 1)
                else:
                    m = DEF_Q.match(s)
                    if not m or m.group(1).split()[0].lower() in PRONOUNS:
                        continue
                    item = ("What is " + lower_words(m.group(1)) + "?", None, 0)
                lst = by_doc.setdefault(c["doc"], [])
                if item[0] not in [x[0] for x in lst]:
                    lst.append(item)

        lists = []
        for items in by_doc.values():
            items = [x for x in items if not x[1] or count[x[1]] <= 2]
            items.sort(key=lambda x: x[2])
            lists.append([x[0] for x in items])

        out, used = [], set()
        for rnd in range(15):
            for lst in lists:
                if rnd < len(lst) and len(out) < n:
                    q = lst[rnd]
                    if q.lower() not in used and self.ask(q)["grounded"]:
                        out.append(q)
                        used.add(q.lower())
            if len(out) >= n:
                break

        terms = self.vec.get_feature_names_out()
        step = max(1, len(self.chunks) // n)
        for i in range(0, len(self.chunks), step):
            if len(out) >= n:
                break
            row = self.matrix[i].toarray()[0]
            for j in row.argsort()[::-1][:3]:
                w = self.surface.get(terms[j])
                if w and len(w) >= 6:
                    q = "What is " + w + "?"
                    if q.lower() not in used and self.ask(q)["grounded"]:
                        out.append(q)
                        used.add(q.lower())
                        break
        return out


if __name__ == "__main__":
    qa = DocQA()
    print("Loaded", len(qa.chunks), "chunks\n")
    print("Suggested:", qa.suggestions, "\n")
    for q in ["Why do leaves look green?", "What is overfitting?",
              "Who won the 2018 World Cup?"]:
        r = qa.ask(q)
        print("Q:", q)
        print("A:", r["answer"], "(score", r["score"], ")\n")