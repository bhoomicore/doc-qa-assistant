# NeonDoc Quest: Document Q&A Assistant

A document question-answering assistant that answers only from the documents you give it, shows the supporting source passage for every answer, and says **"Information not found in provided document."** when the answer is not there. It runs 100% locally with free open-source tools and no paid APIs.

The interface is a neon-yellow game: each question earns XP, you build combos, complete quests and unlock badges.

## Features
- Upload `.txt` and `.pdf` files, and delete documents from the vault
- Sentence-based chunking with overlap, plus cleaning of PDF bullets and headings
- TF-IDF retrieval (scikit-learn) with typo fixing, phrase matching and definition-aware ranking
- Extractive answers that are complete sentences, with the source passage and matched words highlighted
- Zero hallucination: if the best match is below the threshold, the app refuses to answer
- Suggested questions generated automatically from the uploaded documents
- Offline voice questions using the Vosk speech model
- Game layer: XP, levels, combo, quests and badges

## How it works
1. **Ingest:** text is cleaned and split into chunks of about 80 words.
2. **Retrieve:** the question is cleaned and matched against chunks using TF-IDF cosine similarity, term coverage and phrase overlap.
3. **Answer:** the best sentences are returned as the answer, with the full supporting passage.
4. **Fallback:** if term coverage is below 0.45, the answer is "Information not found in provided document."

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
On Mac/Linux, activate with `source venv/bin/activate`.

### Offline voice (optional)
Download `vosk-model-small-en-us-0.15` from https://alphacephei.com/vosk/models and unzip it into `models/`, so the path is `models/vosk-model-small-en-us-0.15/`.

## Run
```
streamlit run app.py
```
Command-line check: `python qa_engine.py`

## Project structure
```
app.py                Streamlit neon game interface
qa_engine.py          Cleaning, chunking, retrieval, answers, suggestions
make_transcripts.py   Rebuilds the transcripts section below
corpus/               Sample documents (3 .txt files)
.streamlit/           Theme configuration
```

## Acceptance criteria
- [x] Every answer shows its supporting source passage
- [x] Ungrounded questions return "Information not found in provided document."
- [x] 100% local, free open-source tooling, no paid APIs

## Sample transcripts
<!-- TRANSCRIPTS START -->
### Grounded: Why do leaves look green?

**Answer:** Chlorophyll absorbs red and blue light and reflects green light, which is why leaves look green. During photosynthesis, plants take in carbon dioxide through small pores called stomata and absorb water through their roots. Using sunlight, they turn these into glucose and release oxygen as a by-product.

**Source:** `photosynthesis.txt`, chunk 1 (score 1.00)

> Photosynthesis is the process by which green plants convert light energy into chemical energy. It takes place mainly in the chloroplasts, which contain the pigment chlorophyll. Chlorophyll absorbs red and blue light and reflects green light, which is why leaves look green. During photosynthesis, plants take in carbon dioxide through small pores called stomata and absorb water through their roots. Using sunlight, they turn these into glucose and release oxygen as a by-product.

### Grounded: How does a solar panel make electricity?

**Answer:** Solar panels use photovoltaic cells made of silicon to convert sunlight directly into electricity. When photons hit the cell, they knock electrons loose and create an electric current. Panels produce direct current, which an inverter converts to alternating current for home use. The efficiency of most commercial solar panels is between 15 and 22 percent. Efficiency drops as panel temperature rises, so ventilation behind the panels helps output.

**Source:** `solar_energy.txt`, chunk 1 (score 0.72)

> Solar panels use photovoltaic cells made of silicon to convert sunlight directly into electricity. When photons hit the cell, they knock electrons loose and create an electric current. Panels produce direct current, which an inverter converts to alternating current for home use. The efficiency of most commercial solar panels is between 15 and 22 percent. Efficiency drops as panel temperature rises, so ventilation behind the panels helps output.

### Grounded: What is overfitting?

**Answer:** Overfitting happens when a model memorises the training data instead of learning general patterns. It can be reduced with more data, regularisation, or by stopping training early. A validation set helps detect overfitting.

**Source:** `machine_learning.txt`, chunk 2 (score 1.00)

> Overfitting happens when a model memorises the training data instead of learning general patterns. It can be reduced with more data, regularisation, or by stopping training early. A validation set helps detect overfitting.

### Grounded: definition of photosynthesis

**Answer:** Photosynthesis is the process by which green plants convert light energy into chemical energy. It takes place mainly in the chloroplasts, which contain the pigment chlorophyll. Chlorophyll absorbs red and blue light and reflects green light, which is why leaves look green. During photosynthesis, plants take in carbon dioxide through small pores called stomata and absorb water through their roots. Using sunlight, they turn these into glucose and release oxygen as a by-product.

**Source:** `photosynthesis.txt`, chunk 1 (score 1.00)

> Photosynthesis is the process by which green plants convert light energy into chemical energy. It takes place mainly in the chloroplasts, which contain the pigment chlorophyll. Chlorophyll absorbs red and blue light and reflects green light, which is why leaves look green. During photosynthesis, plants take in carbon dioxide through small pores called stomata and absorb water through their roots. Using sunlight, they turn these into glucose and release oxygen as a by-product.

### Unsupported: Who won the 2018 World Cup?

**Answer:** Information not found in provided document.

No supporting passage. Best match score 0.00 is below the threshold, so nothing is generated.

### Unsupported: What is the capital of Australia?

**Answer:** Information not found in provided document.

No supporting passage. Best match score 0.00 is below the threshold, so nothing is generated.

<!-- TRANSCRIPTS END -->