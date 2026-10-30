# BiomedBERT-based Classifier for Dentistry Studies

Scientific article classifier (main area and study type) based on the **BiomedBERT** model, with a local graphical interface for classification.

---

## Table of Contents

- [Local Installation (Linux Mint)](#local-installation-linux-mint)
- [How to Use the Interface](#how-to-use-the-interface)
- [Training on Google Colab (with GPU)](#training-on-google-colab-with-gpu)
- [Model Categories](#model-categories)
- [Project Structure](#project-structure)

---

## Local Installation (Linux Mint)

### 1. Install pip
```bash
sudo apt install python3-pip
```

### 2. Install the virtual environment tool
```bash
sudo apt install python3-venv
```

### 3. Install Tkinter (required for the graphical interface)
```bash
sudo apt install python3-tk
```

### 4. Create the virtual environment
```bash
python3 -m venv biomedbert-env
```

### 5. Activate the virtual environment
```bash
source biomedbert-env/bin/activate
```
You will see `(biomedbert-env)` at the beginning of the terminal line — this means it is active. **This command must be run every time you open a new terminal**, before using the program.

### 6. Install the required libraries
```bash
pip install transformers torch flask flask-cors
```

### 7. Verify the installation
```bash
python3 -c "from transformers import AutoTokenizer; print('OK')"
```
If `OK` appears, everything is set up correctly.

---

## How to Use the Interface

The program runs entirely from the file `biomedbert_app.py`. There is no need to start a server or open a browser.

### Option A — Run from the terminal (recommended)

1. **Open a terminal directly in the project folder.**
   On Linux Mint, right-click an empty area inside the folder in the Files app and choose **"Open Terminal Here"**. This way you never need to type or change the folder path.

2. **Activate the virtual environment:**
   ```bash
   source ~/biomedbert-env/bin/activate
   ```

3. **Start the program:**
   ```bash
   python3 biomedbert_app.py
   ```

The graphical window will open. No further terminal interaction is needed.

---

### Option B — Run as a program with a double-click (Linux Mint)

You can make `biomedbert_app.py` behave like a regular application:

1. Right-click `biomedbert_app.py` in the Files app.
2. Go to **Properties → Permissions**.
3. Check the box **"Allow executing file as program"**.
4. Close Properties.
5. Double-click the file — the system will ask whether to run it or view it; choose **"Run"**.

> **Note:** This method requires the virtual environment to already have the libraries installed. On other Linux distributions the path to the Permissions setting may differ, but the concept is the same.

---

### What the interface does

1. Click **"▶ Start BiomedBERT"** to load the model into memory (takes 30–60 seconds; only needed once per session).
2. Click **"Browse…"** and select a `.ris` or `.rdf` file exported from Zotero.
3. Click **"▶ Classify Articles"** and wait for the progress bar to complete.
4. Download the results in one of three formats:
   - **Full JSON** — complete output with all fields and confidence scores
   - **Markdown Table** — formatted report with abstracts and rationale
   - **JSON for Review** — compact output with the essential fields for systematic review

> The model is already trained and saved in the `modelo_treinado(1)/` folder. **No retraining is needed to classify new files.** The program runs fully offline — it never contacts the internet during classification.

---

## Training on Google Colab (with GPU)

Google Colab link: **https://colab.research.google.com/**

### How to enable the GPU in Colab

1. Open a new notebook at https://colab.research.google.com/
2. In the top menu, click **Runtime**.
3. Click **Change runtime type**.
4. Under **Hardware accelerator**, select **GPU** (may appear as T4 GPU).
5. Click **Save**.
6. To confirm the GPU is active, run in a cell:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
   ```
   It should print `True` and the GPU name (e.g. `Tesla T4`).

### Before running the script

Upload the files `area completo.json` and `tipo completo.json` to the `/content/` folder in Colab (folder icon in the left sidebar → upload button).

### Training script

Paste the script below into a Colab cell and run it:

```python
# Install dependencies
!pip install transformers torch accelerate -q

import json, os, torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from torch.utils.data import Dataset

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
print(f"Model: {MODEL_NAME}")

# ── Categories ────────────────────────────────────────────────────────────────
AREAS = [
    "Áreas não relacionadas à saúde",
    "Ciências Biológicas e da Saúde",
    "Medicina",
    "Odontologia",
]

TIPOS = [
    "Caso clínico / Série de casos",
    "Estudo em animais",
    "Estudo in vitro",
    "Não foi possível classificar",
    "Observacional",
    "RCT",
    "Revisão",
]

# ── Input text ────────────────────────────────────────────────────────────────
def texto_artigo(art):
    titulo = art.get("Título completo do estudo", "") or ""
    resumo = art.get("Resumo", "") or ""
    kws    = art.get("Palavras-chave", []) or []
    kw_str = " ".join(kws) if isinstance(kws, list) else str(kws)
    return f"{titulo} [SEP] {resumo} {kw_str}".strip()

# ── Dataset ───────────────────────────────────────────────────────────────────
class ArticleDataset(Dataset):
    def __init__(self, textos, labels, tok):
        self.enc    = tok(textos, truncation=True, padding=True,
                         max_length=512, return_tensors="pt")
        self.labels = torch.tensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
os.makedirs("/content/modelo_treinado", exist_ok=True)

# ── Train AREA (6 epochs) ─────────────────────────────────────────────────────
print("\n=== Loading AREA file ===")
with open("/content/area completo.json", encoding="utf-8") as f:
    dados_area = json.load(f)
print(f"Articles loaded: {len(dados_area)}")

textos_area, labels_area = [], []
nao_encontrados_area = set()
for art in dados_area:
    val = art.get("Área Principal", "")
    if val in AREAS:
        textos_area.append(texto_artigo(art))
        labels_area.append(AREAS.index(val))
    elif val:
        nao_encontrados_area.add(val)

print(f"Valid articles for AREA training: {len(textos_area)}")
if nao_encontrados_area:
    print(f"Unrecognised AREA values: {nao_encontrados_area}")

model_area = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(AREAS))

args_area = TrainingArguments(
    output_dir="/content/modelo_treinado/area_checkpoints",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    save_strategy="no",
    logging_steps=50,
)
Trainer(model=model_area, args=args_area,
        train_dataset=ArticleDataset(textos_area, labels_area, tokenizer)).train()
model_area.save_pretrained("/content/modelo_treinado/area", max_shard_size="90MB")
tokenizer.save_pretrained("/content/modelo_treinado/area")
print("AREA saved.")

# ── Train TIPO (6 epochs) ─────────────────────────────────────────────────────
print("\n=== Loading TIPO file ===")
with open("/content/tipo completo.json", encoding="utf-8") as f:
    dados_tipo = json.load(f)
print(f"Articles loaded: {len(dados_tipo)}")

textos_tipo, labels_tipo = [], []
nao_encontrados_tipo = set()
for art in dados_tipo:
    val = art.get("Tipo de estudo", "")
    if val in TIPOS:
        textos_tipo.append(texto_artigo(art))
        labels_tipo.append(TIPOS.index(val))
    elif val:
        nao_encontrados_tipo.add(val)

print(f"Valid articles for TIPO training: {len(textos_tipo)}")
if nao_encontrados_tipo:
    print(f"Unrecognised TIPO values: {nao_encontrados_tipo}")

model_tipo = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(TIPOS))

args_tipo = TrainingArguments(
    output_dir="/content/modelo_treinado/tipo_checkpoints",
    num_train_epochs=6,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    save_strategy="no",
    logging_steps=50,
)
Trainer(model=model_tipo, args=args_tipo,
        train_dataset=ArticleDataset(textos_tipo, labels_tipo, tokenizer)).train()
model_tipo.save_pretrained("/content/modelo_treinado/tipo", max_shard_size="90MB")
tokenizer.save_pretrained("/content/modelo_treinado/tipo")
print("TIPO saved.")

# ── Zip and download ──────────────────────────────────────────────────────────
import shutil
shutil.make_archive("/content/modelo_treinado", "zip", "/content/modelo_treinado")
from google.colab import files
files.download("/content/modelo_treinado.zip")
print("Download started!")
```

At the end, Colab will automatically download a file called `modelo_treinado.zip` containing both trained models (`area/` and `tipo/`). Extract the zip and place the folder alongside `biomedbert_app.py` to use the models with the graphical interface.

---

## Model Categories

### Main Area (4 categories)
- Áreas não relacionadas à saúde
- Ciências Biológicas e da Saúde
- Medicina
- Odontologia

### Study Type (7 categories)
- Caso clínico / Série de casos
- Estudo em animais
- Estudo in vitro
- Não foi possível classificar
- Observacional
- RCT
- Revisão

---

## Project Structure

```
.
├── biomedbert_app.py              # Graphical interface + classification engine
├── servidor.py                    # Legacy Flask backend (no longer required)
├── interface.html                 # Legacy web interface (no longer required)
├── modelo_treinado(1)/            # Trained model — 90 MB shards
│   ├── area/                      # Main-area classification model
│   │   ├── model-00001-of-00005.safetensors
│   │   ├── ...
│   │   └── tokenizer.json
│   └── tipo/                      # Study-type classification model
│       ├── model-00001-of-00005.safetensors
│       ├── ...
│       └── tokenizer.json
└── README.md
```
