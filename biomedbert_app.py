#!/usr/bin/env python3
"""
BiomedBERT Classifier — Graphical Interface (Tkinter)
Single file: replaces the interface.html + terminal workflow.
Works from any folder with any folder name.
Linux Mint / Ubuntu.
"""

import json
import os
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import xml.etree.ElementTree as ET

# ── Locate the model folder relative to this script ──────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelo_treinado(1)")

# ── Classification categories (must match the trained model) ──────────────────
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

# ── Colour palettes (Zorin OS GTK — ZorinBlue-Light / ZorinBlue-Dark) ─────────
# Values must be unique inside each palette (used to recolour live widgets).
THEMES = {
    "ZorinBlue-Light": {
        "C_BLUE": "#0B78BE", "C_BLUE_D": "#09659F", "C_BLUE_L": "#DCEFFB",
        "C_LINK": "#0A5E96",
        "C_BG": "#F6F6F6", "C_SURFACE": "#FFFFFF", "C_SURFACE2": "#FAFAFA",
        "C_BORDER": "#DADADA", "C_TEXT": "#323232", "C_TEXT_S": "#727272",
        "C_SUCCESS": "#2E9E4F", "C_ERROR": "#E0393E", "C_HEADER": "#EBEBEB",
        "C_HEADER_FG": "#2D2D2D", "C_ON_ACCENT": "#FFFFFE", "C_TROUGH": "#E4E4E4",
    },
    "ZorinBlue-Dark": {
        "C_BLUE": "#1279BE", "C_BLUE_D": "#0E66A1", "C_BLUE_L": "#1E3A4C",
        "C_LINK": "#8FD3F8",
        "C_BG": "#2B2D31", "C_SURFACE": "#35373C", "C_SURFACE2": "#303237",
        "C_BORDER": "#1E1F22", "C_TEXT": "#EDEDED", "C_TEXT_S": "#A4A6AB",
        "C_SUCCESS": "#4CC271", "C_ERROR": "#F0555A", "C_HEADER": "#232428",
        "C_HEADER_FG": "#F2F2F2", "C_ON_ACCENT": "#FFFFFF", "C_TROUGH": "#45474D",
    },
}


def _system_prefers_dark():
    try:
        import subprocess
        out = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True, text=True, timeout=2).stdout
        return "dark" in out.lower()
    except Exception:
        return False


CURRENT_THEME = "ZorinBlue-Dark" if _system_prefers_dark() else "ZorinBlue-Light"


def _apply_palette(name):
    global CURRENT_THEME
    CURRENT_THEME = name
    globals().update(THEMES[name])


_apply_palette(CURRENT_THEME)

# ── Global state ──────────────────────────────────────────────────────────────
_models        = {}   # loaded tokenizer + model pairs
_results_cache = []   # last classification results (for downloads)

# =============================================================================
#  CLASSIFICATION LOGIC  (mirrors servidor.py, without Flask)
# =============================================================================

def load_model(name):
    """Load and cache a tokenizer + model pair from the local folder."""
    if name not in _models:
        # Force fully offline — never contact HuggingFace Hub
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = os.path.join(MODEL_DIR, name)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mod = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True)
        mod.eval()
        _models[name] = (tok, mod)
    return _models[name]


def article_text(art):
    title   = art.get("Título completo do estudo", "") or ""
    abstract = art.get("Resumo", "") or ""
    kws     = art.get("Palavras-chave", []) or []
    kw_str  = " ".join(kws) if isinstance(kws, list) else str(kws)
    return f"{title} [SEP] {abstract} {kw_str}".strip()


def rationale_area(label, confidence):
    pct   = round(confidence * 100, 1)
    level = "high" if confidence >= 0.85 else "moderate" if confidence >= 0.60 else "low"
    return (f"{level.capitalize()} confidence ({pct}%) — BiomedBERT classified as "
            f"'{label}' based on title, abstract and keywords.")


def rationale_tipo(label, confidence):
    pct   = round(confidence * 100, 1)
    level = "high" if confidence >= 0.85 else "moderate" if confidence >= 0.60 else "low"
    return (f"{level.capitalize()} confidence ({pct}%) — BiomedBERT identified a "
            f"'{label}' methodological pattern in the abstract.")


def classify_batches(texts, tok, mod, batch=64):
    import torch
    all_preds, all_confs = [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc   = tok(chunk, truncation=True, padding=True,
                    max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = mod(**enc).logits
        all_preds.extend(torch.argmax(logits, dim=1).tolist())
        all_confs.extend(torch.softmax(logits, dim=1).max(dim=1).values.tolist())
    return all_preds, all_confs


def classify_articles(articles, progress_cb):
    results = []
    texts   = [article_text(a) for a in articles]
    total   = len(articles)

    for name, labels in [("area", AREAS), ("tipo", TIPOS)]:
        msg = ("Classifying main area" if name == "area"
               else "Classifying study type")
        pct = 20 if name == "area" else 60
        progress_cb(pct, f"{msg} ({total} articles)…")
        tok, mod = load_model(name)
        preds, confs = classify_batches(texts, tok, mod)

        if name == "area":
            for i, art in enumerate(articles):
                r = dict(art)
                r["Área Principal"]        = labels[preds[i]]
                r["Confiança Área"]        = round(confs[i], 4)
                r["Justificativa da Área"] = rationale_area(labels[preds[i]], confs[i])
                results.append(r)
        else:
            for i, r in enumerate(results):
                r["Tipo de estudo"]        = labels[preds[i]]
                r["Confiança Tipo"]        = round(confs[i], 4)
                r["Justificativa do Tipo"] = rationale_tipo(labels[preds[i]], confs[i])

    return results


# ── File parsers ──────────────────────────────────────────────────────────────

def parse_ris(content):
    articles, current = [], {}
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "ER  -":
            if current:
                articles.append(current)
                current = {}
            continue
        m = re.match(r'^([A-Z][A-Z0-9])\s+-\s*(.*)', line)
        if not m:
            continue
        tag, val = m.group(1), m.group(2).strip()
        if   tag == "TI": current["Título completo do estudo"] = val
        elif tag == "AB": current["Resumo"] = current.get("Resumo", "") + " " + val
        elif tag == "T2": current["Periódico"] = val
        elif tag == "J2": current["Abreviação"] = val
        elif tag == "SN": current["ISSN"] = val
        elif tag == "AU": current.setdefault("Autores", []).append(val)
        elif tag == "KW": current.setdefault("Palavras-chave", []).append(val)
        elif tag == "DO": current["DOI"] = val
        elif tag == "PY": current["Ano"] = val
        elif tag == "CY": current["Local"] = val
    return articles


def parse_rdf(content):
    articles = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return articles

    ns = {
        "bib":     "http://purl.org/net/biblio#",
        "dc":      "http://purl.org/dc/elements/1.1/",
        "dcterms": "http://purl.org/dc/terms/",
        "foaf":    "http://xmlns.com/foaf/0.1/",
        "prism":   "http://prismstandard.org/namespaces/1.2/basic/",
        "rdf":     "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "z":       "http://www.zotero.org/namespaces/export#",
        "vcard":   "http://nkd.org/vCard#",
    }

    journals = {}
    for j in root.findall(".//bib:Journal", ns):
        about    = j.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
        title    = j.findtext("dc:title",            namespaces=ns, default="")
        alt      = j.findtext("dcterms:alternative", namespaces=ns, default="")
        issn_raw = j.findtext("dc:identifier",       namespaces=ns, default="")
        issn     = issn_raw.replace("ISSN ", "").strip() if "ISSN" in issn_raw else ""
        journals[about] = {"Periódico": title, "Abreviação": alt, "ISSN": issn}

    for art in root.findall(".//bib:Article", ns):
        a = {}
        a["Título completo do estudo"] = (art.findtext("dc:title",        namespaces=ns) or "").strip()
        a["Resumo"]                    = (art.findtext("dcterms:abstract", namespaces=ns) or "").strip()
        a["Local"] = ""

        ref_node = art.find("dcterms:isPartOf", ns)
        if ref_node is not None:
            ref = ref_node.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")
            if ref in journals:
                a.update(journals[ref])

        authors = []
        for person in art.findall(".//foaf:Person", ns):
            sn = person.findtext("foaf:surname",   namespaces=ns, default="")
            gn = person.findtext("foaf:givenName", namespaces=ns, default="")
            if sn:
                authors.append(f"{sn}, {gn}".strip(", "))
        a["Autores"] = authors

        kws = [s.text.strip() for s in art.findall("dc:subject", ns) if s.text]
        a["Palavras-chave"] = kws

        date = art.findtext("dc:date", namespaces=ns, default="")
        a["Ano"] = date[:4] if date else ""

        if a.get("Título completo do estudo"):
            articles.append(a)
    return articles


# ── Export helpers ────────────────────────────────────────────────────────────

def generate_markdown(results):
    lines = []
    lines.append("# BiomedBERT Classification Results\n")
    lines.append(f"Total articles classified: **{len(results)}**\n")
    lines.append("---\n")
    for i, r in enumerate(results, 1):
        title   = r.get("Título completo do estudo", "").strip()
        year    = r.get("Ano", "").strip()
        abstract = r.get("Resumo", "").strip()
        area    = r.get("Área Principal", "")
        tipo    = r.get("Tipo de estudo", "")
        conf_a  = r.get("Confiança Área", 0)
        conf_t  = r.get("Confiança Tipo", 0)
        just_a  = r.get("Justificativa da Área", "")
        just_t  = r.get("Justificativa do Tipo", "")
        journal = r.get("Periódico", "")
        doi     = r.get("DOI", "")

        lines.append(f"## {i}. {title}")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| **Year** | {year} |")
        lines.append(f"| **Journal** | {journal} |")
        if doi:
            lines.append(f"| **DOI** | {doi} |")
        lines.append(f"| **Main Area** | {area} |")
        lines.append(f"| **Study Type** | {tipo} |")
        lines.append(f"| **Area Confidence** | {round(conf_a*100,1)}% |")
        lines.append(f"| **Type Confidence** | {round(conf_t*100,1)}% |")
        lines.append("")
        lines.append("**Area Rationale:**")
        lines.append(f"> {just_a}")
        lines.append("")
        lines.append("**Type Rationale:**")
        lines.append(f"> {just_t}")
        lines.append("")
        lines.append("**Abstract:**")
        lines.append("")
        lines.append(abstract)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def generate_json_for_review(results):
    """Compact JSON with the essential fields for systematic review."""
    output = []
    for r in results:
        output.append({
            "Periódico":                 r.get("Periódico", ""),
            "Abreviação":                r.get("Abreviação", ""),
            "ISSN":                      r.get("ISSN", ""),
            "Área Principal":            r.get("Área Principal", ""),
            "Título completo do estudo": r.get("Título completo do estudo", ""),
            "Tipo de estudo":            r.get("Tipo de estudo", ""),
            "Justificativa da Área":     r.get("Justificativa da Área", ""),
            "Justificativa do Tipo":     r.get("Justificativa do Tipo", ""),
        })
    return json.dumps(output, ensure_ascii=False, indent=2)


# =============================================================================
#  GRAPHICAL INTERFACE (Tkinter)
# =============================================================================

# ── Rounded GTK-style button (Tkinter has no native rounded corners) ─────────
import tkinter.font as tkfont


def _mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


class RoundButton(tk.Canvas):
    """Canvas button with rounded corners that mimics the tk.Button API."""
    _OWN = ("text", "bg", "background", "fg", "foreground", "activebackground",
            "activeforeground", "disabledforeground", "state", "command",
            "font", "padx", "pady", "relief", "bd", "borderwidth")

    def __init__(self, master, **kw):
        self._o = {"text": "", "bg": "#888888", "fg": "#ffffff",
                   "activebackground": "", "activeforeground": "",
                   "disabledforeground": "", "state": "normal",
                   "command": None, "font": ("Cantarell", 9),
                   "padx": 12, "pady": 5, "bd": 0, "relief": "flat"}
        self._hover = False
        self._press = False
        own = {k: kw.pop(k) for k in list(kw) if k in self._OWN}
        super().__init__(master, highlightthickness=0, bd=0,
                         bg=master.cget("bg"), **kw)
        self.configure(**own)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: self._set("_hover", True))
        self.bind("<Leave>", lambda e: (self._set("_hover", False),
                                         self._set("_press", False)))
        self.bind("<ButtonPress-1>", lambda e: self._set("_press", True))
        self.bind("<ButtonRelease-1>", self._release)

    def _norm(self, k):
        return {"background": "bg", "foreground": "fg",
                "borderwidth": "bd"}.get(k, k)

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        rest = {}
        for k, v in kw.items():
            if k in self._OWN:
                self._o[self._norm(k)] = v
            else:
                rest[k] = v
        if rest:
            super().configure(**rest)
        if kw:
            self._resize()
            self._draw()

    config = configure

    def cget(self, k):
        if k in self._OWN:
            return self._o[self._norm(k)]
        return super().cget(k)

    __getitem__ = cget

    def _set(self, attr, val):
        setattr(self, attr, val)
        self._draw()

    def _release(self, e):
        was = self._press
        self._press = False
        self._draw()
        inside = 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height()
        if was and inside and self._o["state"] != "disabled" and self._o["command"]:
            self._o["command"]()

    def _resize(self):
        f = tkfont.Font(font=self._o["font"])
        w = f.measure(self._o["text"]) + 2 * int(self._o["padx"]) + 12
        h = f.metrics("linespace") + 2 * int(self._o["pady"]) + 6
        super().configure(width=w, height=h)

    @staticmethod
    def _rrect(cv, x1, y1, x2, y2, r, **kw):
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return cv.create_polygon(pts, smooth=True, **kw)

    def _draw(self):
        try:
            parent_bg = self.master.cget("bg")
        except tk.TclError:
            return
        super().configure(bg=parent_bg)
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4:
            w, h = int(super().cget("width")), int(super().cget("height"))
        o = self._o
        bg, fg = o["bg"], o["fg"]
        disabled = o["state"] == "disabled"
        if disabled:
            bg = _mix(bg, parent_bg, 0.55)
            fg = _mix(fg, bg, 0.35)
        elif self._press:
            bg = _mix(bg, "#000000", 0.18)
        elif self._hover:
            bg = o["activebackground"] or _mix(bg, "#ffffff", 0.12)
        is_flat = bg.lower() == parent_bg.lower() and int(o["bd"]) == 0
        if not is_flat or self._hover:
            border = _mix(bg, "#000000", 0.12) if bg.lower() != "#ffffff" else C_BORDER
            self._rrect(self, 1, 1, w - 1, h - 1, 7, fill=border, outline=border)
            self._rrect(self, 2, 2, w - 2, h - 2, 6, fill=bg, outline=bg)
        self.create_text(w / 2, h / 2, text=o["text"], fill=fg, font=o["font"])
        super().configure(cursor="arrow" if disabled else "hand2")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BiomedBERT — Article Classifier")
        self.configure(bg=C_BG)
        self.resizable(False, False)
        self.geometry("620x560")

        self._file_path    = None
        self._loading      = False
        self._classifying  = False

        self._style_ttk()
        self._build_ui()
        self._check_model()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Blue title bar
        bar = tk.Frame(self, bg=C_HEADER, height=48)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="🧬  BiomedBERT Classifier",
                 bg=C_HEADER, fg=C_HEADER_FG,
                 font=("Cantarell", 12, "bold")).pack(side="left", padx=12, pady=10)
        tk.Label(bar, text="— Scientific Articles",
                 bg=C_HEADER, fg=C_TEXT_S,
                 font=("Cantarell", 10)).pack(side="left")
        self._btn_theme = RoundButton(bar, text="◐", bg=C_HEADER, fg=C_HEADER_FG,
                                    activebackground=C_BORDER, relief="flat", bd=0,
                                    font=("Cantarell", 12), cursor="hand2",
                                    command=self._toggle_theme)
        self._btn_theme.pack(side="right", padx=10)
        tk.Frame(self, bg=C_BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=C_BG, padx=18, pady=14)
        body.pack(fill="both", expand=True)

        # ── Card: Engine control ──────────────────────────────────────────────
        card_engine = self._card(body, "⚙️  BiomedBERT Engine")

        self._dot = tk.Canvas(card_engine, width=10, height=10, bg=C_SURFACE,
                              highlightthickness=0)
        self._dot.pack(side="left", padx=(0, 6))
        self._draw_dot("#AAAAAA")

        self._lbl_status = tk.Label(card_engine, text="Checking model…",
                                    bg=C_SURFACE, fg=C_TEXT_S,
                                    font=("Cantarell", 9))
        self._lbl_status.pack(side="left", fill="x", expand=True)

        self._btn_engine = RoundButton(card_engine,
                                     text="▶  Start BiomedBERT",
                                     bg=C_BLUE, fg=C_ON_ACCENT, relief="flat",
                                     font=("Cantarell", 9, "bold"),
                                     padx=10, pady=4, cursor="hand2",
                                     command=self._toggle_engine)
        self._btn_engine.pack(side="right", before=self._lbl_status)

        # ── Card: File selection ──────────────────────────────────────────────
        card_file = self._card(body, "📂  RIS or RDF File")

        self._lbl_file = tk.Label(card_file,
                                  text="No file selected",
                                  bg=C_SURFACE2, fg=C_TEXT_S,
                                  font=("Cantarell", 9),
                                  relief="flat", bd=1,
                                  anchor="w", padx=8)
        self._lbl_file.pack(side="left", fill="x", expand=True,
                             ipady=6, padx=(0, 8))

        RoundButton(card_file,
                  text="Browse…",
                  bg=C_BLUE_L, fg=C_LINK, relief="flat",
                  font=("Cantarell", 9, "bold"),
                  padx=10, pady=4, cursor="hand2",
                  command=self._select_file).pack(side="right")

        # ── Classify button ───────────────────────────────────────────────────
        self._btn_classify = RoundButton(body,
                                       text="▶  Classify Articles",
                                       bg=C_BLUE, fg=C_ON_ACCENT, relief="flat",
                                       font=("Cantarell", 10, "bold"),
                                       pady=8, cursor="hand2",
                                       state="disabled",
                                       command=self._start_classification)
        self._btn_classify.pack(fill="x", pady=(4, 0))

        # ── Progress bar ──────────────────────────────────────────────────────
        self._prog_var = tk.DoubleVar(value=0)
        self._prog_bar = ttk.Progressbar(body, variable=self._prog_var,
                                          maximum=100, length=580)
        self._prog_bar.pack(fill="x", pady=(8, 0))

        self._lbl_prog = tk.Label(body, text="",
                                  bg=C_BG, fg=C_TEXT_S,
                                  font=("Cantarell", 8))
        self._lbl_prog.pack(anchor="center", pady=(2, 4))

        # ── Result message ────────────────────────────────────────────────────
        self._lbl_result = tk.Label(body, text="",
                                    bg=C_BG, fg=C_SUCCESS,
                                    font=("Cantarell", 9, "bold"))
        self._lbl_result.pack(anchor="w", pady=(0, 6))

        # ── Card: Downloads ───────────────────────────────────────────────────
        card_dl = self._card(body, "⬇️  Download Results")

        self._btn_json = RoundButton(card_dl, text="⬇ Full JSON",
                                   bg=C_BLUE, fg=C_ON_ACCENT, relief="flat",
                                   font=("Cantarell", 9, "bold"),
                                   padx=8, pady=5, cursor="hand2",
                                   state="disabled",
                                   command=lambda: self._download("json"))
        self._btn_json.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self._btn_md = RoundButton(card_dl, text="⬇ Markdown Table",
                                 bg=C_SURFACE, fg=C_LINK, relief="flat",
                                 font=("Cantarell", 9, "bold"),
                                 padx=8, pady=5, cursor="hand2",
                                 state="disabled", bd=1,
                                 command=lambda: self._download("md"))
        self._btn_md.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self._btn_review = RoundButton(card_dl, text="⬇ JSON for Review",
                                     bg=C_BLUE_L, fg=C_LINK, relief="flat",
                                     font=("Cantarell", 9, "bold"),
                                     padx=8, pady=5, cursor="hand2",
                                     state="disabled",
                                     command=lambda: self._download("review"))
        self._btn_review.pack(side="left", expand=True, fill="x")

        # ── Footer ────────────────────────────────────────────────────────────
        tk.Label(body,
                 text="BiomedBERT · microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 bg=C_BG, fg=C_TEXT_S, font=("Cantarell", 7)).pack(pady=(10, 0))

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _style_ttk(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Horizontal.TProgressbar", background=C_BLUE,
                        troughcolor=C_TROUGH, bordercolor=C_BORDER,
                        lightcolor=C_BLUE, darkcolor=C_BLUE, thickness=6)

    def _toggle_theme(self):
        old = THEMES[CURRENT_THEME]
        new_name = ("ZorinBlue-Dark" if CURRENT_THEME == "ZorinBlue-Light"
                    else "ZorinBlue-Light")
        new = THEMES[new_name]
        mapping = {old[k].lower(): new[k] for k in old}
        _apply_palette(new_name)
        opts = ("bg", "fg", "activebackground", "activeforeground",
                "disabledforeground", "highlightbackground", "selectcolor")

        def walk(w):
            for o in opts:
                try:
                    v = str(w.cget(o)).lower()
                except tk.TclError:
                    continue
                if v in mapping:
                    w.configure(**{o: mapping[v]})
            for c in w.winfo_children():
                walk(c)
        walk(self)
        self._style_ttk()

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=C_BORDER, bd=1)
        outer.pack(fill="x", pady=(0, 10))
        header = tk.Frame(outer, bg=C_SURFACE2)
        header.pack(fill="x")
        tk.Label(header, text=title,
                 bg=C_SURFACE2, fg=C_TEXT,
                 font=("Cantarell", 9, "bold"),
                 anchor="w", padx=10).pack(fill="x", pady=5)
        tk.Frame(outer, bg=C_BORDER, height=1).pack(fill="x")
        body = tk.Frame(outer, bg=C_SURFACE, padx=10, pady=8)
        body.pack(fill="x")
        return body

    def _draw_dot(self, color):
        self._dot.delete("all")
        self._dot.create_oval(1, 1, 9, 9, fill=color, outline=color)

    def _set_status(self, text, dot_color="#AAAAAA"):
        self._lbl_status.config(text=text)
        self._draw_dot(dot_color)

    def _set_downloads_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for btn in (self._btn_json, self._btn_md, self._btn_review):
            btn.config(state=state)

    # ── Model check ───────────────────────────────────────────────────────────

    def _check_model(self):
        ok_area = os.path.isdir(os.path.join(MODEL_DIR, "area"))
        ok_tipo = os.path.isdir(os.path.join(MODEL_DIR, "tipo"))
        if ok_area and ok_tipo:
            self._set_status("Model found. Click Start to load it into memory.", "#AAAAAA")
            self._btn_engine.config(state="normal")
        else:
            folder = os.path.basename(MODEL_DIR)
            self._set_status(
                f"⚠  Folder '{folder}/area' or 'tipo' not found next to this program.",
                C_ERROR)
            self._btn_engine.config(state="disabled")

    # ── Engine on / off ───────────────────────────────────────────────────────

    def _toggle_engine(self):
        if _models:
            _models.clear()
            self._btn_engine.config(text="▶  Start BiomedBERT", bg=C_BLUE)
            self._set_status("Engine stopped.", "#AAAAAA")
            self._btn_classify.config(state="disabled")
        else:
            self._loading = True
            self._btn_engine.config(text="⏳  Loading…", state="disabled", bg=C_BLUE_D)
            self._set_status(
                "Loading models into memory… (may take 1–2 min the first time)", C_LINK)
            threading.Thread(target=self._load_models_bg, daemon=True).start()

    def _load_models_bg(self):
        try:
            load_model("area")
            self.after(0, lambda: self._set_status("Loading study-type model…", C_LINK))
            load_model("tipo")
            self.after(0, self._engine_ready)
        except Exception as e:
            self.after(0, lambda: self._engine_error(str(e)))

    def _engine_ready(self):
        self._loading = False
        self._btn_engine.config(text="⏹  Stop BiomedBERT",
                                state="normal", bg=C_ERROR)
        self._set_status("Engine active — ready to classify.", C_SUCCESS)
        if self._file_path:
            self._btn_classify.config(state="normal")

    def _engine_error(self, msg):
        self._loading = False
        self._btn_engine.config(text="▶  Start BiomedBERT",
                                state="normal", bg=C_BLUE)
        self._set_status(f"Load error: {msg}", C_ERROR)

    # ── File selection ────────────────────────────────────────────────────────

    def _select_file(self):
        path = filedialog.askopenfilename(
            title="Select Zotero export file",
            filetypes=[("RIS / RDF / XML", "*.ris *.rdf *.xml"), ("All files", "*.*")]
        )
        if path:
            self._file_path = path
            self._lbl_file.config(text=f"✔  {os.path.basename(path)}", fg=C_SUCCESS)
            if _models:
                self._btn_classify.config(state="normal")

    # ── Classification ────────────────────────────────────────────────────────

    def _start_classification(self):
        if not self._file_path or self._classifying:
            return
        if not _models:
            messagebox.showwarning("Engine stopped",
                                   "Please start BiomedBERT before classifying.")
            return

        try:
            with open(self._file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("File read error", str(e))
            return

        fname = self._file_path.lower()
        if fname.endswith(".ris"):
            articles = parse_ris(content)
        elif fname.endswith((".rdf", ".xml")):
            articles = parse_rdf(content)
        else:
            messagebox.showerror("Unsupported format",
                                 "Please use a .ris or .rdf file exported from Zotero.")
            return

        if not articles:
            messagebox.showwarning("Empty file",
                                   "No articles were found in the selected file.")
            return

        self._classifying = True
        self._btn_classify.config(state="disabled")
        self._set_downloads_enabled(False)
        self._lbl_result.config(text="")
        self._prog_var.set(5)
        self._lbl_prog.config(text=f"Starting classification of {len(articles)} articles…")

        threading.Thread(target=self._classify_bg,
                         args=(articles,), daemon=True).start()

    def _classify_bg(self, articles):
        def cb(pct, msg):
            self.after(0, lambda p=pct, m=msg: self._update_progress(p, m))
        try:
            global _results_cache
            results = classify_articles(articles, cb)
            _results_cache = results
            self.after(0, lambda: self._classification_done(len(results)))
        except Exception as e:
            self.after(0, lambda: self._classification_error(str(e)))

    def _update_progress(self, pct, msg):
        self._prog_var.set(pct)
        self._lbl_prog.config(text=msg)

    def _classification_done(self, total):
        self._classifying = False
        self._prog_var.set(100)
        self._lbl_prog.config(text=f"✔ {total} articles classified.")
        self._lbl_result.config(
            text=f"✔ Classification complete — {total} articles ready to download.")
        self._btn_classify.config(state="normal")
        self._set_downloads_enabled(True)

    def _classification_error(self, msg):
        self._classifying = False
        self._prog_var.set(0)
        self._lbl_prog.config(text="")
        self._btn_classify.config(state="normal")
        messagebox.showerror("Classification error", msg)

    # ── Downloads ─────────────────────────────────────────────────────────────

    def _download(self, fmt):
        if not _results_cache:
            messagebox.showwarning("No data",
                                   "Run a classification before downloading.")
            return

        if fmt == "json":
            dest = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile="classification_results.json",
                filetypes=[("JSON", "*.json")])
            if dest:
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(_results_cache, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Saved", f"File saved to:\n{dest}")

        elif fmt == "md":
            dest = filedialog.asksaveasfilename(
                defaultextension=".md",
                initialfile="classification_results.md",
                filetypes=[("Markdown", "*.md")])
            if dest:
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(generate_markdown(_results_cache))
                messagebox.showinfo("Saved", f"File saved to:\n{dest}")

        elif fmt == "review":
            dest = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile="classification_for_review.json",
                filetypes=[("JSON", "*.json")])
            if dest:
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(generate_json_for_review(_results_cache))
                messagebox.showinfo("Saved", f"File saved to:\n{dest}")


# =============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()
