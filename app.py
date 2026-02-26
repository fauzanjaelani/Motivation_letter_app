import streamlit as st
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Motivation Letter Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Serif+Display:ital@0;1&display=swap');

    /* ════════════════════════════════════════
       0. HEADER BAR — putih bersih, tanpa hitam
    ════════════════════════════════════════ */
    header[data-testid="stHeader"],
    [data-testid="stHeader"] {
        background-color: #f7f8fc !important;
        border-bottom: 1px solid #e8ecf4 !important;
        box-shadow: none !important;
    }

    /* ── Sembunyikan Fork, Share, menu titik tiga ── */
    [data-testid="stToolbarActions"] { display: none !important; }
    [data-testid="stMainMenu"]       { display: none !important; }
    [data-testid="stDecoration"]     { display: none !important; }
    [data-testid="stToolbar"]        { background: transparent !important; }

    /* ── Tombol sidebar: background gradient biru ── */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stCollapseSidebarButton"] {
        background: linear-gradient(135deg, #4f6ef7 0%, #7c4dff 100%) !important;
        border-radius: 10px !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        border: none !important;
        box-shadow: 0 3px 12px rgba(79,110,247,0.4) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        position: relative !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stCollapseSidebarButton"]:hover {
        box-shadow: 0 5px 18px rgba(79,110,247,0.6) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Sembunyikan semua child (teks icon material) ── */
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stCollapseSidebarButton"] * {
        font-size: 0 !important;
        color: transparent !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* ── Inject ikon ☰ via pseudo-element pada tombol langsung ── */
    [data-testid="stExpandSidebarButton"]::after,
    [data-testid="stCollapseSidebarButton"]::after {
        content: "☰" !important;
        font-size: 18px !important;
        color: #ffffff !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        line-height: 1 !important;
        font-family: sans-serif !important;
        width: auto !important;
        height: auto !important;
    }

    /* ════════════════════════════════════════
       1. FONT GLOBAL
    ════════════════════════════════════════ */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="block-container"],
    .stMarkdown, p, span, label, div {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* ════════════════════════════════════════
       2. BACKGROUND
    ════════════════════════════════════════ */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"] {
        background-color: #f7f8fc !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e8ecf4 !important;
    }

    /* ════════════════════════════════════════
       3. WARNA TEKS GLOBAL
    ════════════════════════════════════════ */
    .stApp, .stMarkdown, .stText,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] td,
    [data-testid="stMarkdownContainer"] th {
        color: #1e2235 !important;
    }

    h1, h2, h3, h4 {
        color: #1e2235 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    label, .stTextArea label, .stTextInput label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: #3d4460 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.01em !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] p,
    small, .caption {
        color: #7b82a0 !important;
    }

    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] div {
        color: #1e2235 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] td,
    [data-testid="stSidebar"] th,
    [data-testid="stSidebar"] label {
        color: #1e2235 !important;
    }

    /* ════════════════════════════════════════
       4. TOMBOL
    ════════════════════════════════════════ */
    .stButton > button {
        color: #ffffff !important;
        background: linear-gradient(135deg, #4f6ef7 0%, #7c4dff 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 10px rgba(79,110,247,0.25) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b5ce4 0%, #6a3ce8 100%) !important;
        box-shadow: 0 4px 18px rgba(79,110,247,0.38) !important;
        transform: translateY(-1px) !important;
    }

    /* Tombol Analisis (primary) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f6ef7 0%, #7c4dff 100%) !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(79,110,247,0.35) !important;
    }

    /* ════════════════════════════════════════
       5. HEADER BOX
    ════════════════════════════════════════ */
    .main-header {
        text-align: center;
        padding: 2.5rem 2rem 2rem;
        background: linear-gradient(135deg, #4f6ef7 0%, #7c4dff 100%);
        border-radius: 20px;
        color: #ffffff !important;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(79,110,247,0.28);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -60px; left: -20px;
        width: 150px; height: 150px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .main-header h1 {
        margin: 0 !important;
        font-size: 2.1rem !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
    }
    .main-header p {
        margin: 0.5rem 0 0 !important;
        opacity: 0.88;
        font-size: 1.02rem !important;
        color: #ffffff !important;
        font-weight: 400 !important;
    }

    /* ════════════════════════════════════════
       6. SCORE CARD
    ════════════════════════════════════════ */
    .score-total-card {
        background: linear-gradient(135deg, #4f6ef7 0%, #7c4dff 100%);
        border-radius: 18px;
        padding: 2rem 1rem;
        text-align: center;
        color: #ffffff !important;
        box-shadow: 0 6px 24px rgba(79,110,247,0.3);
    }
    .score-number  { font-size: 4.5rem; font-weight: 800; line-height: 1; color: #ffffff !important; letter-spacing: -0.04em; }
    .score-denom   { font-size: 1rem; opacity: 0.8; margin-top: 0.15rem; color: #ffffff !important; }
    .category-pill {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.9rem;
        color: #ffffff !important;
        background: rgba(255,255,255,0.22);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* ════════════════════════════════════════
       7. FEEDBACK & QUESTION BOX
    ════════════════════════════════════════ */
    .feedback-box {
        background: #f0f3ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #4f6ef7;
        color: #1e2235 !important;
        margin-top: 0.3rem;
        font-size: 0.93rem;
        line-height: 1.65;
    }
    .question-box {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e0e5f5;
        border-left: 5px solid #4f6ef7;
        font-size: 1rem;
        color: #1e2235 !important;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 10px rgba(79,110,247,0.07);
        line-height: 1.7;
    }

    /* ════════════════════════════════════════
       8. EXPANDER
    ════════════════════════════════════════ */
    [data-testid="stExpander"],
    [data-testid="stExpander"] > details,
    [data-testid="stExpander"] > details > summary,
    [data-testid="stExpander"] > details > div {
        background-color: #ffffff !important;
        color: #1e2235 !important;
    }
    [data-testid="stExpander"] > details {
        border: 1px solid #e0e5f5 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(79,110,247,0.06) !important;
    }
    [data-testid="stExpander"] > details > summary:hover {
        background-color: #f7f9ff !important;
    }
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details > div p,
    [data-testid="stExpander"] details > div span,
    [data-testid="stExpander"] details > div label {
        color: #1e2235 !important;
    }
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] summary svg path {
        fill: #4f6ef7 !important;
        color: #4f6ef7 !important;
    }

    /* ════════════════════════════════════════
       9. TEXTAREA
    ════════════════════════════════════════ */
    textarea {
        background-color: #ffffff !important;
        border: 1.5px solid #d4daef !important;
        border-radius: 12px !important;
        color: #1e2235 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    textarea:focus {
        border-color: #4f6ef7 !important;
        background-color: #fafbff !important;
        box-shadow: 0 0 0 3px rgba(79,110,247,0.12) !important;
    }
    textarea::placeholder {
        color: #a0a8c0 !important;
    }

    /* ════════════════════════════════════════
       10. METRIC CARDS
    ════════════════════════════════════════ */
    [data-testid="metric-container"] {
        background: #ffffff !important;
        border: 1px solid #e0e5f5 !important;
        border-radius: 12px !important;
        padding: 0.9rem 1rem !important;
        box-shadow: 0 2px 8px rgba(79,110,247,0.06) !important;
    }

    /* ════════════════════════════════════════
       11. DIVIDER
    ════════════════════════════════════════ */
    hr { border-color: #e8ecf4 !important; }

    /* ════════════════════════════════════════
       12. SIDEBAR STYLE
    ════════════════════════════════════════ */
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #1e2235 !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8ecf4;
        margin-bottom: 0.8rem !important;
    }
    [data-testid="stSidebar"] table {
        font-size: 0.88rem !important;
    }
    [data-testid="stSidebar"] td, [data-testid="stSidebar"] th {
        padding: 0.35rem 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_handcrafted_features(text):
    features = []
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    features.append(len(text))
    features.append(len(words))
    features.append(len(sentences))
    features.append(len(paragraphs))

    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    features.append(avg_word_length)
    features.append(avg_sentence_length)

    capital_count = sum(1 for s in sentences if s and s[0].isupper())
    capital_ratio = capital_count / len(sentences) if sentences else 0
    features.append(capital_ratio)

    informal_words = ['aku', 'gue', 'gw', 'gak', 'nggak', 'banget', 'kayak', 'gitu', 'soalnya']
    informal_count = sum(1 for w in words if w.lower() in informal_words)
    informal_ratio = informal_count / len(words) if words else 0
    features.append(informal_ratio)

    para_sentence_ratio = len(paragraphs) / len(sentences) if sentences else 0
    features.append(para_sentence_ratio)

    experience_kw = ['pengalaman', 'ketika', 'saat', 'pernah', 'mengikuti', 'kompetisi', 'proyek']
    features.append(sum(1 for k in experience_kw if k in text.lower()))

    reflection_kw = ['belajar', 'memahami', 'menyadari', 'mengajarkan', 'inspirasi', 'percaya', 'yakin']
    features.append(sum(1 for k in reflection_kw if k in text.lower()))

    reason_kw = ['memilih', 'karena', 'alasan', 'tujuan', 'ingin', 'tertarik', 'minat']
    features.append(sum(1 for k in reason_kw if k in text.lower()))

    unique_words = len(set(w.lower() for w in words))
    features.append(unique_words / len(words) if words else 0)

    features.append(1 if re.search(r'\d+', text) else 0)

    transitions = ['selanjutnya', 'kemudian', 'namun', 'oleh karena itu', 'dengan demikian', 'selain itu']
    features.append(sum(1 for t in transitions if t in text.lower()))

    return np.array(features).reshape(1, -1)


def score_to_category(total_score):
    if total_score >= 75:
        return 'Baik'
    elif total_score >= 55:
        return 'Cukup'
    else:
        return 'Kurang'


def generate_feedback(score, aspect):
    templates = {
        'grammar': {
            'high':   "Tata bahasa sangat baik dengan struktur kalimat yang bervariasi dan ejaan yang tepat.",
            'medium': "Tata bahasa cukup baik, namun masih ada ruang perbaikan pada variasi kalimat dan penggunaan kata formal.",
            'low':    "Tata bahasa memerlukan perbaikan signifikan. Perhatikan ejaan, penggunaan kata formal, dan struktur kalimat."
        },
        'flow': {
            'high':   "Alur cerita sangat koheren dengan transisi yang mulus antar paragraf dan pengembangan ide yang logis.",
            'medium': "Alur cukup baik, namun bisa ditingkatkan dengan lebih banyak kata transisi dan koneksi yang lebih jelas antar ide.",
            'low':    "Alur cerita perlu perbaikan. Gunakan kata penghubung dan pastikan ada perkembangan jelas dari pengalaman → refleksi → tujuan."
        },
        'structure': {
            'high':   "Struktur essay sangat baik: paragraf terbagi tepat, pembuka jelas, dan penutup kuat.",
            'medium': "Struktur dasar baik, namun bisa diperkuat dengan pembagian paragraf yang lebih jelas di tiap bagian.",
            'low':    "Struktur perlu perbaikan signifikan. Pecah essay menjadi 3–4 paragraf dengan pembuka, isi, dan penutup yang jelas."
        },
        'relevance': {
            'high':   "Sangat relevan dengan pertanyaan. Semua aspek — latar belakang, pengalaman, alasan — dijelaskan dengan detail yang baik.",
            'medium': "Cukup relevan, namun beberapa aspek bisa lebih detail. Pastikan menjelaskan latar belakang, pengalaman spesifik, dan alasan memilih jurusan.",
            'low':    "Kurang relevan. Pastikan menjawab semua bagian: latar belakang minat, pengalaman yang mempengaruhi, dan alasan memilih jurusan."
        },
        'depth': {
            'high':   "Kedalaman sangat baik: ada contoh konkret, refleksi personal yang kuat, dan detail spesifik yang memperkaya narasi.",
            'medium': "Kedalaman cukup baik, namun bisa diperkaya dengan lebih banyak contoh spesifik dan refleksi yang lebih mendalam.",
            'low':    "Kedalaman masih kurang. Berikan contoh konkret, jelaskan refleksi personal, dan tambahkan detail spesifik (nama kegiatan, angka, dll)."
        }
    }
    level = 'high' if score >= 75 else ('medium' if score >= 55 else 'low')
    return templates[aspect][level]


def predict(text, components):
    models     = components['models']
    vectorizer = components['tfidf_vectorizer']

    tfidf = vectorizer.transform([text]).toarray()
    hc    = extract_handcrafted_features(text)
    X     = np.hstack([tfidf, hc])

    aspects = ['grammar', 'flow', 'structure', 'relevance', 'depth']
    scores  = {a: float(models[a].predict(X)[0]) for a in aspects}

    scores['total'] = float(np.mean([scores[a] for a in aspects]))
    category = score_to_category(scores['total'])

    feedback = {
        a: {'score': round(scores[a], 2),
            'feedback': generate_feedback(scores[a], a)}
        for a in aspects
    }

    words      = text.split()
    paragraphs = [p for p in text.split('\n\n') if p.strip()]

    return {
        'total_score':     round(scores['total'], 2),
        'category':        category,
        'word_count':      len(words),
        'paragraph_count': len(paragraphs),
        'aspects':         feedback
    }


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("motivation_letter_ml_model.joblib")

try:
    components  = load_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def score_color(score):
    if score >= 75: return '#16a34a'
    if score >= 55: return '#d97706'
    return '#dc2626'

def cat_color(cat):
    return {'baik': '#16a34a', 'cukup': '#d97706', 'kurang': '#dc2626'}.get(cat.lower(), '#6b7280')

def cat_emoji(cat):
    return {'baik': '🟢', 'cukup': '🟡', 'kurang': '🔴'}.get(cat.lower(), '⚪')


def radar_chart(aspects):
    labels = ['Grammar', 'Flow', 'Struktur', 'Relevansi', 'Kedalaman']
    keys   = ['grammar', 'flow', 'structure', 'relevance', 'depth']
    vals   = [aspects[k]['score'] for k in keys]
    N      = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    vals_p = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f7f8fc')
    ax.plot(angles, vals_p, 'o-', lw=2.5, color='#4f6ef7')
    ax.fill(angles, vals_p, alpha=0.18, color='#4f6ef7')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, fontweight='bold', color='#1e2235')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=7.5, color='#9ca3af')
    ax.grid(color='#e0e5f5', linestyle='--', lw=0.8, alpha=0.8)
    ax.spines['polar'].set_color('#e0e5f5')
    plt.tight_layout()
    return fig


def bar_chart(aspects):
    labels = ['Grammar', 'Flow', 'Struktur', 'Relevansi', 'Kedalaman']
    keys   = ['grammar', 'flow', 'structure', 'relevance', 'depth']
    vals   = [aspects[k]['score'] for k in keys]
    colors = [score_color(v) for v in vals]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f7f8fc')
    bars = ax.barh(labels, vals, color=colors, edgecolor='white', lw=1.5, height=0.52)
    ax.set_xlim(0, 100)
    ax.axvline(55, color='#dc2626', ls='--', lw=1.2, alpha=0.6, label='Batas Kurang (55)')
    ax.axvline(75, color='#16a34a', ls='--', lw=1.2, alpha=0.6, label='Batas Baik (75)')
    for bar, val in zip(bars, vals):
        ax.text(min(val + 1.5, 95), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontweight='bold', fontsize=9.5, color='#1e2235')
    ax.set_xlabel('Skor (0–100)', fontsize=9.5, color='#7b82a0')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e8ecf4')
    ax.spines['bottom'].set_color('#e8ecf4')
    ax.tick_params(colors='#7b82a0')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📌 Tentang Aplikasi")
    st.markdown("""
Aplikasi ini mengevaluasi kualitas **motivation letter** secara otomatis menggunakan model Machine Learning.

**Model:** TF-IDF + Random Forest  


---
**Aspek yang dinilai:**
| Aspek | Keterangan |
|---|---|
| ✍️ Grammar | Tata bahasa & ejaan |
| 🔄 Flow | Alur & koherensi |
| 🏗️ Struktur | Susunan paragraf |
| 🎯 Relevansi | Kesesuaian jawaban |
| 🌊 Kedalaman | Detail & refleksi |

---
**Kategori Skor:**
- 🟢 **Baik** — skor ≥ 75
- 🟡 **Cukup** — skor 55–74
- 🔴 **Kurang** — skor < 55

---
**💡 Tips penulisan:**
- Gunakan bahasa **formal**
- Bagi menjadi **3–4 paragraf**
- Ceritakan **pengalaman konkret**
- Sertakan **angka / nama kegiatan**
- Gunakan **kata transisi** antar paragraf
""")


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
import base64, os

def get_logo_b64():
    logo_path = os.path.join(os.path.dirname(__file__), "bear_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_b64()
logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:90px;height:90px;border-radius:50%;margin-bottom:0.7rem;border:3px solid rgba(255,255,255,0.45);box-shadow:0 4px 16px rgba(0,0,0,0.15);" /><br/>' if logo_b64 else ""

st.markdown(f"""
<div class="main-header">
    {logo_html}
    <h1>📝 Motivation Letter Analyzer</h1>
    <p>Evaluasi kualitas jawaban motivation letter Anda secara instan dengan AI</p>
</div>
""", unsafe_allow_html=True)

# ── Model tidak ditemukan ──
if not model_ready:
    st.error("""
**❌ File model tidak ditemukan.**

Pastikan file `motivation_letter_ml_model.joblib` berada di folder yang **sama** dengan `app.py`, lalu jalankan ulang aplikasi.

> **Cara mendapatkan file model:** jalankan notebook `Motivation_Letter_AI_revised_(2).ipynb` hingga cell terakhir — file `.joblib` akan tersimpan otomatis.
""")
    st.stop()

# ── Pertanyaan ──
st.markdown("""
<div class="question-box">
    <strong>📋 Pertanyaan:</strong><br>
    Apa latar belakang minat akademik Anda, pengalaman yang mempengaruhinya,
    dan alasan Anda memilih jurusan yang dituju?
</div>
""", unsafe_allow_html=True)

# ── Tombol Bersihkan ──
if 'essay' not in st.session_state:
    st.session_state['essay'] = ''

col_btn, _ = st.columns([1, 6])
with col_btn:
    if st.button("🗑️ Bersihkan"):
        st.session_state['essay'] = ''
        st.rerun()

essay = st.text_area(
    "Tulis jawaban Anda di sini:",
    key='essay',
    height=230,
    placeholder="Ketik atau paste jawaban motivation letter Anda...",
)

word_count = len(essay.split()) if essay.strip() else 0
st.caption(f"📊 Jumlah kata: **{word_count}**  |  Disarankan minimal 50 kata untuk hasil yang akurat.")

analyze = st.button("🔍 Analisis Sekarang", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# HASIL ANALISIS
# ─────────────────────────────────────────────
if analyze:
    if not essay.strip():
        st.error("❌ Kolom jawaban masih kosong. Silakan isi terlebih dahulu.")
        st.stop()
    if word_count < 10:
        st.warning("⚠️ Teks terlalu pendek. Tambahkan lebih banyak kalimat agar analisis lebih akurat.")
        st.stop()

    with st.spinner("🤖 Menganalisis jawaban Anda..."):
        result = predict(essay, components)

    st.markdown("---")
    st.markdown("## 📊 Hasil Analisis")

    col_total, col_stats = st.columns([1, 2])

    with col_total:
        st.markdown(f"""
        <div class="score-total-card">
            <div class="score-number">{result['total_score']}</div>
            <div class="score-denom">/ 100</div>
            <div>
                <span class="category-pill">
                    {cat_emoji(result['category'])} {result['category'].upper()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.markdown("#### 📈 Statistik Teks")
        m1, m2 = st.columns(2)
        m1.metric("Jumlah Kata", result['word_count'])
        m2.metric("Jumlah Paragraf", result['paragraph_count'])

        st.markdown("**Progress Skor Total:**")
        st.progress(result['total_score'] / 100)

        if result['total_score'] >= 75:
            st.success("Motivation letter Anda berkualitas **Baik**. Terus pertahankan!")
        elif result['total_score'] >= 55:
            st.warning("Motivation letter Anda berkualitas **Cukup**. Masih ada ruang untuk ditingkatkan.")
        else:
            st.error("Motivation letter Anda berkualitas **Kurang**. Perhatikan rekomendasi di bawah.")

    st.markdown("---")

    col_radar, col_bar = st.columns(2)
    with col_radar:
        st.markdown("#### 🕸️ Radar Chart")
        st.pyplot(radar_chart(result['aspects']), use_container_width=True)
    with col_bar:
        st.markdown("#### 📊 Skor per Aspek")
        st.pyplot(bar_chart(result['aspects']), use_container_width=True)

    st.markdown("---")

    st.markdown("### 📋 Feedback Detail per Aspek")

    aspect_meta = [
        ('grammar',   '✍️ Tata Bahasa',  'Grammar'),
        ('flow',      '🔄 Alur Cerita',  'Flow'),
        ('structure', '🏗️ Struktur',     'Structure'),
        ('relevance', '🎯 Relevansi',    'Relevance'),
        ('depth',     '🌊 Kedalaman',    'Depth'),
    ]

    for key, label_id, label_en in aspect_meta:
        data  = result['aspects'][key]
        score = data['score']
        color = score_color(score)
        level = '🟢 Baik' if score >= 75 else ('🟡 Cukup' if score >= 55 else '🔴 Kurang')

        with st.expander(f"{label_id} / {label_en}  —  **{score}/100**  ({level})", expanded=True):
            col_s, col_f = st.columns([1, 3])
            with col_s:
                st.progress(score / 100)
                st.markdown(
                    f"<div style='text-align:center;font-size:2.2rem;font-weight:800;color:{color};font-family:Plus Jakarta Sans,sans-serif'>"
                    f"{score}</div>",
                    unsafe_allow_html=True
                )
            with col_f:
                st.markdown(
                    f"<div class='feedback-box'>{data['feedback']}</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    st.markdown("### 💡 Rekomendasi Peningkatan")

    names_id = {
        'grammar':   'Tata Bahasa',
        'flow':      'Alur Cerita',
        'structure': 'Struktur',
        'relevance': 'Relevansi',
        'depth':     'Kedalaman'
    }

    low_list = [(k, v) for k, v in result['aspects'].items() if v['score'] < 55]
    mid_list = [(k, v) for k, v in result['aspects'].items() if 55 <= v['score'] < 75]

    if not low_list and not mid_list:
        st.success("🎉 Semua aspek sudah sangat baik. Motivation letter Anda siap digunakan!")
    else:
        if low_list:
            st.error("**⚠️ Perlu perbaikan segera (skor < 55):**")
            for k, v in low_list:
                st.markdown(f"- **{names_id[k]}** ({v['score']}/100): {v['feedback']}")
        if mid_list:
            st.warning("**📈 Dapat ditingkatkan (skor 55–74):**")
            for k, v in mid_list:
                st.markdown(f"- **{names_id[k]}** ({v['score']}/100): {v['feedback']}")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#a0a8c0;font-size:0.82rem;font-family:Plus Jakarta Sans,sans-serif;padding:0.5rem 0 1rem;'>"
    "Motivation Letter Analyzer · TF-IDF + Random Forest · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)