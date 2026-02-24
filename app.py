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
    /* ════════════════════════════════════════
       1. BACKGROUND — putih bersih
    ════════════════════════════════════════ */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"] {
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f4ff !important;
    }

    /* ════════════════════════════════════════
       2. WARNA TEKS GLOBAL — gelap agar terbaca di bg putih
    ════════════════════════════════════════ */
    /* Teks utama / body */
    .stApp, .stMarkdown, .stText,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] td,
    [data-testid="stMarkdownContainer"] th {
        color: #1a1a2e !important;
    }

    /* Heading h1–h4 di halaman utama */
    h1, h2, h3, h4 {
        color: #1a1a2e !important;
    }

    /* Label input / widget */
    label, .stTextArea label, .stTextInput label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }

    /* Caption / teks kecil */
    .stCaption, [data-testid="stCaptionContainer"] p,
    small, .caption {
        color: #555577 !important;
    }

    /* Metric label & value */
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] div {
        color: #1a1a2e !important;
    }

    /* Teks di sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] td,
    [data-testid="stSidebar"] th,
    [data-testid="stSidebar"] label {
        color: #1a1a2e !important;
    }

    /* ── Ikon & tombol navigasi sidebar (collapse, menu, dll) — hitam ── */
    [data-testid="stSidebar"] button svg,
    [data-testid="stSidebar"] button svg path,
    [data-testid="stSidebar"] button svg polyline,
    [data-testid="stSidebar"] button svg line,
    [data-testid="stSidebar"] button svg rect,
    [data-testid="stSidebarCollapseButton"] svg path,
    [data-testid="collapsedControl"] svg path,
    button[data-testid="stBaseButton-headerNoPadding"] svg path,
    header button svg path,
    header button svg {
        fill: #000000 !important;
        color: #000000 !important;
        stroke: #000000 !important;
    }

    /* Tombol collapse sidebar */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        color: #000000 !important;
        opacity: 1 !important;
    }

    /* Teks di dalam expander */
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    details summary p {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }

    /* Teks tombol */
    .stButton > button {
        color: #ffffff !important;
        background-color: #667eea !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #4a5fd4 !important;
    }

    /* Divider */
    hr { border-color: #d0d4e8 !important; }

    /* ════════════════════════════════════════
       3. HEADER BOX — Motivation Letter Analyzer
    ════════════════════════════════════════ */
    .main-header {
        text-align: center;
        padding: 1.8rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px;
        color: #ffffff !important;
        margin-bottom: 2rem;
        box-shadow: 0 4px 18px rgba(102,126,234,0.25);
    }
    .main-header h1 { margin: 0; font-size: 2rem; color: #ffffff !important; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.92; font-size: 1rem; color: #ffffff !important; }

    /* ════════════════════════════════════════
       4. SCORE CARD
    ════════════════════════════════════════ */
    .score-total-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px;
        padding: 1.8rem 1rem;
        text-align: center;
        color: #ffffff !important;
        box-shadow: 0 4px 18px rgba(102,126,234,0.25);
    }
    .score-number  { font-size: 4rem; font-weight: 800; line-height: 1; color: #ffffff !important; }
    .score-denom   { font-size: 1.1rem; opacity: 0.85; margin-top: 0.2rem; color: #ffffff !important; }
    .category-pill {
        display: inline-block;
        padding: 0.35rem 1.1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.8rem;
        color: #ffffff !important;
    }

    /* ════════════════════════════════════════
       5. FEEDBACK & QUESTION BOX
    ════════════════════════════════════════ */
    .feedback-box {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        border-left: 4px solid #667eea;
        color: #1a1a2e !important;
        margin-top: 0.3rem;
    }
    .question-box {
        background: #eef2ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #667eea;
        font-size: 1rem;
        color: #1a1a2e !important;
        margin-bottom: 1rem;
    }


    /* ════════════════════════════════════════
       6. EXPANDER — paksa light mode (backup anti dark mode)
    ════════════════════════════════════════ */
    [data-testid="stExpander"],
    [data-testid="stExpander"] > details,
    [data-testid="stExpander"] > details > summary,
    [data-testid="stExpander"] > details > div {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }
    [data-testid="stExpander"] > details {
        border: 1px solid #d0d8f0 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(102,126,234,0.08) !important;
    }
    [data-testid="stExpander"] > details > summary:hover {
        background-color: #f0f4ff !important;
    }
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details > div p,
    [data-testid="stExpander"] details > div span,
    [data-testid="stExpander"] details > div label {
        color: #1a1a2e !important;
    }
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] summary svg path {
        fill: #667eea !important;
        color: #667eea !important;
    }

    /* ════════════════════════════════════════
       6. TEXTAREA — biru
    ════════════════════════════════════════ */
    textarea {
        background-color: #ddeeff !important;
        border: 2px solid #4a90e2 !important;
        border-radius: 8px !important;
        color: #1a1a2e !important;
    }
    textarea:focus {
        border-color: #1a6fc4 !important;
        background-color: #cce4ff !important;
        box-shadow: 0 0 0 3px rgba(74,144,226,0.25) !important;
    }
    textarea::placeholder {
        color: #6699cc !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS (sesuai notebook revisi terbaru)
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
    """Rule-based threshold — sesuai notebook revisi terbaru."""
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
    # Threshold disesuaikan dengan notebook: Baik >= 75, Cukup >= 55, Kurang < 55
    level = 'high' if score >= 75 else ('medium' if score >= 55 else 'low')
    return templates[aspect][level]


def predict(text, components):
    """
    Predict using updated model structure:
    - 5 regressors only (grammar, flow, structure, relevance, depth)
    - Total = simple average of 5 components
    - Category = rule-based threshold (no label_encoder needed)
    """
    models     = components['models']
    vectorizer = components['tfidf_vectorizer']

    tfidf = vectorizer.transform([text]).toarray()
    hc    = extract_handcrafted_features(text)
    X     = np.hstack([tfidf, hc])

    aspects = ['grammar', 'flow', 'structure', 'relevance', 'depth']
    scores  = {a: float(models[a].predict(X)[0]) for a in aspects}

    # Total = rata-rata sederhana 5 komponen (sesuai notebook)
    scores['total'] = float(np.mean([scores[a] for a in aspects]))

    # Kategori = rule-based threshold (sesuai notebook)
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
    if score >= 75: return '#28a745'
    if score >= 55: return '#f0a500'
    return '#dc3545'

def cat_color(cat):
    return {'baik': '#28a745', 'cukup': '#f0a500', 'kurang': '#dc3545'}.get(cat.lower(), '#6c757d')

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
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    ax.plot(angles, vals_p, 'o-', lw=2, color='#667eea')
    ax.fill(angles, vals_p, alpha=0.22, color='#667eea')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=7.5, color='gray')
    ax.grid(color='gray', linestyle='--', lw=0.5, alpha=0.4)
    ax.spines['polar'].set_visible(False)
    plt.tight_layout()
    return fig


def bar_chart(aspects):
    labels = ['Grammar', 'Flow', 'Struktur', 'Relevansi', 'Kedalaman']
    keys   = ['grammar', 'flow', 'structure', 'relevance', 'depth']
    vals   = [aspects[k]['score'] for k in keys]
    colors = [score_color(v) for v in vals]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    bars = ax.barh(labels, vals, color=colors, edgecolor='white', lw=1.2, height=0.52)
    ax.set_xlim(0, 100)
    ax.axvline(55, color='#dc3545', ls='--', lw=1.2, alpha=0.7, label='Batas Kurang (55)')
    ax.axvline(75, color='#28a745', ls='--', lw=1.2, alpha=0.7, label='Batas Baik (75)')
    for bar, val in zip(bars, vals):
        ax.text(min(val + 1.5, 95), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontweight='bold', fontsize=9.5)
    ax.set_xlabel('Skor (0–100)', fontsize=9.5)
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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
logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:90px;height:90px;border-radius:50%;margin-bottom:0.6rem;border:3px solid rgba(255,255,255,0.5);" /><br/>' if logo_b64 else ""

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
    <strong> Pertanyaan:</strong><br>
    Apa latar belakang minat akademik Anda, pengalaman yang mempengaruhinya,
    dan alasan Anda memilih jurusan yang dituju?
</div>
""", unsafe_allow_html=True)

# ── Tombol bantuan ──
col_btn1, col_btn2, _ = st.columns([1, 1, 5])
with col_btn1:
    use_example = st.button("📋 Lihat Contoh")
with col_btn2:
    clear = st.button("🗑️ Bersihkan")

EXAMPLE = """Ketertarikan saya terhadap Teknik Informatika berawal dari pengalaman masa kecil ketika ayah saya, seorang engineer, sering membawa pulang perangkat elektronik untuk diperbaiki. Saya terbiasa melihat bagaimana ia menganalisis masalah secara sistematis dan menemukan solusi kreatif.

Pengalaman yang paling berkesan adalah saat SMA ketika saya bergabung dengan tim robotika sekolah. Kami mengembangkan robot line follower untuk kompetisi regional, dan saya bertanggung jawab atas pemrograman algoritma navigasi. Proses debugging dan optimasi kode mengajarkan saya pentingnya berpikir logis dan teliti.

Saya memilih Teknik Informatika karena program studi ini menawarkan fondasi kuat dalam algoritma, struktur data, dan pengembangan perangkat lunak yang saya butuhkan untuk mencapai tujuan karir di bidang AI dan data science."""

if use_example:
    st.session_state['essay'] = EXAMPLE
if clear:
    st.session_state['essay'] = ''

essay = st.text_area(
    "Tulis jawaban Anda di sini:",
    value=st.session_state.get('essay', ''),
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

    # ── Skor Total & Statistik ──
    col_total, col_stats = st.columns([1, 2])

    with col_total:
        st.markdown(f"""
        <div class="score-total-card">
            <div class="score-number">{result['total_score']}</div>
            <div class="score-denom">/ 100</div>
            <div>
                <span class="category-pill" style="background:{cat_color(result['category'])};">
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

    # ── Visualisasi ──
    col_radar, col_bar = st.columns(2)
    with col_radar:
        st.markdown("#### 🕸️ Radar Chart")
        st.pyplot(radar_chart(result['aspects']), use_container_width=True)
    with col_bar:
        st.markdown("#### 📊 Skor per Aspek")
        st.pyplot(bar_chart(result['aspects']), use_container_width=True)

    st.markdown("---")

    # ── Feedback Detail ──
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
                    f"<div style='text-align:center;font-size:2.2rem;font-weight:800;color:{color}'>"
                    f"{score}</div>",
                    unsafe_allow_html=True
                )
            with col_f:
                st.markdown(
                    f"<div class='feedback-box'>{data['feedback']}</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ── Rekomendasi ──
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
    "<div style='text-align:center;color:gray;font-size:0.82rem;'>"
    "Motivation Letter Analyzer · TF-IDF + Random Forest · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)