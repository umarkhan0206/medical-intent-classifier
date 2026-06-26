"""
Streamlit demo: NHS Medical Intent Classification (DistilBERT)

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from inference import MedicalIntentClassifier  # noqa: E402

st.set_page_config(
    page_title="Medical Intent Classifier",
    page_icon="⚕",
    layout="centered",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;500;600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink:        #0B2540;
    --parchment:  #F7F3EC;
    --parchment2: #EFE9DC;
    --sage:       #2E7D52;
    --sage-bg:    #DCEFE3;
    --ochre:      #D9650B;
    --ochre-bg:   #FCE4CC;
    --slate:      #6B7585;
    --rule:       #D8D0C0;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--parchment);
}

header[data-testid="stHeader"] { background: transparent; }

.cn-title {
    font-family: 'IBM Plex Serif', serif;
    font-weight: 600;
    font-size: 2.1rem;
    color: var(--ink);
    letter-spacing: -0.01em;
    margin-bottom: 0.1rem;
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
}
.cn-title .rx {
    font-size: 1.6rem;
    color: var(--sage);
}
.cn-subtitle {
    color: var(--slate);
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

.cn-chips-plain {
    font-size: 0.85rem;
    color: var(--slate);
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 1.3rem;
    line-height: 1.6;
}
.cn-chips-wrap {
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 1.3rem;
}

.cn-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--slate);
    margin-bottom: 0.4rem;
}

div[data-testid="stTextInput"] input {
    background: white;
    border: 1px solid var(--rule);
    border-radius: 4px;
    font-size: 1rem;
    color: var(--ink);
    padding: 0.7rem 0.9rem;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--ink);
    box-shadow: none;
}

div[data-testid="stButton"] button {
    background: white;
    border: 1px solid var(--rule);
    border-radius: 20px;
    color: var(--ink);
    font-size: 0.82rem;
    padding: 0.3rem 0.9rem;
    transition: all 0.15s ease;
}
div[data-testid="stButton"] button:hover {
    border-color: var(--sage);
    color: var(--sage);
}


.cn-card {
    background: white;
    border: 1px solid var(--rule);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.cn-card.confident { border-left: 4px solid var(--sage); }
.cn-card.uncertain { border-left: 4px solid var(--ochre); }

.cn-result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.cn-result-label.confident { color: var(--sage); }
.cn-result-label.uncertain { color: var(--ochre); }

.cn-result-title {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0.2rem 0 0.3rem 0;
}
.cn-result-desc {
    color: var(--slate);
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

.cn-gauge-wrap {
    margin: 1rem 0 0.3rem 0;
}
.cn-gauge-label {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--slate);
    margin-bottom: 0.3rem;
}
.cn-gauge-track {
    position: relative;
    height: 10px;
    background: var(--parchment2);
    border-radius: 5px;
    overflow: visible;
}
.cn-gauge-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.4s ease;
}
.cn-gauge-fill.confident { background: var(--sage); }
.cn-gauge-fill.uncertain { background: var(--ochre); }
.cn-gauge-threshold {
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 2px;
    background: var(--ink);
    left: 85%;
}
.cn-gauge-threshold::after {
    content: "threshold 85%";
    position: absolute;
    top: -18px;
    left: 50%;
    transform: translateX(-50%);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink);
    white-space: nowrap;
}

.cn-evidence-intro {
    color: var(--slate);
    font-size: 0.85rem;
    margin-bottom: 0.8rem;
}
.cn-word-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.45rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.cn-word-chip {
    background: var(--parchment2);
    color: var(--ink);
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    min-width: 90px;
    text-align: center;
}
.cn-word-bar-track {
    flex: 1;
    height: 6px;
    background: var(--parchment2);
    border-radius: 3px;
    position: relative;
}
.cn-word-bar-fill {
    position: absolute;
    top: 0; bottom: 0;
    border-radius: 3px;
}
.cn-word-weight {
    width: 60px;
    text-align: right;
    color: var(--slate);
    font-size: 0.78rem;
}

.cn-candidate-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--parchment2);
    font-size: 0.92rem;
    color: var(--ink);
}
.cn-candidate-row:last-child { border-bottom: none; }
.cn-candidate-conf {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--slate);
}

section[data-testid="stSidebar"] {
    background: var(--parchment2);
    border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"] .cn-eyebrow { color: var(--ink); opacity: 0.6; }

.cn-journey-sidebar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.6rem 0 0.8rem 0;
}
.cn-journey-sidebar .num {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.1rem;
    font-weight: 600;
}
.cn-journey-sidebar .num.before { color: var(--ochre); }
.cn-journey-sidebar .num.after { color: var(--sage); }
.cn-journey-sidebar .arrow { color: var(--rule); }
.cn-journey-sidebar .cap {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--slate);
    text-transform: uppercase;
}

.cn-footer {
    margin-top: 2rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--rule);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.8rem;
}
.cn-footer-links {
    display: flex;
    gap: 1.2rem;
    font-size: 0.85rem;
}
.cn-footer-links a {
    color: var(--ink);
    text-decoration: none;
    border-bottom: 1px solid var(--rule);
}
.cn-footer-links a:hover { border-bottom-color: var(--sage); color: var(--sage); }
.cn-footer-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--sage);
    background: var(--sage-bg);
    border: 1px solid var(--sage);
    border-radius: 14px;
    padding: 0.25rem 0.8rem;
}
.cn-footer-stack {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--slate);
    width: 100%;
    margin-top: 0.4rem;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    return MedicalIntentClassifier()


CATEGORY_DESCRIPTIONS = {
    "causes": "What might have caused or triggered this",
    "exams and tests": "Diagnostic tests and how conditions are checked",
    "information": "General explanation of what a condition is",
    "inheritance": "Whether something is genetic or runs in families",
    "precautions": "What to avoid or do to stay safe",
    "side effects": "Reactions to medication or treatment",
    "symptoms": "What a condition feels like / physical signs",
    "treatment and outlook": "How something is treated and what to expect long-term",
    "when to contact a medical professional": "Whether and how urgently to seek care",
}

CATEGORY_EXAMPLE_MAP = {
    "causes": "what's actually triggering this",
    "exams and tests": "what kind of test will they do to check this",
    "information": "can someone explain what this actually means",
    "inheritance": "does this run in families",
    "precautions": "what should i avoid doing because of this",
    "side effects": "is feeling sick normal after starting this medicine",
    "symptoms": "what does this actually feel like",
    "treatment and outlook": "how do they actually treat this",
    "when to contact a medical professional": "is this serious enough to go to a and e",
}


def render_category_chips():
    cats = " · ".join(c.title() for c in CATEGORY_DESCRIPTIONS)
    st.markdown(
        f'<div class="cn-chips-plain">Classifies into 9 categories: {cats}</div>',
        unsafe_allow_html=True,
    )


def render_gauge(confidence, status):
    pct = confidence * 100
    st.markdown(
        f"""
        <div class="cn-gauge-wrap">
            <div class="cn-gauge-label">
                <span>CONFIDENCE</span>
                <span>{pct:.1f}%</span>
            </div>
            <div class="cn-gauge-track">
                <div class="cn-gauge-fill {status}" style="width:{pct}%"></div>
                <div class="cn-gauge-threshold"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence(explanation):
    max_abs = max(abs(w) for _, w in explanation) or 1.0
    st.markdown(
        '<div class="cn-evidence-intro">Word-level evidence — how much each token '
        'pushed the model toward this category.</div>',
        unsafe_allow_html=True,
    )
    for word, weight in explanation:
        color = "#2E7D52" if weight > 0 else "#D9650B"
        width_pct = min(abs(weight) / max_abs * 100, 100)
        bar_left = "50%" if weight >= 0 else f"{50 - width_pct/2}%"
        st.markdown(
            f"""
            <div class="cn-word-row">
                <span class="cn-word-chip">{word}</span>
                <div class="cn-word-bar-track">
                    <div class="cn-word-bar-fill" style="left:{bar_left}; width:{width_pct/2}%; background:{color};"></div>
                </div>
                <span class="cn-word-weight">{weight:+.3f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result(result):
    status = result["status"]

    if status == "confident":
        label = result["prediction"]
        conf = result["confidence"]

        st.markdown(
            f"""
            <div class="cn-card confident">
                <div class="cn-result-label confident">ROUTED &mdash; CONFIDENT</div>
                <div class="cn-result-title">{label.title()}</div>
                <div class="cn-result-desc">{CATEGORY_DESCRIPTIONS.get(label, "")}</div>
            """,
            unsafe_allow_html=True,
        )
        render_gauge(conf, "confident")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Why was this prediction made?"):
            render_evidence(result["explanation"])

    else:
        conf = result["confidence"]
        st.markdown(
            f"""
            <div class="cn-card uncertain">
                <div class="cn-result-label uncertain">FLAGGED &mdash; HUMAN REVIEW</div>
                <div class="cn-result-title">Below confidence threshold</div>
                <div class="cn-result-desc">The model isn't certain enough to route this automatically.</div>
            """,
            unsafe_allow_html=True,
        )
        render_gauge(conf, "uncertain")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="cn-eyebrow" style="margin-top:1.2rem;">POSSIBLE ROUTES, RANKED</div>',
            unsafe_allow_html=True,
        )
        rows = "".join(
            f'<div class="cn-candidate-row"><span>{c["label"].title()}</span>'
            f'<span class="cn-candidate-conf">{c["confidence"]:.1%}</span></div>'
            for c in result["candidates"]
        )
        st.markdown(f'<div class="cn-card" style="margin-top:0.4rem;">{rows}</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div class="cn-footer">
            <div class="cn-footer-links">
                <a href="https://github.com/umarkhan0206/medical-intent-classifier" target="_blank">GitHub</a>
                <a href="https://www.linkedin.com/in/umarkhan0206" target="_blank">LinkedIn</a>
                <a href="https://umarkhan0206.github.io" target="_blank">Portfolio</a>
            </div>
            <span class="cn-footer-badge">Open to work &middot; Sept 2026</span>
            <div class="cn-footer-stack">DistilBERT &middot; MedQuAD &middot; LIME &middot; Streamlit &middot; PyTorch &middot; 92.3% stress-test accuracy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.markdown(
        """
        <div class="cn-title"><span class="rx">⚕</span> Medical Intent Classifier</div>
        <div class="cn-subtitle">
            DistilBERT triage routing prototype &middot; type a patient query as you naturally would
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_category_chips()

    with st.sidebar:
        st.markdown('<div class="cn-eyebrow">ABOUT THIS PROJECT</div>', unsafe_allow_html=True)
        st.markdown(
            """
This model classifies patient queries into one of 9 clinical intent
categories, trained on the MedQuAD dataset and augmented with
LLM-generated realistic patient phrasing (Gemini + Groq).
            """
        )

        st.markdown('<div class="cn-eyebrow" style="margin-top:1.2rem;">THE JOURNEY</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="cn-journey-sidebar">
                <span class="num before">18%</span>
                <span class="arrow">&rarr;</span>
                <span class="num after">92.3%</span>
            </div>
            <div class="cap">real-world accuracy, before / after 12 LIME-diagnosed fixes</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The model first scored 100% on a standard test split — a red flag. "
            "An independent, leakage-checked stress test exposed it was actually "
            "only 18% reliable on natural phrasing."
        )

        st.markdown(
            '<div class="cn-eyebrow" style="margin-top:1.2rem;">KEY DESIGN CHOICES</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
- 85% confidence threshold triggers human review for uncertain queries
- LIME explainability shows which words drove each decision
- Validated on an independent, leakage-checked 104-question stress test,
  not just the standard train/test split
            """
        )
        st.divider()
        st.caption(
            "This is a routing aid, **not a diagnostic tool**. "
            "It does not replace clinical judgement."
        )
        st.divider()
        st.caption("Built by Umar Khan — MSc Advanced Computer Science, University of Leeds")

    classifier = load_classifier()

    st.markdown('<div class="cn-eyebrow">PATIENT QUERY</div>', unsafe_allow_html=True)
    query = st.text_input(
        " ",
        placeholder="e.g. did i get this from my mum or dad",
        label_visibility="collapsed",
    )

    

    st.markdown(
        '<div class="cn-eyebrow" style="margin-top:0.8rem;">TRY AN EXAMPLE</div>',
        unsafe_allow_html=True,
    )
    examples = [
        "what are the symptoms of diabetes",
        "did i get this from my mum or dad",
        "would a pharmacist be enough or do i need an actual doctor",
        "is hair loss something this treatment is known for",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex[:22] + ("..." if len(ex) > 22 else ""), key=ex):
            query = ex

    if query:
        with st.spinner("Classifying..."):
            result = classifier.classify(query)
        render_result(result)

    render_footer()


if __name__ == "__main__":
    main()