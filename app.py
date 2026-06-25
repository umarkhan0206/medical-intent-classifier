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
    page_icon="🩺",
    layout="centered",
)


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


def render_result(result):
    if result["status"] == "confident":
        label = result["prediction"]
        conf = result["confidence"]

        st.success(f"**Routed to: {label.title()}**")
        st.caption(CATEGORY_DESCRIPTIONS.get(label, ""))
        st.progress(conf, text=f"Confidence: {conf:.1%}")

        with st.expander("Why was this prediction made?"):
            st.markdown(
                "The words below show how much each contributed to the model's "
                "decision. Positive values pushed *towards* this category."
            )
            for word, weight in result["explanation"]:
                bar_color = "🟩" if weight > 0 else "🟥"
                st.write(f"{bar_color} `{word}` — {weight:+.3f}")

    else:
        conf = result["confidence"]
        st.warning(
            f"**Confidence too low ({conf:.1%}) to route automatically.** "
            "This query has been flagged for human review."
        )
        st.markdown("**Possible routes, ranked by likelihood:**")
        for c in result["candidates"]:
            st.write(f"- {c['label'].title()} — {c['confidence']:.1%}")


def main():
    st.title("🩺 NHS Medical Intent Classifier")
    st.markdown(
        "A DistilBERT-based triage routing prototype. Type a patient query "
        "below as you naturally would — informal phrasing, typos, and all."
    )

    with st.sidebar:
        st.header("About this project")
        st.markdown(
            """
This model classifies patient queries into one of 9 clinical intent
categories, trained on the MedQuAD dataset and augmented with
LLM-generated realistic patient phrasing (Gemini + Groq).

**Key design choices:**
- 85% confidence threshold triggers human review for uncertain queries
- LIME explainability shows which words drove each decision
- Validated on an independent, leakage-checked 104-question stress test
  (92.3% accuracy), not just the standard train/test split

This is a routing aid, **not a diagnostic tool**. It does not replace
clinical judgement.
            """
        )
        st.divider()
        st.caption("Built by Umar Khan — MSc Advanced Computer Science, University of Leeds")

    classifier = load_classifier()

    query = st.text_input(
        "Patient query",
        placeholder="e.g. did i get this from my mum or dad",
    )

    examples = [
        "what are the symptoms of diabetes",
        "did i get this from my mum or dad",
        "would a pharmacist be enough or do i need an actual doctor",
        "is hair loss something this treatment is known for",
    ]
    st.caption("Try an example:")
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex[:20] + "...", key=ex):
            query = ex

    if query:
        with st.spinner("Classifying..."):
            result = classifier.classify(query)
        render_result(result)


if __name__ == "__main__":
    main()