# Medical Intent Classifier

A DistilBERT-based NHS triage routing prototype — and the story of how a model that scored **100% accuracy** turned out to be only **18% reliable**, and what it took to actually fix that.

---

## The model that lied to me

Early in this project, a fine-tuned DistilBERT model hit 100% accuracy on the standard train/test split for classifying patient queries into 10 clinical intent categories (symptoms, causes, treatment, when to seek care, etc).

A perfect score on a held-out test set should be a red flag, not a celebration — and it was. I built an independent, hand-written set of realistic patient questions ("did i get this from my mum or dad", "would a pharmacist be enough or do i need an actual doctor") that never touched the training pipeline. The model scored **18%**.

It had memorised the *shape* of the augmented training data, not the *meaning* of patient language.

I used [LIME](https://github.com/marcotcr/lime) to find out why, and the answer changed with every fix:

| Round | What LIME showed | What I changed |
|---|---|---|
| 1 | Predictions leaned on generic filler words (`this`, `my`, `am`) instead of medical content | Rewrote the augmentation to inject real content words per category |
| 2 | A single connector word (`after`) had become a spurious signal for one category because it appeared disproportionately in that category's templates | Rebalanced templates so no connector word was category-exclusive |
| 3 | Two categories (`treatment`, `outlook`) were genuinely ambiguous to a human, not just to the model | Merged them — a clinical/structural decision, not a data fix |
| 4 onward | Specific, named failure patterns (`"runs on my dad's side"`, `"wait it out or get seen"`, physical symptom phrasing) | Generated targeted training examples for each gap using Gemini + Groq |

I also caught myself shipping a leakage bug: a "fix" round had accidentally copied verbatim phrases from the test set into the training templates, inflating the score. I audited for it, found it, removed it, and rebuilt the test set from scratch to be sure.

**12 iterations later: 92.3% on a 104-question, leakage-checked, independently-written stress test** — not the inflated train/test number.

---

## Why this matters for a triage system

A model that's *confidently wrong* is more dangerous than one that knows it's unsure. So the final system doesn't just classify — it knows when not to:

- **85% confidence threshold.** Below it, the system doesn't guess — it surfaces the top 3 candidate categories for human review instead.
- Across the stress test, **89.4%** of queries were confident and correctly routed automatically; for the remaining **uncertain 10.6%**, the true category was still in the top 3 **90.9%** of the time.
- **LIME explainability** on every confident prediction — see exactly which words drove the decision, not just the output.

This is a routing aid, **not a diagnostic tool**, and the interface says so. It does not replace clinical judgement.

---

## Methodology

**Dataset:** [MedQuAD](https://github.com/abachaa/MedQuAD) (47k+ medical Q&A pairs, NIH sources), balanced to 1,000 questions per category, 10 → merged to 9 final categories.

**Baselines:** Multinomial Naive Bayes (91% accuracy, 0.20 F1 on the "information" class — couldn't distinguish general questions from specific ones) and Random Forest (1.00 F1 on clean data — classic lexical memorisation, confirmed by collapsing on noisy/colloquial input).

**Augmentation pipeline (12 iterations):**
- v1–v2: rule-based synonym replacement and typo injection
- v3: comprehensive handwritten phrase banks (LIME-driven, fixed generic-word reliance)
- v4–v6: LLM-generated phrasing (Gemini + Groq) for natural diversity at scale
- v7: merged `treatment` + `outlook` after confirming the ambiguity was genuine, not just a data gap — while explicitly keeping `when to contact a medical professional` separate, since collapsing the urgency category for a small accuracy gain would have undermined the system's actual safety value
- v8–v12: targeted generation against every specific, LIME-diagnosed failure pattern, re-validated against an expanding, leakage-audited stress test each round

**Final model:** DistilBERT (`distilbert-base-uncased`), fine-tuned 3 epochs, AdamW, `max_length=64`, trained on Aire HPC (University of Leeds), GPU runs of ~50 seconds per epoch.

**Validation:** a 104-question stress test, written independently of the training pipeline, audited twice for data leakage (once after catching a real leak). This — not the train/test split — is the number reported here.

| Metric | Train/test split | Independent stress test |
|---|---|---|
| Naive Bayes | 91% acc / 0.20 F1 on "information" | not tested (baseline only) |
| Random Forest | 1.00 F1 (clean data) | collapses on noisy phrasing |
| DistilBERT (early) | **100%** | **18%** |
| DistilBERT (final, v12) | 99.4% | **92.3%** |

---

## Running the demo

```bash
git clone https://github.com/umarkhan0206/medical-intent-classifier.git
cd medical-intent-classifier
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m streamlit run app.py
```

The model is hosted on [Hugging Face Hub](https://huggingface.co/umarkhan0206/medical-intent-classifier) and downloads automatically on first run — no manual setup needed beyond the steps above.

## Project structure