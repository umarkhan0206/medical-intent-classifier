"""
Inference pipeline for the medical intent classifier.

Combines:
  1. DistilBERT classification
  2. Human-in-the-loop confidence thresholding (default 85%)
  3. LIME-based word-level explainability for confident predictions

Used by the Streamlit demo app.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert_medical_intent_v12"
MAX_LENGTH = 64
CONFIDENCE_THRESHOLD = 0.85


class MedicalIntentClassifier:
    def __init__(self, model_dir=MODEL_DIR, threshold=CONFIDENCE_THRESHOLD):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.eval()

        with open(Path(model_dir) / "label_map.json", "r") as f:
            label_map = json.load(f)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(list(label_map.keys()))

        self.threshold = threshold
        self.explainer = LimeTextExplainer(class_names=list(self.label_encoder.classes_))

    def _predict_proba(self, texts):
        results = []
        for t in texts:
            inputs = self.tokenizer(
                t, return_tensors="pt", truncation=True,
                padding="max_length", max_length=MAX_LENGTH
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            results.append(probs)
        return np.array(results)

    def classify(self, text: str) -> dict:
        """Classify text with confidence thresholding."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding="max_length", max_length=MAX_LENGTH
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        top3_idx = torch.argsort(probs, descending=True)[:3]

        candidates = [
            {"label": self.label_encoder.inverse_transform([idx.item()])[0],
             "confidence": probs[idx].item()}
            for idx in top3_idx
        ]
        top_confidence = candidates[0]["confidence"]

        result = {
            "query": text,
            "confidence": top_confidence,
            "candidates": candidates,
        }

        if top_confidence >= self.threshold:
            result["status"] = "confident"
            result["prediction"] = candidates[0]["label"]
            result["explanation"] = self._explain(text, result["prediction"])
        else:
            result["status"] = "uncertain"
            result["prediction"] = None
            result["explanation"] = None

        return result

    def _explain(self, text, predicted_label, num_features=6, num_samples=200):
        class_names = list(self.label_encoder.classes_)
        label_idx = class_names.index(predicted_label)
        explanation = self.explainer.explain_instance(
            text, self._predict_proba, num_features=num_features,
            num_samples=num_samples, labels=[label_idx]
        )
        return explanation.as_list(label=label_idx)


if __name__ == "__main__":
    clf = MedicalIntentClassifier()
    for q in [
        "what are the symptoms of diabetes",
        "did i get this from my mum or dad",
        "would a pharmacist be enough or do i need an actual doctor",
    ]:
        result = clf.classify(q)
        print(f"\nQuery: {q}")
        print(f"Status: {result['status']}")
        if result["status"] == "confident":
            print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
            for word, weight in result["explanation"]:
                print(f"  {word:15s} {weight:+.4f}")
        else:
            for c in result["candidates"]:
                print(f"  {c['label']}: {c['confidence']:.2%}")