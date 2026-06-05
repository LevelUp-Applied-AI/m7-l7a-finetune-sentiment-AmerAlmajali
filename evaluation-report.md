# Module 7 Week A — Lab Evaluation Report

## Dataset
The AARSynth app reviews dataset contains 7,472 reviews across 9 apps with 3 sentiment classes (negative, neutral, positive), split 80/20 into 5,977 training examples and 1,495 test examples using seed=42.

## Model and Hyperparameters
- **Backbone:** distilbert-base-uncased
- **Number of labels:** 3
- **Learning rate:** 5e-5
- **Epochs:** 2
- **Batch size:** 8
- **Max length:** 128
- **Seed:** 42
- **Training time:** ~75 minutes on CPU (Windows, Intel Core i7)

## Metrics on the Test Split

Aggregate:

| Metric   | Value  |
|----------|--------|
| Accuracy | 0.6321 |
| Macro-F1 | 0.6307 |

Per class (read from `metrics.json`):

| Class    | F1     | Precision | Recall |
|----------|--------|-----------|--------|
| Negative | 0.7103 | 0.7131    | 0.7074 |
| Neutral  | 0.4898 | 0.4659    | 0.5162 |
| Positive | 0.6922 | 0.7248    | 0.6623 |

## Confusion Matrix

|          | negative | neutral | positive |
|----------|----------|---------|----------|
| negative | 353      | 128     | 18       |
| neutral  | 108      | 239     | 116      |
| positive | 34       | 146     | 353      |

## Three Qualitative Error Examples

### Example 1 — Gold: `negative` → Predicted: `positive`
- **Text:** *"i have had this app for a while and it works really well"*
- **Gold label:** negative
- **Predicted label:** positive
- **Predicted probability for gold label:** 0.0112

This is likely a labeling artifact in the dataset — the sentence reads as clearly positive to both the model and a human reader ("works really well"). The model's prediction is arguably correct, suggesting the gold label may have been assigned based on context outside the review text (e.g., a low star rating despite positive wording). This highlights a known challenge with sentiment datasets where rating-based labels conflict with the actual review language.

---

### Example 2 — Gold: `neutral` → Predicted: `negative`
- **Text:** *"no way to redeem tickets."*
- **Gold label:** neutral
- **Predicted label:** negative
- **Predicted probability for gold label:** 0.0216

The phrase "no way" is a strong negative cue that the model latched onto, pushing it toward negative. A human annotator might read this as a neutral feature request or factual complaint rather than an emotionally negative review — but the language is brief and lacks any balancing phrase, making the distinction subtle. Short, blunt sentences like this are systematically hard for the model because there is little context to disambiguate tone from fact.

---

### Example 3 — Gold: `positive` → Predicted: `negative`
- **Text:** *"it is a waste app and waste of time. it is only useful for downloading the movies"*
- **Gold label:** positive
- **Predicted label:** negative
- **Predicted probability for gold label:** 0.0072

This is the most informative error: the review opens with strong negative language ("waste app", "waste of time") but the gold label is positive, likely because the annotator weighted the second clause ("useful for downloading movies") more heavily. The model, however, overweighted the dominant negative phrases in the first sentence — a classic mixed-sentiment failure where the model captures the loudest signal rather than the overall intent. This pattern is common in reviews that criticize specific limitations while still recommending the app for a narrow use case.

## Hugging Face Hub Model URL
https://huggingface.co/ameralmajali1/m7-app-review-sentiment
