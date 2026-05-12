# Calibration Analysis

## Reliability diagram interpretation

Every bar in the diagram sits **below** the perfect-calibration diagonal, meaning the model is **over-confident** across all confidence levels — its stated confidence consistently exceeds its actual accuracy.

Key bucket values read from the diagram:

- **0.3–0.4 bucket**: empirical accuracy ≈ 0.33, confidence ≈ 0.35 — a small ~2-point gap, closest to calibrated.
- **0.5–0.6 bucket**: empirical accuracy ≈ 0.48, confidence ≈ 0.55 — a ~7-point gap.
- **0.6–0.7 bucket**: empirical accuracy ≈ 0.51, confidence ≈ 0.65 — a ~14-point gap.
- **0.7–0.8 bucket**: empirical accuracy ≈ 0.55, confidence ≈ 0.75 — a ~20-point gap, the worst in the diagram.
- **0.8–0.9 bucket**: empirical accuracy ≈ 0.71, confidence ≈ 0.85 — a ~14-point gap.
- **0.9–1.0 bucket**: empirical accuracy ≈ 0.86, confidence ≈ 0.95 — still ~9 points below the diagonal.

The 0.0–0.2 buckets are empty, which is expected — fine-tuned transformers rarely output very low softmax values.

## Expected Calibration Error

**ECE ≈ 0.12** *(replace with the exact value printed by `calibration.py`)*.

An ECE of ~0.12 means the model overstates its confidence by 12 percentage points on average. When it says it is 75% confident, it is correct only about 55% of the time. For production use this is a meaningful risk: any downstream system that gates on a confidence threshold (e.g. routing high-confidence predictions to auto-approve) will pass through far more errors than expected. Raw probability scores from this model should not be trusted for threshold-based decisions without recalibration.

## A specific calibration pattern

The most striking pattern is **over-confidence in the 0.7–0.8 bucket**, where the model claims ~75% confidence but achieves only ~55% accuracy — a 20-point gap. This is the hallmark of a model that learned strong surface-level cues (e.g. words like "great", "terrible") during fine-tuning and fires high-confidence predictions whenever those cues appear, even on reviews where the overall sentiment is mixed or ambiguous. Because the AARSynth dataset skews positive, the model over-learned positive-class features and pushes softmax output into the 0.7–0.8 range for many borderline positive/neutral examples, inflating confidence without a matching improvement in accuracy.

## A proposed engineering action

Apply **temperature scaling with T > 1.0** (softening). After training, hold out ~10% of the training data as a calibration split not used during fine-tuning. Search for the scalar T that minimises negative log-likelihood on that split when logits are divided by T before softmax. For an over-confident model T will be greater than 1.0 — typically 1.2–1.8 — which flattens the distribution and moves bars toward the diagonal. This requires no retraining, does not change accuracy or predicted class labels at all, and is a single scalar applied at inference time. After finding T, re-plot the reliability diagram on a held-out validation set and confirm ECE drops below 0.05 before using probability scores in any threshold-based downstream logic.
