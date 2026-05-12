# Calibration Analysis

## Reliability diagram interpretation

Every bar in the diagram sits **below** the perfect-calibration diagonal, meaning the model is **under-confident** across all confidence levels — it hedges more than the data warrants.

Key bucket values read from the diagram:

- **0.3–0.4 bucket**: empirical accuracy ≈ 0.34, confidence ≈ 0.35 — nearly on the diagonal, the closest to calibrated.
- **0.5–0.6 bucket**: empirical accuracy ≈ 0.48, confidence ≈ 0.55 — a ~7-point gap.
- **0.6–0.7 bucket**: empirical accuracy ≈ 0.50, confidence ≈ 0.65 — a ~15-point gap, the widest in the mid-range.
- **0.8–0.9 bucket**: empirical accuracy ≈ 0.71, confidence ≈ 0.85 — a ~14-point gap.
- **0.9–1.0 bucket**: empirical accuracy ≈ 0.85, confidence ≈ 0.95 — still ~10 points below the diagonal.

The 0.0–0.2 buckets are empty, which is expected — fine-tuned transformers rarely output very low softmax values.

## Expected Calibration Error

**ECE ≈ 0.12** (replace with the value printed by `calibration.py`).

An ECE of ~0.12 means that on average the model's stated confidence overshoots its actual accuracy by 12 percentage points in the wrong direction — when it says 70% confident it is correct only about 55% of the time. For production use this means raw probability scores **cannot be used directly** for threshold-based routing or confidence badges surfaced to end users; they must be recalibrated first. The risk here is less severe than over-confidence (the model won't falsely suppress human review) but still material — any downstream system that gates on a confidence threshold of e.g. 0.8 will pass through far more uncertain predictions than intended.

## A specific calibration pattern

The dominant pattern is **uniform under-confidence caused by the neutral class acting as a probability sink**. On a 3-class problem (positive / neutral / negative), whenever the model is uncertain between positive and negative it spreads probability mass toward neutral rather than committing. This systematically drags the max-class probability below the true accuracy, producing the consistent below-diagonal pattern visible across every bucket. The effect is largest in the 0.6–0.7 range — exactly the confidence band where positive/neutral boundary cases cluster — where the gap reaches ~15 points. This pattern arose because cross-entropy training penalises confident wrong predictions, so the model learned to hedge on ambiguous reviews rather than commit, and neutral provided a convenient hedge target.

## A proposed engineering action

Apply **temperature scaling with T < 1.0** (sharpening). After training, hold out ~10% of the training data as a calibration split (not used during fine-tuning). Search for the scalar T that minimises negative log-likelihood on that split when logits are divided by T before softmax. For an under-confident model T will be less than 1.0 — typically in the 0.6–0.9 range — which sharpens the distribution and moves bars toward the diagonal. This requires no retraining, does not change accuracy at all, and can be wrapped as a one-line post-processing step at inference time. After finding T, re-plot the reliability diagram on a separate validation set and confirm ECE drops below 0.05 before trusting probability scores in any downstream threshold logic.
