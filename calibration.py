"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import os
import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Binning convention: bin edges are np.linspace(0, 1, n_bins + 1).
    A probability p falls in bin i if edges[i] <= p < edges[i+1], with
    the last bin inclusive on the right (so p == 1.0 lands in the last
    bin, not out of range).

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bucket_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    # np.digitize gives 1-indexed bins; subtract 1 for 0-indexed.
    # Clip so that p == 1.0 (which digitize places in bin n_bins) lands in
    # the last valid bin (n_bins - 1), matching the spec's inclusive-right rule.
    bucket_indices = np.clip(np.digitize(confidences, bin_edges) - 1, 0, n_bins - 1)

    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bucket_indices == b
        bucket_counts[b] = int(np.sum(mask))
        if bucket_counts[b] > 0:
            bucket_accuracies[b] = float(np.mean(predictions[mask] == y_true[mask]))
        # else: leave accuracy as 0.0 for empty buckets

    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    bucket_confidence is the mean of the max predicted probabilities within the bucket.
    A perfectly calibrated model has ECE = 0.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bucket_indices = np.clip(np.digitize(confidences, bin_edges) - 1, 0, n_bins - 1)

    N = len(y_true)
    ece = 0.0

    for b in range(n_bins):
        mask = bucket_indices == b
        count = np.sum(mask)
        if count == 0:
            continue
        accuracy = float(np.mean(predictions[mask] == y_true[mask]))
        confidence = float(np.mean(confidences[mask]))
        ece += (count / N) * abs(accuracy - confidence)

    return float(ece)


def plot_reliability(
    centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str
) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(
        centers,
        accs,
        width=width * 0.9,
        edgecolor="black",
        alpha=0.8,
        label="Empirical accuracy",
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from manual_eval import manual_predict

    # Load model and tokenizer from Lab 7A output
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")

    # Load test data from predictions.csv (already has true labels)
    df = pd.read_csv("predictions.csv")
    texts = df["text"].tolist()
    label2idx = {v: k for k, v in model.config.id2label.items()}
    y_true = df["label"].map(label2idx).to_numpy(dtype=int)

    # Run inference
    _, probs = manual_predict(model, tokenizer, texts)

    # Reliability diagram
    os.makedirs("figures", exist_ok=True)
    centers, accs, counts = reliability_diagram(probs, y_true)
    plot_reliability(
        centers, accs, counts, output_path="figures/reliability-diagram.png"
    )
    print("Reliability diagram saved to figures/reliability-diagram.png")

    # ECE
    ece = expected_calibration_error(probs, y_true)
    print(f"ECE = {ece:.4f}")


if __name__ == "__main__":
    main()
