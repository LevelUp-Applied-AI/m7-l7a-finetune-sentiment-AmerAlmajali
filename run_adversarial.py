"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return model, tokenizer
    # TODO: AutoModelForSequenceClassification.from_pretrained(model_path)
    # TODO: AutoTokenizer.from_pretrained(model_path)
    # TODO: return both
    raise NotImplementedError


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    df = pd.read_csv(adv_csv_path)
    predicted_labels = []
    predicted_probabilities = []
    correct_list = []

    # Put model in evaluation mode
    model.eval()

    # Loop through every example
    for _, row in df.iterrows():

        text = row["text"]
        true_label = row["expected_label"]

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Disable gradient calculation
        with torch.no_grad():

            # Forward pass
            outputs = model(**inputs)

            # Convert logits -> probabilities
            probs = torch.softmax(outputs.logits, dim=-1)

            # Get best class index
            pred_idx = torch.argmax(probs, dim=-1).item()

            # Confidence of predicted class
            pred_prob = probs[0][pred_idx].item()

        # Convert index -> label name
        pred_label = model.config.id2label[pred_idx]

        # Check correctness
        is_correct = pred_label == true_label

        # Store results
        predicted_labels.append(pred_label)
        predicted_probabilities.append(pred_prob)
        correct_list.append(is_correct)

    # Add new columns
    df["predicted_label"] = predicted_labels
    df["predicted_probability"] = predicted_probabilities
    df["correct"] = correct_list

    return df

    # TODO: read adv_csv_path with pandas
    # TODO: for each row, tokenize + forward pass + softmax + argmax
    # TODO: convert argmax index to label name via model.config.id2label
    # TODO: build a results DataFrame with predicted_label, predicted_probability, correct
    # TODO: return the DataFrame
    raise NotImplementedError


def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    main()
