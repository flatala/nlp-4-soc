import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def evaluate_entailment(
    data,
    result_dir: str,
    tokenizer_path: str = "roberta-large-mnli",
    model_path: str = "roberta-large-mnli",
    results_filename: str = "results.json",
    stats_filename: str = "stats.json"
):
    
    os.makedirs(result_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    detailed = []
    e2x_scores, x2e_scores = [], []

    for item in data:
        premise = " ".join(item["evidences"])
        hypo   = item["explanation"]

        # evidence → explanation
        enc_fwd   = tokenizer(premise, hypo, return_tensors="pt", truncation=True)
        logits_fwd= nli_model(**enc_fwd).logits
        prob_fwd  = torch.softmax(logits_fwd, dim=1)[0, 2].item()

        # explanation → evidence
        enc_bwd   = tokenizer(hypo, premise, return_tensors="pt", truncation=True)
        logits_bwd= nli_model(**enc_bwd).logits
        prob_bwd  = torch.softmax(logits_bwd, dim=1)[0, 2].item()

        item["e2x_entail_prob"] = prob_fwd
        item["x2e_entail_prob"] = prob_bwd
        detailed.append(item)

        e2x_scores.append(prob_fwd)
        x2e_scores.append(prob_bwd)


    with open(os.path.join(result_dir, results_filename), "w") as f:
        json.dump(detailed, f, indent=2)

    stats = {
        "avg_e2x_entail_prob": sum(e2x_scores) / len(e2x_scores) if e2x_scores else 0.0,
        "avg_x2e_entail_prob": sum(x2e_scores) / len(x2e_scores) if x2e_scores else 0.0,
    }

    with open(os.path.join(result_dir, stats_filename), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved {len(detailed)} items to `{result_dir}/{results_filename}`")
    print(f"Averages written to `{result_dir}/{stats_filename}`")
    return detailed, stats