import os
import re
import asyncio 
import json

def build_prompt(claim, explanation, evidences, label):
    evidence_str = "\n".join([f"{i+1}. {e}" for i, e in enumerate(evidences)])
    return f"""
    You are an expert fact-checking evaluator. Your task is to assess the **faithfulness** of an explanation given a claim and supporting evidence.

    Definition of "faithfulness":
    An explanation is faithful if it only uses information present in the evidence and does not add unsupported claims or hallucinations.

    Evaluate the following:

    Claim: "{claim}"
    Label: "{label}"

    Evidence:
    "{evidence_str}"

    Explanation:
    "{explanation}"

    Please answer:
    1. Does the explanation make any factual claims not found in the evidence? (Yes/No)
    2. Does the explanation faithfully reflect the key points in the evidence? (Yes/No)
    3. Faithfulness Score (0–5): How faithful is the explanation to the evidence and claim?
    4. Justify your score in 1–2 sentences.
        """.strip()

async def evaluator_llm(prompt):
    # Call model
    return 

async def evaluate_geval(
    data,
    result_dir: str,
    results_filename: str = "results.json",
    stats_filename: str = "stats.json"
    ):

    os.makedirs(result_dir, exist_ok=True)

    async def score_sample(item):
        prompt = build_prompt(
            claim=item["statement"],
            explanation=item["explanation"],
            evidences=item["evidences"],
            label=item["label"]
        )
        raw_response = await evaluator_llm(prompt)  # your async call

        # Parse answers using regex or simple line splits
        lines = raw_response.strip().splitlines()

        # Defaults
        q1 = q2 = justification = ""
        score = 0.0

        for line in lines:
            line = line.strip()
            if line.startswith("1."):
                q1 = line[2:].strip()
            elif line.startswith("2."):
                q2 = line[2:].strip()
            elif line.lower().startswith("3."):
                # Extract number after colon
                match = re.search(r"(\d+(\.\d+)?)", line)
                if match:
                    score = float(match.group(1))
            elif line.lower().startswith("4."):
                justification = line[2:].strip()

        return {
            "statement": item["statement"],
            "explanation": item["explanation"],
            "evidences": item["evidences"],
            "label": item["label"],
            "q1_factual_claims_not_in_evidence": q1,
            "q2_reflects_key_points": q2,
            "faithfulness_score_0_5": score,
            "justification": justification,
            "raw_response": raw_response
        }

    detailed = await asyncio.gather(*(score_sample(item) for item in data))

    with open(os.path.join(result_dir, results_filename), "w") as f:
        json.dump(detailed, f, indent=4)

    scores = [item["faithfulness_score_0_5"] for item in detailed]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    stats = {
        "avg_geval_score_0_5": avg_score
    }

    with open(os.path.join(result_dir, stats_filename), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved {len(detailed)} items to `{result_dir}/{results_filename}`")
    print(f"Averages written to `{result_dir}/{stats_filename}`")

    return detailed, stats