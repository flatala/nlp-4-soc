import os
import asyncio
import json
import re
import ollama
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama


def clean_text(text):
    if isinstance(text, str):
        # Remove trailing commas and unnecessary whitespace
        text = re.sub(r",\s*$", "", text.strip())
        return text
    elif isinstance(text, list):
        return [clean_text(item) for item in text]
    return text


async def evaluate_ragas(
    data,
    result_dir: str,
    results_filename: str = "results.json",
    stats_filename: str = "stats.json",
    evaluator_model: str = "llama3:8b",
):
    try:
        ollama.show(evaluator_model)
    except Exception as _:
        print(
            f"Ollama model {evaluator_model} not found. Attempting to pull it from Ollama..."
        )
        ollama.pull(evaluator_model)

    evaluator_llm = ChatOllama(
        model=evaluator_model,
        temperature=0,
    )
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=evaluator_model,
            temperature=0,
        )
    )

    os.makedirs(result_dir, exist_ok=True)
    scorer = Faithfulness(llm=evaluator_llm)

    async def score_sample(item):
        sample = SingleTurnSample(
            user_input=item["statement"],
            response=item["explanation"],
            retrieved_contexts=item["evidences"],
        )

        try:
            score = await scorer.single_turn_ascore(sample)
        except Exception as e:
            print(f"⚠️ Scoring failed: {e}")
            score = -1.0

        return {
            "statement": item["statement"],
            "explanation": item["explanation"],
            "evidences": item["evidences"],
            "faithfulness_score": float(score),
        }

    results = await asyncio.gather(*(score_sample(item) for item in data))

    # Save results
    with open(os.path.join(result_dir, results_filename), "w") as f:
        json.dump(results, f, indent=4)

    # Compute average faithfulness score
    faithfulness_scores = [
        item["faithfulness_score"]
        for item in results
        if item["faithfulness_score"] != -1.0
    ]
    avg_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores
        else 0.0
    )

    stats = {"avg_faithfulness_score": avg_faithfulness}

    # Save stats
    with open(os.path.join(result_dir, stats_filename), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved {len(results)} items to `{result_dir}/{results_filename}`")
    print(f"Averages written to `{result_dir}/{stats_filename}`")

    return results, stats
