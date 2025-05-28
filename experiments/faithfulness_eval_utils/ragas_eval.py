import os
import asyncio
import json
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import Faithfulness

# Initialize LLM as judge model here
evaluator_llm = None

async def evaluate_ragas(
    data,
    result_dir: str,
    results_filename: str = "results.json",
    stats_filename: str = "stats.json"
    ):
    
    os.makedirs(result_dir, exist_ok=True)

    scorer = Faithfulness(llm=evaluator_llm)

    async def score_sample(item):
        sample = SingleTurnSample(
            user_input=item["statement"],
            response=item["explanation"],
            retrieved_contexts=item["evidences"]
        )
        score = await scorer.single_turn_ascore(sample)
        return {
            "statement": item["statement"],
            "explanation": item["explanation"],
            "evidences": item["evidences"],
            "faithfulness_score": float(score)
        }

    results = await asyncio.gather(*(score_sample(item) for item in data))

    # Save results
    with open(os.path.join(result_dir, results_filename), "w") as f:
        json.dump(results, f, indent=4)

    # Compute average faithfulness score
    faithfulness_scores = [item["faithfulness_score"] for item in results]
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0

    stats = {
        "avg_faithfulness_score": avg_faithfulness
    }

    # Save stats
    with open(os.path.join(result_dir, stats_filename), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Saved {len(results)} items to `{result_dir}/{results_filename}`")
    print(f"Averages written to `{result_dir}/{stats_filename}`")

    return results, stats
