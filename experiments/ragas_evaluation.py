# %%
import json
import os
import asyncio
from time import time
from faithfulness_eval_utils.ragas_eval import evaluate_ragas

# %%
EVALUATOR_MODEL = "llama3:8b"

MODEL = "gpt4o_non_cot"
# MODEL CHOICES ['mistral_7b_cot', 'mistral_7b_non_cot', 'gpt4o_cot', 'gpt4o_non_cot', 'deepseek_r1_32b_cot']


async def main():
    t1 = time()
    directory_path = f"../inference/converted_outputs/{MODEL}/"
    all_files = os.listdir(directory_path)
    json_files = [f for f in all_files if f.endswith(".json")]

    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        print(f"Processing file: {file_path}...")
        with open(file_path, "r") as fin:
            data = json.load(fin)
            t2 = time()
            stats_output_filename = f"{filename.replace('.json', '.stats.json')}"
            results_output_filename = f"{filename.replace('.json', '.results.json')}"

            await evaluate_ragas(
                data,
                evaluator_model=EVALUATOR_MODEL,
                results_filename=results_output_filename,
                stats_filename=stats_output_filename,
                result_dir=f"output/ragas/{MODEL}",
            )
            t3 = time()
            print(
                f"Processed {filename} in {(t3 - t2)/60:.2f} minutes. Total elapsed time: {(t3 - t1)/60:.2f} minutes."
            )


# Run the async function
if __name__ == "__main__":
    asyncio.run(main())

# %%
