import json
from faithfulness_eval_utils.entailment_eval import evaluate_entailment
from faithfulness_eval_utils.ragas_eval import evaluate_ragas
from faithfulness_eval_utils.geval import evaluate_geval

async def evaluate_faithfulness(
    data,
    result_dir: str,
    tokenizer_path: str = "roberta-large-mnli",
    model_path: str = "roberta-large-mnli",
    geval_model: str = "llama3:8b",
    ragas_model: str = "llama3:8b",
    results_filename: str = "results.json",
    stats_filename: str = "stats.json",
    eval_types = ['enaitlment', 'ragas', 'geval']
):
    
    if 'ragas' in eval_types:
        await evaluate_ragas(
            data, 
            result_dir=f"{result_dir}/ragas", 
            evaluator_model=ragas_model, 
            results_filename=results_filename,
            stats_filename=stats_filename
        )

    if 'geval' in eval_types:
        await evaluate_geval(
            data, 
            result_dir=f"{result_dir}/ragas", 
            evaluator_model=geval_model, 
            results_filename=results_filename,
            stats_filename=stats_filename
        )

    if 'entailment' in eval_types:
        evaluate_entailment(
            data, 
            result_dir=f"{result_dir}/entailment", 
            tokenizer_path=tokenizer_path, 
            model_path=model_path, 
            results_filename=results_filename,
            stats_filename=stats_filename
        )