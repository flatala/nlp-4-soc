# To implement ARES for scoring your RAG system and comparing to other RAG configurations, you need three components:​

# A human preference validation set of annotated query, document, and answer triples for the evaluation criteria (e.g. context relevance, answer faithfulness, and/or answer relevance). There should be at least 50 examples but several hundred examples is ideal.
# A set of few-shot examples for scoring context relevance, answer faithfulness, and/or answer relevance in your system
# A much larger set of unlabeled query-document-answer triples outputted by your RAG system for scoring
#
# TERMINOLOGY:
# IDP: In-Domain Prompts
# UES: unlabeled Evaluation Set
# ARES: automataed RAG Evaluation System
#
# NOTES:
# KILT db is 37GB...
# patched to work:
# General_Binary_Classifier.py:
#   replace `from datasets import load_metric` with `import evaluate`
#   replace `load_metric` with `evaluate.laod`
# Prepate_KILT_dataset.py:
#     add argument to `load_dataset` trust_remote_code=True
import os
from dotenv import load_dotenv
from ares import ARES

load_dotenv()


def run_ues_idp():
    """
    Step 1: Run Unbiased Evaluation from Supervised LLM Judge Scoring
    This step uses LLM-based judge scoring with in-domain prompts on unlabeled data
    """
    azure_openai_host = os.getenv("AZURE_OPENAI_HOST")
    openai_model = os.getenv("OPENAI_MODEL")

    ues_idp_config = {
        "in_domain_prompts_dataset": "nq_few_shot_prompt_for_judge_scoring.tsv",
        "unlabeled_evaluation_set": "nq_unlabeled_output.tsv",
        "model_choice": "gpt-3.5-turbo-0125",
    }

    if openai_model is not None:
        ues_idp_config["model_choice"] = openai_model
    if azure_openai_host is not None:
        ues_idp_config["azure_openai_host"] = azure_openai_host

    ares = ARES(ues_idp=ues_idp_config)
    results = ares.ues_idp()
    print("UES-IDP Results:")
    print(results)
    return results


def generate_synthetic_data():
    """
    Step 2: Generate synthetic queries for training
    This step creates synthetic data based on the labeled output
    """
    synth_config = {
        "document_filepaths": ["nq_labeled_output.tsv"],
        "few_shot_prompt_filename": "nq_few_shot_prompt_for_synthetic_query_generation.tsv",
        "synthetic_queries_filenames": ["synthetic_queries_1.tsv"],
        "documents_sampled": 6189,
    }

    ares_module = ARES(synthetic_query_generator=synth_config)
    results = ares_module.generate_synthetic_data()
    print("Synthetic Data Generation Results:")
    print(results)
    return results


def train_classifier():
    """
    Step 3: Train the classifier model
    This step trains a classifier model using synthetic and labeled data
    """
    classifier_config = {
        "training_dataset": ["synthetic_queries_1.tsv"],
        "validation_set": ["nq_labeled_output.tsv"],
        "label_column": ["Context_Relevance_Label"],
        "num_epochs": 10,
        "patience_value": 3,
        "learning_rate": 5e-6,
        "assigned_batch_size": 1,
        "gradient_accumulation_multiplier": 32,
    }

    ares = ARES(classifier_model=classifier_config)
    results = ares.train_classifier()
    print("Classifier Training Results:")
    print(results)
    return results


def evaluate_rag():
    """
    Step 4: Run Prediction-Powered Inference (PPI) for RAG evaluation
    This step evaluates the RAG system using trained models

    Expected Output:
    Context_Relevance_Label Scoring
    ARES Ranking
    ARES Prediction: [0.6056978059262574]
    ARES Confidence Interval: [[0.547, 0.664]]
    Number of Examples in Evaluation Set: [4421]
    Ground Truth Performance: [0.6]
    ARES LLM Judge Accuracy on Ground Truth Labels: [0.789]
    Annotated Examples used for PPI: 300
    """
    ppi_config = {
        "evaluation_datasets": ["nq_unlabeled_output.tsv"],
        "checkpoints": ["Context_Relevance_Label_nq_labeled_output_date_time.pt"],
        "rag_type": "question_answering",
        "labels": ["Context_Relevance_Label"],
        "gold_label_path": "nq_labeled_output.tsv",
    }

    ares = ARES(ppi=ppi_config)
    results = ares.evaluate_RAG()
    print("RAG Evaluation Results:")
    print(results)
    return results


def download_dataset():
    """
    Fetches NQ datasets with ratios including 0.5, 0.6, 0.7, etc.
    For purposes of our quick start guide, we rename nq_ratio_0.5 to nq_unlabeled_output and nq_labeled_output.
    """
    ares = ARES()
    ares.KILT_dataset("nq")


if __name__ == "__main__":
    # Run each step of the ARES evaluation pipeline
    download_dataset()

    run_ues_idp()
    generate_synthetic_data()
    train_classifier()
    evaluate_rag()

    print("ARES evaluation completed successfully.")
