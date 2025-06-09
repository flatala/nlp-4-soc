import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
try:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM
    from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from typing import List, Dict, Any

def load_fact_check_data(file_path: str) -> List[Dict[Any, Any]]:
    """Load the fact-checking JSON data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []

def convert_to_ragas_format(data: List[Dict[Any, Any]]) -> Dataset:
    """Convert fact-checking data to RAGAS evaluation format."""
    
    # Prepare data for RAGAS evaluation
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in data:
        # Use the statement as the question
        question = item.get('statement', '').strip('"')
        questions.append(question)
        
        # Use the model explanation as the answer
        answer = item.get('explanation', '')
        answers.append(answer)
        
        # Use evidences as context
        evidences = item.get('evidences', [])
        context = ' '.join(evidences) if evidences else ''
        contexts.append([context])  # RAGAS expects list of contexts
        
        # Use the original label as ground truth
        ground_truth = item.get('label', '')
        # Convert to more descriptive ground truth
        if ground_truth == 'SUPPORTED':
            gt_text = f"The statement is SUPPORTED by the evidence."
        elif ground_truth == 'NOT_SUPPORTED':
            gt_text = f"The statement is NOT SUPPORTED by the evidence."
        else:
            gt_text = f"The statement label is {ground_truth}."
        
        ground_truths.append(gt_text)
    
    # Create dataset
    dataset_dict = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    }
    
    return Dataset.from_dict(dataset_dict)

def setup_llm_and_embeddings(provider: str = "ollama", **kwargs):
    """
    Setup LLM and embeddings for RAGAS evaluation.
    
    Args:
        provider: Either "ollama" or "azure"
        **kwargs: Additional configuration parameters
    
    Returns:
        Tuple of (llm, embeddings)
    """
    
    if provider == "ollama":
        # Setup Ollama (local)
        llm_model = kwargs.get('llm_model', 'llama3.1:8b')  # Default model
        embedding_model = kwargs.get('embedding_model', 'nomic-embed-text')
        
        print(f"Setting up Ollama with LLM: {llm_model}, Embeddings: {embedding_model}")
        
        # Create Ollama LLM and embeddings
        llm = OllamaLLM(model=llm_model, temperature=0.1)
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        return ragas_llm, ragas_embeddings
        
    elif provider == "azure":
        # Setup Azure OpenAI
        azure_endpoint = kwargs.get('azure_endpoint', os.getenv('AZURE_OPENAI_ENDPOINT'))
        api_key = kwargs.get('api_key', os.getenv('AZURE_OPENAI_API_KEY'))
        api_version = kwargs.get('api_version', '2024-02-01')
        llm_deployment = kwargs.get('llm_deployment', 'gpt-4')
        embedding_deployment = kwargs.get('embedding_deployment', 'text-embedding-ada-002')
        
        if not azure_endpoint or not api_key:
            raise ValueError("Azure endpoint and API key must be provided")
        
        print(f"Setting up Azure OpenAI with LLM: {llm_deployment}, Embeddings: {embedding_deployment}")
        
        # Create Azure OpenAI LLM and embeddings
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=llm_deployment,
            temperature=0.1
        )
        
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=embedding_deployment
        )
        
        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        return ragas_llm, ragas_embeddings
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def evaluate_fact_checking_with_ragas(
    file_path: str, 
    output_path: str = None, 
    provider: str = "ollama",
    **llm_kwargs
) -> Dict[str, float]:
    """
    Evaluate fact-checking output using RAGAS metrics.
    
    Args:
        file_path: Path to the JSON file containing fact-checking results
        output_path: Optional path to save detailed results
        provider: Either "ollama" or "azure"
        **llm_kwargs: Additional LLM configuration parameters
    
    Returns:
        Dictionary containing evaluation metrics
    """
    
    # Load data
    data = load_fact_check_data(file_path)
    if not data:
        return {}
    
    # Convert to RAGAS format
    print("Converting data to RAGAS format...")
    dataset = convert_to_ragas_format(data)
    
    # Print sample data for verification
    print("\nSample data:")
    print(f"Question: {dataset['question'][0]}")
    print(f"Answer: {dataset['answer'][0][:200]}...")
    print(f"Context: {dataset['contexts'][0][0][:200]}...")
    print(f"Ground Truth: {dataset['ground_truth'][0]}")
    
    # Setup LLM and embeddings
    try:
        llm, embeddings = setup_llm_and_embeddings(provider, **llm_kwargs)
    except Exception as e:
        print(f"Error setting up LLM/embeddings: {e}")
        return {}
    
    # Import RAGAS with proper configuration
    from ragas import RunConfig
    
    # Create run config with custom LLM and embeddings
    run_config = RunConfig(
        max_workers=1,  # Process one at a time to avoid overwhelming local models
        max_wait=300,   # 5 minute timeout per evaluation
    )
    
    print(f"\nEvaluating with RAGAS metrics...")
    print(f"Dataset size: {len(dataset)}")
    
    try:
        # Run evaluation with custom LLM and embeddings
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_similarity,
                answer_correctness
            ],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config
        )
        
        print("\n" + "="*50)
        print("RAGAS EVALUATION RESULTS")
        print("="*50)
        
        # Display results
        for metric_name, score in result.items():
            if isinstance(score, (int, float)):
                print(f"{metric_name}: {score:.4f}")
        
        # Save detailed results if output path provided
        if output_path:
            # Convert to DataFrame for easier analysis
            df_results = result.to_pandas()
            df_results.to_csv(output_path.replace('.json', '_detailed.csv'), index=False)
            
            # Save summary results
            summary_results = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            with open(output_path, 'w') as f:
                json.dump(summary_results, f, indent=2)
            
            print(f"\nDetailed results saved to: {output_path.replace('.json', '_detailed.csv')}")
            print(f"Summary results saved to: {output_path}")
        
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {}

def analyze_results_by_label(file_path: str, results_df: pd.DataFrame = None):
    """Analyze RAGAS results by original fact-checking labels."""
    
    data = load_fact_check_data(file_path)
    if not data or results_df is None:
        return
    
    # Add original labels to results
    labels = [item.get('label', '') for item in data]
    results_df['original_label'] = labels
    
    print("\n" + "="*50)
    print("ANALYSIS BY FACT-CHECKING LABEL")
    print("="*50)
    
    # Group by label and calculate mean scores
    grouped = results_df.groupby('original_label').agg({
        'faithfulness': 'mean',
        'answer_relevancy': 'mean',
        'context_precision': 'mean',
        'context_recall': 'mean',
        'answer_similarity': 'mean',
        'answer_correctness': 'mean'
    }).round(4)
    
    print(grouped)
    
    # Count distribution
    print(f"\nLabel Distribution:")
    print(results_df['original_label'].value_counts())

def main():
    """Main function to run RAGAS evaluation."""
    
    # Configuration
    config = {
        "provider": "ollama",  # Change to "azure" if using Azure OpenAI
        "input_file": "inference_outputs/converted_outputs/mistral_7b_non_cot/politi_hop_mistral_7b_no_cot.json",
        "output_file": "ragas_evaluation_results.json"
    }
    
    # Ollama configuration (adjust models as needed)
    ollama_config = {
        "llm_model": "llama3.1:8b",           # Can use: llama3.1:8b, llama3:7b, mistral:7b, etc.
        "embedding_model": "nomic-embed-text"  # Can use: nomic-embed-text, all-minilm, etc.
    }
    
    # Azure configuration (uncomment and fill if using Azure)
    # azure_config = {
    #     "azure_endpoint": "https://your-resource.openai.azure.com/",
    #     "api_key": "your-api-key",
    #     "api_version": "2024-02-01",
    #     "llm_deployment": "gpt-4",
    #     "embedding_deployment": "text-embedding-ada-002"
    # }
    
    # Check if file exists
    if not os.path.exists(config["input_file"]):
        print(f"Input file not found: {config['input_file']}")
        print("Please ensure the file path is correct.")
        return
    
    # Choose configuration based on provider
    llm_kwargs = ollama_config if config["provider"] == "ollama" else {}
    
    print(f"Starting RAGAS evaluation using {config['provider']}...")
    
    # Run evaluation
    results = evaluate_fact_checking_with_ragas(
        config["input_file"], 
        config["output_file"],
        provider=config["provider"],
        **llm_kwargs
    )
    
    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"Overall Performance Summary:")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
    else:
        print("Evaluation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()