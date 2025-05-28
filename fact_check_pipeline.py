"""
NOTE: Before running this script, ensure you have the Ollama server running locally:

- Open a terminal and run:
    ollama serve

- Open **another** terminal and run:
    ollama run mistral:7b

Finally, run this script with (e.g.):
    python fact_check_pipeline.py --input test_claims.jsonl --model deepseek-r1:32b 
    NOTE: (the model must match the one you ran with ollama)
"""
import sys
import argparse
import json
import requests
import random
from glob import glob
from os.path import join

def load_records(path: str) -> list:
    """
    Load records from a JSONL or plain JSON file.
    If the file starts with '[', it's treated as a JSON array; otherwise, JSONL.
    """
    with open(path, 'r') as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

class FactCheckPipeline:
    """
    Pipeline to verify claims against provided evidence, with optional few-shot.
    """

    VERIFY_PROMPT = staticmethod(lambda c, e, e1, e2, e3, e4: f"""
Evaluate if this claim is supported by the evidence. Reply with a JSON object with:
- verdict: "SUPPORTED", "NOT_SUPPORTED"
- confidence: a number between 0 and 1
- explanation: brief explanation of your verdict

Use the following examples as a guide to do chain of thought reasoning:
Example 1: {e1}

Example 2: {e2}

Example 3: {e3}

Example 4: {e4}

Now evaluate:
Claim: {c}
Evidence: {e}
""")

    def __init__(self, model: str, shots: int = 0, examples_dir: str = None):
        self.name = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.shots = shots
        self.examples_dir = examples_dir

    def __call__(self, claim: str, evidence=None) -> dict:
        # Use only the provided evidence (do not fetch from API)
        if isinstance(evidence, list):
            evidence_str = "\n".join(evidence)
        else:
            evidence_str = evidence or ""

        verdict = self.verify_claim(claim, evidence_str)
        return {"evidence": evidence_str, **verdict}

    def verify_claim(self, claim: str, evidence: str) -> dict:
        # Load up to `shots` .txt few-shot examples
        e_txts = [""] * 4
        if self.examples_dir and self.shots > 0:
            files = glob(join(self.examples_dir, "*.txt"))
            random.shuffle(files)
            for i, fn in enumerate(files[: self.shots]):
                with open(fn, 'r') as f:
                    e_txts[i] = f.read().strip()

        prompt = self.VERIFY_PROMPT(claim, evidence, *e_txts)
        response = self.query_ollama(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "explanation": "Failed to parse model response"
            }

    def query_ollama(self, prompt: str) -> str:
        data = {
            "model": self.name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = requests.post(self.ollama_url, json=data, timeout=30)
            resp.raise_for_status()
            j = resp.json()
            if "response" in j:
                return j["response"].strip()
            if isinstance(j.get("choices"), list) and j["choices"]:
                return j["choices"][0].get("text", "").strip()
            return j.get("text", "").strip()
        except requests.RequestException as e:
            return f"Error: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fact-check claims in JSON/JSONL using Ollama, with gold evidence"
    )
    parser.add_argument(
        "--input", type=str, required=True, default="/Users/s4m/Documents/TUD/Q4/NLP for Society/nlp-4-soc/sample_complex.json",
        help="Path to input JSONL or JSON file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Where to write JSONL predictions (default stdout)"
    )
    parser.add_argument(
        "--model", type=str, default="mistral:7b",
        help="Ollama model name (default: deepseek-r1:32b)"
    )
    parser.add_argument(
        "--examples", type=str, default="examples",
        help="Directory of up to 4 .txt few-shot files"
    )
    parser.add_argument(
        "--shots", type=int, default=4,
        help="Number of few-shot examples to include"
    )

    args = parser.parse_args()

    records = load_records(args.input)
    pipe = FactCheckPipeline(
        model=args.model,
        shots=args.shots,
        examples_dir=args.examples
    )

    out = open(args.output, 'w') if args.output else sys.stdout
    for record in records:
        claim = record["claim"]
        # Only use the "evidences" field (ignore "label" or other keys)
        gold = record.get("evidences", [])
        record["predicted"] = pipe(claim, gold)
        out.write(json.dumps(record) + "\n")
    if args.output:
        out.close()
