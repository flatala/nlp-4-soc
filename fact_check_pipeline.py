import sys, argparse
import json, requests


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
    # Prompt to retrieve supporting evidence for a claim
    EVIDENCE_PROMPT = staticmethod(lambda text: f"""
    You are acting as a knowledge retrieval system. Provide relevant factual information 
    about this claim: {text}
    """)

    # Prompt to verify a claim against provided evidence
    VERIFY_PROMPT = staticmethod(lambda c, e: f"""
    Evaluate if this claim is supported by the evidence. Reply with a JSON object with:
    - verdict: \"SUPPORTED\", \"REFUTED\", or \"NOT_ENOUGH_INFO\"
    - confidence: a number between 0 and 1
    - explanation: brief explanation of your verdict
    
    Claim: {c}
    Evidence: {e}
    """)

    def __init__(self, model: str):
        self.name = model
        self.ollama_url = "http://localhost:11434/api/generate"

    def fact_check(self, text: str) -> dict:
        evidence = self.get_evidence(text)
        verdict = self.verify_claim(text, evidence)
        return {
            "evidence": evidence,
            **verdict
        }
    __call__ = fact_check

    def get_evidence(self, claim: str) -> str:
        prompt = self.EVIDENCE_PROMPT(claim)
        return self.query_ollama(prompt)

    def verify_claim(self, claim: str, evidence: str) -> dict:
        prompt = self.VERIFY_PROMPT(claim, evidence)
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

            # 1. If Ollama returns 'response', use that
            if "response" in j:
                return j["response"].strip()
            # 2. Otherwise look for 'choices'[0]['text']
            if isinstance(j.get("choices"), list) and j["choices"]:
                return j["choices"][0].get("text", "").strip()
            # 3. Finally fall back to top-level 'text'
            return j.get("text", "").strip()
        except requests.RequestException as e:
            return f"Error: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fact-check claims in a JSON or JSONL dataset using Ollama"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input JSONL or JSON file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write predictions (JSONL). Defaults to stdout."
    )
    parser.add_argument(
        "--model", type=str, default="mistral:7b",
        help="Ollama model name (default: mistral:7b)"
    )
    args = parser.parse_args()

    # Load input records (JSON or JSONL)
    records = load_records(args.input)

    pipe = FactCheckPipeline(model=args.model)
    out = open(args.output, 'w') if args.output else sys.stdout

    for record in records:
        claim = record.get("claim", "")
        record["predicted"] = pipe(claim)
        out.write(json.dumps(record) + "\n")

    if args.output:
        out.close()
