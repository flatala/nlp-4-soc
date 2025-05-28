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

import sys, os, argparse
import json, requests
import glob


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
    Pipeline to retrieve evidence and verify claims, with optional few-shot prompting using external text files.
    """
    # Prompt templates
    EVIDENCE_PROMPT = staticmethod(lambda text: f"""
    You are acting as a knowledge retrieval system. Provide relevant factual information 
    about this claim: {text}
    """
    )

    VERIFY_PROMPT = staticmethod(lambda c, e, e1, e2, e3, e4: f"""
    Evaluate if this claim is supported by the evidence. Reply with a JSON object with:
    - verdict: \"SUPPORTED\" or \"NOT_SUPPORTED\"
    - confidence: a number between 0 and 1
    - explanation: brief explanation of your verdict

    Use the following examples as a guide:
    Example 1:\n{e1}\n
    Example 2:\n{e2}\n
    Example 3:\n{e3}\n
    Example 4:\n{e4}\n
    Now evaluate:
    Claim: {c}
    Evidence: {e}
    """
    )

    def __init__(self, model: str):
        self.name = model
        self.ollama_url = "http://localhost:11434/api/generate"

    def __call__(self, claim: str) -> dict:
        evidence = self.get_evidence(claim)
        verdict = self.verify_claim(claim, evidence)
        return {"evidence": evidence, **verdict}

    def get_evidence(self, claim: str) -> str:
        prompt = self.EVIDENCE_PROMPT(claim)
        return self.query_ollama(prompt)

    def verify_claim(self, claim: str, evidence: str) -> dict:
        prompt = self._build_verify_prompt(claim, evidence)
        response = self.query_ollama(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "explanation": "Failed to parse model response"
            }

    def _build_verify_prompt(self, claim: str, evidence: str) -> str:
        """
        Load four example texts from examples/*.txt and fill into the VERIFY_PROMPT.
        """
        # Find up to 4 example text files
        paths = sorted(glob.glob(os.path.join('examples', '*.txt')))[:4]
        texts = []
        for p in paths:
            try:
                with open(p, 'r') as f:
                    texts.append(f.read().strip())
            except IOError:
                texts.append('')
        assert len(texts) == 4, "Expected exactly 4 example files in 'examples/' directory"
        # Unpack four examples
        e1, e2, e3, e4 = texts
        return self.VERIFY_PROMPT(claim, evidence, e1, e2, e3, e4)

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
        description="Fact-check claims in a JSON or JSONL dataset using Ollama with file-based few-shot examples"
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
        "--model", type=str, default="deepseek-r1:32b",
        help="Ollama model name (default: deepseek-r1:32b)"
    )
    args = parser.parse_args()

    records = load_records(args.input)
    pipe = FactCheckPipeline(model=args.model)
    out = open(args.output, 'w') if args.output else sys.stdout

    for record in records:
        claim = record.get("claim", "")
        record["predicted"] = pipe(claim)
        out.write(json.dumps(record) + "\n")

    if args.output:
        out.close()
