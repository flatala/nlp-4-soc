"""
NOTE: Before running this script, ensure you have an .env file with a valid OpenAI API key:

Run this script with (e.g.):
    python fact_check_pipeline.py --input test_claims.jsonl --model gpt-4o
"""

import sys
import argparse
import json
import random
import os
import re
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from glob import glob
from os.path import join

load_dotenv()


def load_records(path: str) -> list:
    """
    Load records from a JSONL or plain JSON file.
    If the file starts with '[', it's treated as a JSON array; otherwise, JSONL.
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


class FactCheckPipeline:
    """
    Pipeline to verify claims against provided evidence, with optional few-shot.
    """

    VERIFY_PROMPT = staticmethod(
        lambda c, e, e1, e2, e3, e4: f"""
Evaluate if this claim is supported by the evidence. Reply with a JSON object with:
- verdict: "SUPPORTED", "NOT_SUPPORTED"
- confidence: a number between 0 and 1
- explanation: brief explanation of your verdict

Use the following examples as a guide:
Example 1: {e1}

Example 2: {e2}

Example 3: {e3}

Example 4: {e4}

Now evaluate:
Claim: {c}
Evidence: {e}
"""
    )

    def __init__(
        self,
        model: str,
        shots: int = 0,
        examples_dir: str = None,
        use_azure_openai: bool = False,
    ):
        self.name = model
        if use_azure_openai:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2025-01-01-preview",
            )
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
                with open(fn, "r", encoding="utf-8") as f:
                    e_txts[i] = f.read().strip()

        prompt = self.VERIFY_PROMPT(claim, evidence, *e_txts)
        response = self.query_openai(prompt)
        # print("RAW MODEL OUTPUT ➜", response[:400], "...\n")
        # Try to extract JSON object from the response string
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass  # will handle below

        # If extraction or parsing fails, return error dict
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "explanation": "Failed to parse model response",
        }

    def query_openai(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.name,
                store=True,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fact-check claims in JSON/JSONL using OpenAI, with gold evidence"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        default="sample_complex.json",
        help="Path to input JSONL or JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write JSONL predictions (default stdout)",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name")
    parser.add_argument(
        "--examples",
        type=str,
        default="examples_no_cot",
        help="Directory of up to 4 .txt few-shot files",
    )
    parser.add_argument(
        "--shots", type=int, default=4, help="Number of few-shot examples to include"
    )
    parser.add_argument(
        "--use-azure-openai",
        action="store_true",
        help="Use Azure OpenAI instead of OpenAI API",
    )

    args = parser.parse_args()

    records = load_records(args.input)
    pipe = FactCheckPipeline(
        model=args.model,
        shots=args.shots,
        examples_dir=args.examples,
        use_azure_openai=args.use_azure_openai,
    )

    out = open(args.output, "w", buffering=1) if args.output else sys.stdout
    # NOTE: for the sample_complex.json we use "statement" instead of claim. For the unified datasets we do use claim.
    count = 0
    for record in records:
        claim = record["claim"]
        gold = record.get("evidences", [])
        result = pipe(claim, gold)
        record["predicted"] = result
        out.write(json.dumps(record) + "\n")
        count += 1
        if count == 2050:
            break

    if args.output:
        out.close()
