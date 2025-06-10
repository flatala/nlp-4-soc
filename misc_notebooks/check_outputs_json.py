import os
import json
from collections import defaultdict


def analyze_json_structures(directory="outputs/raw_outputs/"):
    """
    Analyze the structure of all JSON files in a directory and its subdirectories.
    Returns a dictionary with file paths as keys and their structures as values.
    """
    structures = {}
    found_files = []

    # Walk through all subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Analyzing {file_path}...")
                found_files.append(file_path)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        # Try to read the first item to determine structure
                        first_line = f.readline().strip()
                        if first_line.startswith("["):
                            # It's an array, read the whole file
                            f.seek(0)
                            data = json.load(f)
                            if data and isinstance(data, list) and data:
                                structures[file_path] = {
                                    "type": "array",
                                    "sample": data[0],
                                    "keys": list(data[0].keys())
                                    if isinstance(data[0], dict)
                                    else None,
                                    "count": len(data),
                                }
                        else:
                            # It might be a JSONL file (one JSON object per line)
                            f.seek(0)
                            data = []
                            for i, line in enumerate(f):
                                if i >= 5:  # Just read a few lines for analysis
                                    break
                                try:
                                    item = json.loads(line.strip())
                                    data.append(item)
                                except json.JSONDecodeError:
                                    pass

                            if data:
                                structures[file_path] = {
                                    "type": "jsonl",
                                    "sample": data[0],
                                    "keys": list(data[0].keys())
                                    if isinstance(data[0], dict)
                                    else None,
                                    "count": len(data),
                                }
                except Exception as e:
                    structures[file_path] = {"type": "error", "error": str(e)}

    print(f"Found {len(found_files)} JSON files")
    return structures


def print_structure_summary(structures):
    """Print a summary of the different structures found."""
    print(f"{len(structures)} JSON files analyzed with the following structures:\n")
    structure_types = defaultdict(list)

    for file_path, structure in structures.items():
        if structure["type"] in ["array", "jsonl"]:
            # Create a structure signature based on the keys
            if structure["keys"]:
                signature = tuple(sorted(structure["keys"]))
                structure_types[signature].append(file_path)
        else:
            structure_types[("error",)].append(file_path)

    print(f"\nFound {len(structure_types)} different structure types:")

    for i, (signature, files) in enumerate(structure_types.items(), 1):
        print(f"\nStructure Type {i}: {', '.join(signature)}")
        print(f"Found in {len(files)} files:")
        for file in files:
            print(f"  - {file}")
            # Print a sample from the first file
            if structures[file]["type"] != "error":
                print(
                    f"  Sample: {json.dumps(structures[file]['sample'], indent=2)[:200]}..."
                )


def convert_json_files(
    directory="outputs/raw_outputs/", output_dir="outputs/converted_outputs"
):
    """
    Convert JSON files from the found structure to the target structure.
    Source format: {'claim', 'evidences', 'id', 'label', 'predicted'}
    Target format: {'statement', 'label', 'evidences', 'explanation'}
    """
    os.makedirs(output_dir, exist_ok=True)
    converted_count = 0

    # Walk through all subdirectories
    for root, _, files in os.walk(directory):
        print(f"\nProcessing directory: {root}")
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Converting {file_path}...")

                try:
                    # Load source data
                    with open(file_path, "r", encoding="utf-8") as f:
                        source_data = json.load(f)

                    # Convert the data
                    converted_data = []
                    for item in source_data:
                        # Create a converted item with the target structure
                        predicted = item.get("predicted")
                        converted_item = {
                            "id": item.get("id"),
                            "statement": item.get("claim"),
                            "label": item.get("label"),
                            "evidences": item.get("evidences"),
                            "model_verdict": predicted.get("verdict"),
                            "explanation": predicted.get("explanation"),
                            "confidence": predicted.get("confidence"),
                        }
                        converted_data.append(converted_item)

                    # Create a similar directory structure in the output folder
                    rel_path = os.path.relpath(root, directory)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    print(f"Output directory created/loaded: {output_subdir}")

                    # Save the converted data
                    output_path = os.path.join(output_subdir, file)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(converted_data, f, indent=2, ensure_ascii=False)

                    converted_count += 1
                    print(f"Successfully converted to {output_path}")

                except Exception as e:
                    print(f"Error converting {file_path}: {e}")

    print(f"Conversion completed. {converted_count} files converted.")


def validate_conversions(
    original_dir="outputs/raw_outputs/", converted_dir="outputs/converted_outputs"
):
    """
    Validate that all files were properly converted and have the correct structure.
    """
    print("\nValidating conversions...")

    target_keys = ["statement", "label", "evidences", "explanation"]
    validation_issues = []

    for root, _, files in os.walk(converted_dir):
        print(f"\nValidating directory: {root}")
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(f"Validating {file_path}...")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if not isinstance(data, list):
                        validation_issues.append(f"{file_path}: Not a list")
                        continue

                    if not data:
                        validation_issues.append(f"{file_path}: Empty list")
                        continue

                    # Check the first item structure
                    first_item = data[0]
                    missing_keys = [key for key in target_keys if key not in first_item]

                    if missing_keys:
                        validation_issues.append(
                            f"{file_path}: Missing keys: {', '.join(missing_keys)}"
                        )

                    # Check corresponding original file has same number of items
                    rel_path = os.path.relpath(file_path, converted_dir)
                    original_path = os.path.join(original_dir, rel_path)

                    if os.path.exists(original_path):
                        with open(original_path, "r", encoding="utf-8") as f:
                            original_data = json.load(f)

                        if len(original_data) != len(data):
                            validation_issues.append(
                                f"{file_path}: Item count mismatch - Original: {len(original_data)}, Converted: {len(data)}"
                            )
                    else:
                        validation_issues.append(
                            f"{file_path}: Couldn't find original file for comparison"
                        )

                except Exception as e:
                    validation_issues.append(
                        f"{file_path}: Error during validation: {e}"
                    )

    if validation_issues:
        print("\nValidation issues found:")
        for issue in validation_issues:
            print(f" - {issue}")
    else:
        print("All conversions validated successfully!")


def analyze_json_files():
    """Analyze JSON files in the 'outputs' directory and print their structures."""
    structures = analyze_json_structures()
    print_structure_summary(structures)

    with open("sample_complex.json", "r") as f:
        target_format = json.load(f)

    print("\nTarget format structure:")
    print(f"Keys: {list(target_format[0].keys())}")
    print(f"Sample: {json.dumps(target_format[0], indent=2)}")


def convert_and_validate():
    """Convert JSON files in the 'outputs' directory to the target structure and validate the conversions."""
    convert_json_files()
    validate_conversions()


if __name__ == "__main__":
    if False:
        analyze_json_files()
        print("\nAnalysis complete.")
    convert_and_validate()
