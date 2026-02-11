import os
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="Push dataset to Hugging Face Hub")
    parser.add_argument("--file", type=str, default="verified_pairs_improvement.jsonl", help="Path to the jsonl file")
    parser.add_argument("--repo_id", type=str, default="JasonYan777/verified_pairs_improvement", help="Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional if logged in via cli)")
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading dataset from {file_path}...")
    # Load as a json dataset
    dataset = load_dataset("json", data_files=file_path, split="train")

    print(f"Pushing dataset to {args.repo_id}...")
    dataset.push_to_hub(args.repo_id, token=args.token, private=args.private)
    print("Successfully pushed dataset to the Hub.")

if __name__ == "__main__":
    main()
