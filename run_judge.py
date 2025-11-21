import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

def parse_verdict(evaluation_text):
    clean_text = evaluation_text.strip()
    if clean_text.lower().startswith("summary a"):
        return "Summary A"
    if clean_text.lower().startswith("summary b"):
        return "Summary B"
    
    if "Summary A" in evaluation_text and "Summary B" not in evaluation_text:
        return "Summary A"
    if "Summary B" in evaluation_text and "Summary A" not in evaluation_text:
        return "Summary B"
        
    match = re.search(r"Verdict\**:\s*(Summary [AB])", evaluation_text, re.IGNORECASE)
    if match:
        return match.group(1)
        
    match = re.search(r"Winner\**:\s*(Summary [AB])", evaluation_text, re.IGNORECASE)
    if match:
        return match.group(1)
        
    return "Unknown"

def judge_pair(client, source, summary_a, summary_b):
    prompt = (
        f"Here is a source text:\n{source}\n\n"
        f"Summary A:\n{summary_a}\n\n"
        f"Summary B:\n{summary_b}\n\n"
        "Which summary is better? Please answer with 'Summary A' or 'Summary B' and provide a brief explanation."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "You are an expert summarization evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"Judge error: {e}")
        return "Error"

def main():
    load_dotenv()
    
    judged_path = "results/test_2_refinement_judged.json"
    input_path = "results/test_2_refinement.json"
    
    if os.path.exists(judged_path):
        print(f"Loading existing judged results from {judged_path}...")
        with open(judged_path, "r") as f:
            results = json.load(f)
    elif os.path.exists(input_path):
        print(f"Loading unjudged results from {input_path}...")
        with open(input_path, "r") as f:
            results = json.load(f)
            
        try:
            with open("data/pairwise_test.jsonl", "r") as f:
                test_data = [json.loads(line) for line in f]
                
            print("Backfilling source text...")
            for res in results:
                idx = res['id']
                if idx < len(test_data):
                    user_content = test_data[idx]['messages'][0]['content']
                    try:
                        source_text = user_content.split("Here is a source text:\n")[1].split("\n\nSummary A:\n")[0]
                        res['source'] = source_text
                    except IndexError:
                        pass
        except FileNotFoundError:
            print("Warning: data/pairwise_test.jsonl not found, skipping backfill.")
    else:
        print("Error: No results file found.")
        return

    # 3. Run Judge
    print("\n--- Running LLM Judge on Refinement Results ---")
    client = OpenAI() 
    
    wins_baseline_1 = 0
    wins_baseline_2 = 0
    ties = 0
    
    for i, res in enumerate(results):
        if 'judge_verdict' in res and res['judge_verdict'] and res['judge_verdict'] != "Error":
            verdict = res['judge_verdict']
            print(f"Pair {i+1}/{len(results)}: Using cached verdict.")
        else:
            print(f"Judging pair {i+1}/{len(results)}...")
            verdict = judge_pair(client, res['source'], res['baseline_1_artifact'], res['baseline_2_artifact'])
            res['judge_verdict'] = verdict
        
        parsed = parse_verdict(verdict)
        print(f"  Verdict: {parsed}")
        
        if parsed == "Summary A":
            wins_baseline_1 += 1
        elif parsed == "Summary B":
            wins_baseline_2 += 1
        else:
            ties += 1
            
    with open("results/test_2_refinement_judged.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Judge Results ---")
    print(f"Baseline 1 (Base Critic) Wins: {wins_baseline_1}")
    print(f"Baseline 2 (Trained Critic) Wins: {wins_baseline_2}")
    print(f"Ties/Unknown: {ties}")
    
    total = wins_baseline_1 + wins_baseline_2 + ties
    if total > 0:
        print(f"Win Rate for Trained Critic: {wins_baseline_2/total:.2%}")

if __name__ == "__main__":
    main()
