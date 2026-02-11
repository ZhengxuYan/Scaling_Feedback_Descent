import json
import os
from typing import List, Dict
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from bespokelabs import curator

load_dotenv()

class RationaleOutput(BaseModel):
    verdict: str = Field(description="The verdict of the comparison, either 'Summary A' or 'Summary B'.")
    rationale: str = Field(description="A brief explanation of why one summary is better than the other.")

class SummarizationJudge(curator.LLM):
    def prompt(self, input: Dict) -> List[Dict[str, str]]:
        source = input["source"]
        summary_a = input["summary_a"]
        summary_b = input["summary_b"]
        
        user_content = (
            f"Here is a source text:\n{source}\n\n"
            f"Summary A:\n{summary_a}\n\n"
            f"Summary B:\n{summary_b}\n\n"
            "Which summary is better? Please answer with 'Summary A' or 'Summary B' and provide a brief explanation.\n"
            "Provide your response in JSON format with keys 'verdict' and 'rationale'."
        )
        
        return [
            {"role": "system", "content": "You are an expert summarization evaluator. Output JSON only."},
            {"role": "user", "content": user_content}
        ]

    def parse(self, input: Dict, response: str) -> Dict:
        # Check agreement with ground truth
        try:
            # Clean response to ensure it's valid JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            
            data = json.loads(response)
            model_verdict = data.get("verdict", "Unknown")
            model_rationale = data.get("rationale", "")
        except:
            model_verdict = "Unknown"
            model_rationale = ""
            
        ground_truth_winner = input["ground_truth_winner"]
        
        # Normalize verdict string
        if "summary a" in model_verdict.lower():
            clean_verdict = "Summary A"
        elif "summary b" in model_verdict.lower():
            clean_verdict = "Summary B"
        else:
            clean_verdict = "Unknown"
            
        is_agreement = (clean_verdict == ground_truth_winner)
        
        return {
            **input,
            "model_verdict": clean_verdict,
            "model_rationale": model_rationale,
            "is_agreement": is_agreement
        }

def create_synthetic_rationale(winner_cand, loser_cand):
    """
    Synthesize a rationale based on scores, including detailed metrics if available.
    """
    w_metrics = winner_cand.get('details', {})
    l_metrics = loser_cand.get('details', {})
    
    def get_scores(metrics):
        for key in ['feedback-c3', 'feedback-c4']:
            if key in metrics and metrics[key]:
                return metrics[key]
        return {}

    w_scores = get_scores(w_metrics)
    l_scores = get_scores(l_metrics)
    
    details = []
    for metric in ['faithfulness_score', 'completeness_score', 'conciseness_score']:
        w_val = float(w_scores.get(metric, 0))
        l_val = float(l_scores.get(metric, 0))
        if w_val > l_val:
            details.append(f"better {metric.replace('_score', '')} ({w_val} vs {l_val})")
        elif l_val > w_val:
            details.append(f"worse {metric.replace('_score', '')} ({w_val} vs {l_val})")
            
    reason_text = ", ".join(details) if details else "better overall quality"
    
    return f"The winner produced a better summary because it has {reason_text}. (Overall Score: {winner_cand['score']} vs {loser_cand['score']})"

def construct_comparison_data():
    print("Loading dataset...")
    dataset = load_dataset("DISLab/FeedSum", split="train")
    
    grouped_data = {}
    
    print("Grouping data by ID...")
    for example in dataset:
        doc_id = example['doc_id']
        
        if doc_id not in grouped_data:
            grouped_data[doc_id] = {
                "document": example['document'],
                "candidates": []
            }
        
        try:
            f_score = float(example['feedback-c1']['score']) if 'feedback-c1' in example else 0
            c_score = float(example['feedback-c2']['score']) if 'feedback-c2' in example else 0
            total_score = (f_score + c_score) / 2
        except:
            total_score = 0

        grouped_data[doc_id]['candidates'].append({
            "summary": example['summary'],
            "model_name": example['summarizer'],
            "score": total_score,
            "details": example
        })

    threshold = 1000
    print(f"Filtering for contexts > {threshold} words...")

    judge = SummarizationJudge(model_name="gpt-5-mini")
    inputs = []

    print("Creating pairs...")
    import random
    for doc_id, data in grouped_data.items():
        # Check document length
        if len(data['document'].split()) <= threshold:
            continue

        cands = data['candidates']
        
        if len(cands) < 2:
            continue
            
        cand_a = cands[0]
        cand_b = cands[1]
        
        # Randomize order to avoid position bias
        if random.random() < 0.5:
            cand_a, cand_b = cand_b, cand_a
        
        if cand_a['score'] > cand_b['score']:
            ground_truth_winner = "Summary A"
            winner_cand = cand_a
            loser_cand = cand_b
        elif cand_b['score'] > cand_a['score']:
            ground_truth_winner = "Summary B"
            winner_cand = cand_b
            loser_cand = cand_a
        else:
            continue
            
        inputs.append({
            "source": data['document'],
            "summary_a": cand_a['summary'],
            "summary_b": cand_b['summary'],
            "ground_truth_winner": ground_truth_winner,
            "winner_cand": winner_cand,
            "loser_cand": loser_cand
        })

    print(f"Processing {len(inputs)} pairs with Curator...")
    results = judge(inputs)
    
    # Handle CuratorResponse
    if hasattr(results, "dataset"):
        results_list = list(results.dataset)
    else:
        results_list = list(results)
        
    training_examples = []
    model_usage_count = 0
    
    for res in results_list:
        ground_truth_winner = res["ground_truth_winner"]
        winner_cand = res["winner_cand"]
        loser_cand = res["loser_cand"]
        
        # Fallback rationale
        template_rationale = create_synthetic_rationale(winner_cand, loser_cand)
        
        if res["is_agreement"]:
            final_rationale = res["model_rationale"]
            model_usage_count += 1
        else:
            final_rationale = template_rationale
            
        user_content = (
            f"Here is a source text:\n{res['source']}\n\n"
            f"Summary A:\n{res['summary_a']}\n\n"
            f"Summary B:\n{res['summary_b']}\n\n"
            "Which summary is better and why?"
        )
        
        # If model rationale is used, we assume it's a good explanation.
        # We can format it nicely if it's just the explanation text.
        # The model output 'rationale' field is just the explanation string.
        
        assistant_content = f"**Verdict**: {ground_truth_winner}\n\n**Feedback**: {final_rationale}"
        
        training_examples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "used_model": res["is_agreement"]
        })
    
    print(f"Created {len(training_examples)} comparison pairs.")
    print(f"Used model rationale for {model_usage_count} pairs ({model_usage_count/len(training_examples):.1%})")
    
    # Remove metadata key before saving
    final_examples = []
    for ex in training_examples:
        final_examples.append({"messages": ex["messages"]})

    # Split into train and test
    test_size = 20
    if len(final_examples) > test_size:
        test_data = final_examples[:test_size]
        train_data = final_examples[test_size:]
    else:
        print("Warning: Not enough data for split. Using all for train.")
        train_data = final_examples
        test_data = []

    os.makedirs("data", exist_ok=True)
    
    with open("data/pairwise_train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("data/pairwise_test.jsonl", "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")

    print("Data preparation complete.")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == "__main__":
    construct_comparison_data()


