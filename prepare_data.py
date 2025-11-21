import json
import os
from datasets import load_dataset

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

    training_examples = []
    
    threshold = 1000
    print(f"Filtering for contexts > {threshold} words...")

    print("Creating pairs...")
    for doc_id, data in grouped_data.items():
        # Check document length
        if len(data['document'].split()) <= threshold:
            continue

        cands = data['candidates']
        
        if len(cands) < 2:
            continue
            
        cand_a = cands[0]
        cand_b = cands[1]
        
        if cand_a['score'] > cand_b['score']:
            winner = "Summary A"
            rationale = create_synthetic_rationale(cand_a, cand_b)
        elif cand_b['score'] > cand_a['score']:
            winner = "Summary B"
            rationale = create_synthetic_rationale(cand_b, cand_a)
        else:
            continue

        user_content = (
            f"Here is a source text:\n{data['document']}\n\n"
            f"Summary A:\n{cand_a['summary']}\n\n"
            f"Summary B:\n{cand_b['summary']}\n\n"
            "Which summary is better and why?"
        )
        
        assistant_content = f"**Verdict**: {winner}\n\n**Feedback**: {rationale}"
        
        training_examples.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })
        
    print(f"Created {len(training_examples)} comparison pairs.")
    
    # Split into train and test
    test_size = 20
    if len(training_examples) > test_size:
        test_data = training_examples[:test_size]
        train_data = training_examples[test_size:]
    else:
        print("Warning: Not enough data for split. Using all for train.")
        train_data = training_examples
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

if __name__ == "__main__":
    construct_comparison_data()
