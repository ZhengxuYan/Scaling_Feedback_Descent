from dotenv import load_dotenv
import asyncio
import json
import os
import re
import tinker
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerMessageCompleter

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CRITIC_MODEL_PATH = "tinker://ffc5255d-7991-5c6d-a09b-b2c8d2e0f879:train:0/sampler_weights/final" 

async def generate_summary(completer, text):
    messages = [
        renderers.Message(role="user", content=f"Summarize the following text:\n\n{text}"),
    ]
    output = await completer(messages)
    return output["content"]

async def evaluate_pair(completer, source_text, summary_a, summary_b):
    user_content = (
        f"Here is a source text:\n{source_text}\n\n"
        f"Summary A:\n{summary_a}\n\n"
        f"Summary B:\n{summary_b}\n\n"
        "Which summary is better and why?"
    )
    
    messages = [
        renderers.Message(role="user", content=user_content),
    ]
    output = await completer(messages)
    return output["content"]

async def improve_summary(completer, source_text, summary, feedback):
    messages = [
        renderers.Message(role="user", content=f"Here is a source text:\n{source_text}\n\nHere is a draft summary:\n{summary}\n\nHere is feedback on the summary:\n{feedback}\n\nPlease rewrite the summary to address the feedback."),
    ]
    output = await completer(messages)
    return output["content"]

def parse_verdict(evaluation_text):
    if "Summary A" in evaluation_text and "Summary B" not in evaluation_text:
        return "Summary A"
    if "Summary B" in evaluation_text and "Summary A" not in evaluation_text:
        return "Summary B"
    match = re.search(r"Verdict\**:\s*(Summary [AB])", evaluation_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "Unknown"

async def run_test_1_preference_accuracy(critic_completer, test_data):
    print("\n--- Running Test 1: Preference Prediction Accuracy ---")
    correct = 0
    total = 0
    results = []

    for i, example in enumerate(test_data):
        # Parse input
        user_content = example['messages'][0]['content']
        try:
            parts = user_content.split("Here is a source text:\n")[1].split("\n\nSummary A:\n")
            source_text = parts[0]
            parts2 = parts[1].split("\n\nSummary B:\n")
            summary_a = parts2[0]
            summary_b = parts2[1].split("\n\nWhich summary is better and why?")[0]
        except IndexError:
            continue

        # Ground Truth
        assistant_content = example['messages'][1]['content']
        ground_truth_verdict = parse_verdict(assistant_content)

        # Model Prediction
        evaluation = await evaluate_pair(critic_completer, source_text, summary_a, summary_b)
        predicted_verdict = parse_verdict(evaluation)

        is_correct = (predicted_verdict == ground_truth_verdict)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "id": i,
            "predicted": predicted_verdict,
            "ground_truth": ground_truth_verdict,
            "evaluation": evaluation,
            "correct": is_correct
        })

    accuracy = correct / total if total > 0 else 0
    print(f"Test 1 Accuracy: {accuracy:.2%} ({correct}/{total})")
    return results

async def run_test_2_feedback_descent(base_completer, critic_completer, test_data):
    print("\n--- Running Test 2: Iterative Refinement Comparison ---")
    results = []

    for i, example in enumerate(test_data):
        print(f"Processing refinement example {i+1}/{len(test_data)}...")
        
        # Extract Source
        user_content = example['messages'][0]['content']
        try:
            source_text = user_content.split("Here is a source text:\n")[1].split("\n\nSummary A:\n")[0]
        except IndexError:
            continue

        # 1. Initial Generation (Generate 2 drafts)
        draft1 = await generate_summary(base_completer, source_text)
        draft2 = await generate_summary(base_completer, source_text)
        
        # --- Baseline 1: Base Model as Critic ---
        eval_base = await evaluate_pair(base_completer, source_text, draft1, draft2)
        verdict_base = parse_verdict(eval_base)
        winner_base = draft1 if verdict_base == "Summary A" else draft2
        
        # Refine
        refined_base = await improve_summary(base_completer, source_text, winner_base, eval_base)
        
        # Final Selection (Compare Winner vs Refined)
        final_eval_base = await evaluate_pair(base_completer, source_text, winner_base, refined_base)
        final_verdict_base = parse_verdict(final_eval_base)
        final_artifact_base = winner_base if final_verdict_base == "Summary A" else refined_base

        # --- Baseline 2: Trained Critic ---
        eval_trained = await evaluate_pair(critic_completer, source_text, draft1, draft2)
        verdict_trained = parse_verdict(eval_trained)
        winner_trained = draft1 if verdict_trained == "Summary A" else draft2
        
        # Refine
        refined_trained = await improve_summary(base_completer, source_text, winner_trained, eval_trained)
        
        # Final Selection (Compare Winner vs Refined)
        final_eval_trained = await evaluate_pair(critic_completer, source_text, winner_trained, refined_trained)
        final_verdict_trained = parse_verdict(final_eval_trained)
        final_artifact_trained = winner_trained if final_verdict_trained == "Summary A" else refined_trained

        results.append({
            "id": i,
            "source": source_text, # Full context for judge
            "baseline_1_artifact": final_artifact_base,
            "baseline_2_artifact": final_artifact_trained,
            "critique_base": eval_base,
            "critique_trained": eval_trained
        })

    return results

def judge_pair_with_gpt4(source, summary_a, summary_b):
    from openai import OpenAI
    client = OpenAI()
    
    prompt = (
        f"Here is a source text:\n{source}\n\n"
        f"Summary A:\n{summary_a}\n\n"
        f"Summary B:\n{summary_b}\n\n"
        "Which summary is better? Please answer with 'Summary A' or 'Summary B' and provide a brief explanation."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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

async def main():
    load_dotenv()
    try:
        with open("data/pairwise_test.jsonl", "r") as f:
            test_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Error: data/pairwise_test.jsonl not found.")
        return
    
    print("Initializing Models...")
    
    try:
        service_client = tinker.ServiceClient(api_key=os.environ.get("TINKER_API_KEY"))
        
        # Get Tokenizer (shared for base and critic as they use the same base model)
        tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL_NAME)

        # Base Model Setup
        base_client = service_client.create_sampling_client(base_model=BASE_MODEL_NAME)
        base_renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
        base_renderer = renderers.get_renderer(base_renderer_name, tokenizer=tokenizer)
        
        base_completer = TinkerMessageCompleter(
            sampling_client=base_client,
            renderer=base_renderer,
            max_tokens=1024
        )

        # Critic Model Setup
        if "YOUR_MODEL_UUID" in CRITIC_MODEL_PATH:
            print("WARNING: CRITIC_MODEL_PATH is not set. Skipping execution.")
            return

        critic_client = service_client.create_sampling_client(base_model=BASE_MODEL_NAME, model_path=CRITIC_MODEL_PATH)
        # Assuming same renderer for critic as it's the same base model
        critic_completer = TinkerMessageCompleter(
            sampling_client=critic_client,
            renderer=base_renderer,
            max_tokens=1024
        )
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Run Test 1
    results_1 = await run_test_1_preference_accuracy(critic_completer, test_data)
    
    # Run Test 2
    results_2 = await run_test_2_feedback_descent(base_completer, critic_completer, test_data)

    # Run Judge on Test 2 Results
    print("\n--- Running LLM Judge on Refinement Results ---")
    wins_baseline_1 = 0
    wins_baseline_2 = 0
    ties = 0
    
    for res in results_2:
        # Baseline 1 is "Summary A" in judge prompt, Baseline 2 is "Summary B"
        verdict = judge_pair_with_gpt4(res['source'], res['baseline_1_artifact'], res['baseline_2_artifact'])
        res['judge_verdict'] = verdict
        
        parsed = parse_verdict(verdict)
        if parsed == "Summary A":
            wins_baseline_1 += 1
        elif parsed == "Summary B":
            wins_baseline_2 += 1
        else:
            ties += 1
            
    print(f"Judge Results: Baseline 1 (Base Critic) Wins: {wins_baseline_1}, Baseline 2 (Trained Critic) Wins: {wins_baseline_2}, Ties/Unknown: {ties}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/test_1_accuracy.json", "w") as f:
        json.dump(results_1, f, indent=2)
    with open("results/test_2_refinement.json", "w") as f:
        json.dump(results_2, f, indent=2)
    
    print("\nEvaluation Complete. Results saved to 'results/'.")

if __name__ == "__main__":
    asyncio.run(main())
