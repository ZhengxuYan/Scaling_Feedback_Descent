from dotenv import load_dotenv
import asyncio
import json
import os
import re
import tinker
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerMessageCompleter
from openai import AsyncOpenAI

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
CRITIC_MODEL_PATH = "tinker://926a008d-7ea8-5228-95c9-558764984c26:train:0/sampler_weights/final" 
FRONTIER_MODELS = ["gpt-5.1", "gpt-5-mini", "gpt-4o-mini"]

async def generate_summary(completer, text):
    messages = [
        renderers.Message(role="user", content=f"Summarize the following text:\n\n{text}"),
    ]
    output = await completer(messages)
    return output["content"]

async def generate_frontier_summary(client, text, model="gpt-5.1"):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ]
    }
    # Some models don't support temperature=0
    if model != "gpt-5-mini":
        kwargs["temperature"] = 0

    try:
        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Frontier generation error ({model}): {e}")
        return "Error"

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
    cleaned = evaluation_text.strip()
    if cleaned.lower().startswith("summary a"):
        return "Summary A"
    if cleaned.lower().startswith("summary b"):
        return "Summary B"
        
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
        print(f"Processing Test 1 example {i+1}/{len(test_data)}...")
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
    existing_results_map = {}
    
    # Caching: Load existing results
    # Caching: Load existing results (from 4B run to reuse frontier artifacts)
    results_path = "results/test_2_refinement_4b.json"
    if os.path.exists(results_path):
        print(f"Loading existing results from {results_path}...")
        try:
            with open(results_path, "r") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    # Only keep frontier artifacts to avoid re-generation, but force re-run of baselines
                    filtered_item = {'id': item['id']}
                    for k, v in item.items():
                        if k.startswith('frontier_artifact_'):
                            filtered_item[k] = v
                    existing_results_map[item['id']] = filtered_item
        except json.JSONDecodeError:
            print("Error reading existing results, starting fresh.")

    # Initialize Frontier Client
    frontier_client = AsyncOpenAI()

    for i, example in enumerate(test_data):
        print(f"Processing refinement example {i+1}/{len(test_data)}...")
        
        # Extract Source
        user_content = example['messages'][0]['content']
        try:
            source_text = user_content.split("Here is a source text:\n")[1].split("\n\nSummary A:\n")[0]
        except IndexError:
            continue

        # Check cache
        cached_item = existing_results_map.get(i)
        
        # Initialize variables
        final_artifact_base = None
        final_artifact_trained = None
        final_artifact_3_step = None
        eval_base = None
        eval_trained = None
        
        # Load from cache if available
        if cached_item:
            final_artifact_base = cached_item.get('baseline_1_artifact')
            final_artifact_trained = cached_item.get('baseline_2_artifact')
            final_artifact_3_step = cached_item.get('baseline_3_artifact')
            eval_base = cached_item.get('critique_base')
            eval_trained = cached_item.get('critique_trained')

        # Determine what needs to be run
        need_run_base = (final_artifact_base is None)
        need_run_trained = (final_artifact_trained is None)
        need_run_3_step = (final_artifact_3_step is None)

        if need_run_base or need_run_trained or need_run_3_step:
            # 1. Initial Generation (Generate 2 drafts)
            # We must generate drafts if we are running ANY baseline logic that requires them.
            # Even if we have B1 cached, we need drafts for B2/B3 if they are missing.
            draft1 = await generate_summary(base_completer, source_text)
            draft2 = await generate_summary(base_completer, source_text)
            
            # --- Baseline 1: Base Model as Critic ---
            if need_run_base:
                eval_base = await evaluate_pair(base_completer, source_text, draft1, draft2)
                verdict_base = parse_verdict(eval_base)
                winner_base = draft1 if verdict_base == "Summary A" else draft2
                
                # Refine
                refined_base = await improve_summary(base_completer, source_text, winner_base, eval_base)
                
                # Final Selection (Compare Winner vs Refined)
                final_eval_base = await evaluate_pair(base_completer, source_text, winner_base, refined_base)
                final_verdict_base = parse_verdict(final_eval_base)
                final_artifact_base = winner_base if final_verdict_base == "Summary A" else refined_base

            # --- Baseline 2 & 3: Trained Critic ---
            if need_run_trained or need_run_3_step:
                # Step 1 (Baseline 2 Logic)
                eval_trained = await evaluate_pair(critic_completer, source_text, draft1, draft2)
                verdict_trained = parse_verdict(eval_trained)
                winner_trained = draft1 if verdict_trained == "Summary A" else draft2
                
                # Refine
                refined_trained = await improve_summary(base_completer, source_text, winner_trained, eval_trained)
                
                # Final Selection (Compare Winner vs Refined)
                final_eval_trained_new = await evaluate_pair(critic_completer, source_text, winner_trained, refined_trained)
                final_verdict_trained = parse_verdict(final_eval_trained_new)
                final_artifact_trained_new = winner_trained if final_verdict_trained == "Summary A" else refined_trained
                
                # If we needed to run trained (B2), save it. 
                # If we already had it cached, we keep the cached version (final_artifact_trained) 
                # but we use the NEW intermediate state for B3 to ensure continuity in THIS run.
                if need_run_trained:
                    final_artifact_trained = final_artifact_trained_new

                # --- Baseline 3: 3-Step Feedback Descent ---
                if need_run_3_step:
                    # Start with the winner of the initial round and the initial feedback
                    current_best = winner_trained
                    current_feedback = eval_trained
                    
                    # We already did 1 step above (producing refined_trained and final_artifact_trained_new)
                    # But the loop logic below does 3 steps. 
                    # If we want "3 steps total", we should do 2 more steps starting from the result of step 1?
                    # OR, we can just run the loop 3 times from the start.
                    # The previous implementation ran the loop 3 times from the start (winner_trained).
                    # Let's stick to that for simplicity and robustness.
                    
                    current_best = winner_trained
                    current_feedback = eval_trained
                    
                    for _ in range(3):
                        # Refine
                        refined_candidate = await improve_summary(base_completer, source_text, current_best, current_feedback)
                        
                        # Compare (Current Best vs Refined Candidate)
                        comparison_eval = await evaluate_pair(critic_completer, source_text, current_best, refined_candidate)
                        comparison_verdict = parse_verdict(comparison_eval)
                        
                        # Update if Refined is better
                        if comparison_verdict == "Summary B":
                            current_best = refined_candidate
                        
                        # Always update feedback for the next step
                        current_feedback = comparison_eval
                    
                    final_artifact_3_step = current_best

        # --- Frontier Models (Single Turn) ---
        frontier_artifacts = {}
        for model in FRONTIER_MODELS:
            key = f"frontier_artifact_{model}"
            
            if cached_item and key in cached_item:
                 print(f"  Using cached frontier artifact for {model} ID {i}")
                 frontier_artifacts[key] = cached_item[key]
            else:
                 print(f"  Generating frontier artifact for {model} ID {i}...")
                 frontier_artifacts[key] = await generate_frontier_summary(frontier_client, source_text, model=model)

        result_entry = {
            "id": i,
            "source": source_text, # Full context for judge
            "baseline_1_artifact": final_artifact_base,
            "baseline_2_artifact": final_artifact_trained,
            "baseline_3_artifact": final_artifact_3_step,
            "critique_base": eval_base,
            "critique_trained": eval_trained
        }
        # Add all frontier artifacts
        result_entry.update(frontier_artifacts)
        
        # Preserve other keys from cached_item (like judge verdicts) if they exist
        if cached_item:
            for k, v in cached_item.items():
                if k not in result_entry:
                    result_entry[k] = v
            
        results.append(result_entry)

    return results

def judge_pair(source, summary_a, summary_b):
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
    print("\n--- Running Test 1: Preference Prediction Accuracy ---")
    if os.path.exists("results/test_1_accuracy_235b.json"):
        print("Using cached results for Test 1")
        with open("results/test_1_accuracy_235b.json", "r") as f:
            results_1 = json.load(f)
        
        # Recalculate accuracy from cached results
        correct = sum(1 for r in results_1 if r['predicted'] == r['ground_truth']) # Changed from 'predicted_preference' to 'predicted' based on run_test_1_preference_accuracy output
        total = len(results_1)
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Test 1 Accuracy: {accuracy:.2f}% ({correct}/{total})")
    else:
        results_1 = await run_test_1_preference_accuracy(critic_completer, test_data)
        # Save Test 1 results immediately
        os.makedirs("results", exist_ok=True)
        with open("results/test_1_accuracy_235b.json", "w") as f:
            json.dump(results_1, f, indent=2)
    
    # Run Test 2
    results_2 = await run_test_2_feedback_descent(base_completer, critic_completer, test_data)

    # Run Judge on Test 2 Results
    print("\n--- Running LLM Judge on Refinement Results ---")
    wins_baseline_1 = 0
    wins_baseline_2 = 0
    ties = 0
    
    wins_1_step = 0
    wins_3_step = 0
    ties_steps = 0
    
    # Initialize counters for all frontier models
    frontier_stats = {model: {"wins": 0, "losses": 0, "ties": 0} for model in FRONTIER_MODELS}
    frontier_stats_3_step = {model: {"wins": 0, "losses": 0, "ties": 0} for model in FRONTIER_MODELS}

    for res in results_2:
        # Comparison 1: Base Critic vs Trained Critic
        # Base is "Summary A", Trained is "Summary B"
        if 'judge_verdict' in res:
            verdict_base = res['judge_verdict']
        else:
            verdict_base = judge_pair(res['source'], res['baseline_1_artifact'], res['baseline_2_artifact'])
            res['judge_verdict'] = verdict_base
        
        parsed_base = parse_verdict(verdict_base)
        if parsed_base == "Summary A":
            wins_baseline_1 += 1
        elif parsed_base == "Summary B":
            wins_baseline_2 += 1
        else:
            ties += 1

        # Comparison 2: 1-Step vs 3-Step Feedback Descent
        # 1-Step is "Summary A", 3-Step is "Summary B"
        if 'judge_verdict_steps' in res:
            verdict_steps = res['judge_verdict_steps']
        else:
            verdict_steps = judge_pair(res['source'], res['baseline_2_artifact'], res['baseline_3_artifact'])
            res['judge_verdict_steps'] = verdict_steps
            
        parsed_steps = parse_verdict(verdict_steps)
        if parsed_steps == "Summary A":
            wins_1_step += 1
        elif parsed_steps == "Summary B":
            wins_3_step += 1
        else:
            ties_steps += 1

        # Comparison 2+: Frontier Models vs Trained Critic
        for model in FRONTIER_MODELS:
            key = f"frontier_artifact_{model}"
            if key not in res:
                print(f"Warning: Missing artifact for {model}")
                continue
            else:
                artifact = res[key]

            # Frontier is "Summary A", Trained Critic is "Summary B"
            verdict_key = f'judge_verdict_{model}'
            if verdict_key in res:
                verdict_frontier = res[verdict_key]
            else:
                verdict_frontier = judge_pair(res['source'], artifact, res['baseline_2_artifact'])
                res[verdict_key] = verdict_frontier
            
            parsed_frontier = parse_verdict(verdict_frontier)
            if parsed_frontier == "Summary A":
                frontier_stats[model]["wins"] += 1
            elif parsed_frontier == "Summary B":
                frontier_stats[model]["losses"] += 1
            else:
                frontier_stats[model]["ties"] += 1

            # Comparison 3: Frontier Models vs 3-Step Feedback Descent
            # Frontier is "Summary A", 3-Step is "Summary B"
            verdict_key_3 = f'judge_verdict_{model}_vs_3step'
            if verdict_key_3 in res:
                verdict_frontier_3 = res[verdict_key_3]
            else:
                verdict_frontier_3 = judge_pair(res['source'], artifact, res['baseline_3_artifact'])
                res[verdict_key_3] = verdict_frontier_3
            
            parsed_frontier_3 = parse_verdict(verdict_frontier_3)
            if parsed_frontier_3 == "Summary A":
                frontier_stats_3_step[model]["wins"] += 1
            elif parsed_frontier_3 == "Summary B":
                frontier_stats_3_step[model]["losses"] += 1
            else:
                frontier_stats_3_step[model]["ties"] += 1
            
    print(f"Judge Results (Base vs Trained 1-Step): Base Critic Wins: {wins_baseline_1}, Trained Critic Wins: {wins_baseline_2}, Ties/Unknown: {ties}")
    print(f"Judge Results (1-Step vs 3-Step): 1-Step Wins: {wins_1_step}, 3-Step Wins: {wins_3_step}, Ties/Unknown: {ties_steps}")
    
    for model in FRONTIER_MODELS:
        stats = frontier_stats[model]
        print(f"Judge Results ({model} vs Trained 1-Step): {model} Wins: {stats['wins']}, Trained Critic Wins: {stats['losses']}, Ties/Unknown: {stats['ties']}")

    for model in FRONTIER_MODELS:
        stats = frontier_stats_3_step[model]
        print(f"Judge Results ({model} vs Trained 3-Step): {model} Wins: {stats['wins']}, Trained Critic Wins: {stats['losses']}, Ties/Unknown: {stats['ties']}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/test_1_accuracy_235b.json", "w") as f:
        json.dump(results_1, f, indent=2)
    with open("results/test_2_refinement_235b.json", "w") as f:
        json.dump(results_2, f, indent=2)
    
    print("\nEvaluation Complete. Results saved to 'results/'.")

if __name__ == "__main__":
    asyncio.run(main())
