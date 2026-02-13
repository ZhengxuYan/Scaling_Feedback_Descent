import os
import sys
import datetime
import uuid
import re
import json
import random
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_api import APIClient
from custom_api import prompts as prompts_module # Avoiding conflict if prompts is used as variable

# --- Configuration ---
# List of models to benchmark
MODELS_TO_TEST = [
    "tinker/Qwen/Qwen3-4B-Instruct-2507",
    # Add other models here, e.g.:
    # "openai/gpt-4o",
    # "meta-llama/Llama-3.1-70B-Instruct",
]

FINAL_EVAL_MODEL = "anthropic/claude-sonnet-4-5-20250929"
DATASET_NAME = "facebook/research-plan-gen"
CONFIG_NAME = "arxiv"
SPLIT = "train"
NUM_SAMPLES = 5  # Number of samples to test per model

# --- Helper Functions (Copied/Adapted from run_experiment.py) ---

def parse_solution(text):
    match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def parse_evaluation_xml(text):
    """
    Parses the XML output from the evaluation model.
    Returns a list of dictionaries, each containing 'item_num', 'criteria', 'reasoning', and 'errors'.
    """
    items = []
    # Regex to capture each item block
    item_pattern = re.compile(r'<item num=(\d+)>(.*?)</item>', re.DOTALL)
    
    # Regex to capture content within tags
    criteria_pattern = re.compile(r'<criteria>(.*?)</criteria>', re.DOTALL)
    reasoning_pattern = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
    errors_pattern = re.compile(r'<errors>(.*?)</errors>', re.DOTALL)

    for match in item_pattern.finditer(text):
        item_num = match.group(1)
        content = match.group(2)
        
        criteria = ""
        criteria_match = criteria_pattern.search(content)
        if criteria_match:
            criteria = criteria_match.group(1).strip()
            
        reasoning = ""
        reasoning_match = reasoning_pattern.search(content)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            
        errors = ""
        errors_match = errors_pattern.search(content)
        if errors_match:
            errors = errors_match.group(1).strip()
            
        items.append({
            'item_num': item_num,
            'criteria': criteria,
            'reasoning': reasoning,
            'errors': errors
        })
    return items

def main():
    print("--- Initializing Benchmark Script ---")
    client = APIClient()

    # Generate Run ID
    run_id = f"benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    print(f"Run ID: {run_id}")

    # Load Dataset
    print(f"Loading dataset {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT, streaming=True)
    
    # Select fixed samples for consistency
    # We buffer a bit and select the same indices for all models in this run
    print(f"Selecting {NUM_SAMPLES} samples...")
    buffer = []
    MAX_BUFFER = 100
    for i, row in enumerate(ds):
        if i >= MAX_BUFFER:
            break
        buffer.append(row)
    
    # Use a fixed seed for selection to ensure reproducibility across runs if needed, 
    # but here we just want the same samples for all models in *this* run.
    random.seed(42) 
    selected_indices = sorted(random.sample(range(len(buffer)), NUM_SAMPLES))
    samples = [buffer[i] for i in selected_indices]
    
    # Prepare Evaluation Template (from run_experiment.py)
    eval_prompt_template = """Evaluate if the Proposed Research Plan satisfies the Research Scenario based on the provided evaluation criteria.

# Research Scenario
{scenario}

You have to evaluate each of the rubric items provided below.
# Rubric
{rubric_items}

# Reference Solution
Here is a reference solution written by an expert:
{reference_solution}
* It is just meant to demonstrate one possible approach that satisfies the scenario. It is not necessary for the proposed research plan you are grading to match all details in the reference solution.
* The Research Plan you have to grade might have different design choices. This is okay, if the choices are valid, and supported with correct rationale.

# Proposed Research Plan
{proposed_plan}

# Instructions
First, come up with weaknesses of the proposed plan specific to the scenario.
Then, return the following nested XML block for each of the grading items (always close opened XML tags):

<rubric>
<item num=1>
<criteria> Repeat the rubric item string you are checking here. </criteria>
<reasoning> Analyze if the proposed plan violates any of the below desiderata with respect to the rubric item.

**DESIDERATA**
1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item? An exception is if the criteria says "such as", "for example", or "including", the response does not have to include the same examples listed to meet the criteria, but whatever is provided must be valid and reasonable.
2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it? There should be no self-proclaimed claims of handling something without doing so. There should be no vague terms, ambiguity, or lack of clarity. It should be described in simple to understand language.
3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses in the part of the plan addressing this rubric item that invalidate its satisfaction of the rubric item?
4. WELL JUSTIFIED RATIONALE: Is the part of the plan relevant to this grading item well-motivated and justified? For example, are there convincing arguments provided for how the plan handles this grading item is better than simpler solutions or alternate hypotheses?
5. COST AND EFFORT EFFICIENT: Does the plan handle this rubric item cost efficiently, without being unnecessarily complex? Check if a solution requiring less human effort or resources would be equally effective for this rubric item.
6. NO ETHICAL ISSUES: Does this part of the plan have any potential for negative consequences, or is it ethically problematic?
7. CONSISTENT WITH OVERALL PLAN: Is this part of the plan consistent with the rest of the plan? Check if it contradicts any other parts of the plan.

- Be skeptical, careful, and come up with valid criticisms. Be as strict as possible, while being unbiased and reasonable.
- Note that the plan should not just say it satisfies these desiderata, don't be fooled by that. Check carefully WHETHER, HOW and WHY the proposed plan meets each desiderata for this rubric item one by one.
- Based on the above analysis, list the desiderata numbers that are violated. If no part of the plan addresses this rubric item, list all desiderata numbers (1,2,3,4,5,6,7).
</reasoning>
<errors>[comma-separated list of desiderata numbers that are violated or "none" if no violations]</errors>
</item>

... Similarly, for all rubric items...

<item num=n>
...
</item>
</rubric>
"""
    
    # Get generation template using the imported module
    gen_template = prompts_module.get_prompt("generation_template")
    if not gen_template:
        print("ERROR: 'generation_template' not found in prompts.yaml")
        return

    all_results = []
    
    print(f"Models to test: {MODELS_TO_TEST}")

    # To store incrementally, we'll maintain the file path
    output_file = f"benchmark_results_{run_id}.json"

    for model_name in MODELS_TO_TEST:
        print(f"\n>>> Benchmarking Model: {model_name} <<<")
        
        for i, row in enumerate(samples):
            print(f"\n   Processing Sample {i+1}/{NUM_SAMPLES} (ID: {row.get('q_id', 'N/A')})")
            
            scenario = row.get('Goal', '')
            rubric_list = row.get('Rubric', [])
            reference_solution_str = row.get('Reference solution', '')
            
            # Format Rubric String for Generation
            rubric_items_str = ""
            for idx, r_item in enumerate(rubric_list):
                rubric_items_str += f"Item {idx+1}: {r_item}\n"
            
            # Prepare Generation Prompt
            formatted_prompt = gen_template.format(
                scenario=scenario,
                rubric_section=rubric_items_str
            )
            
            # 1. Generate Plan
            print("      Generating plan...")
            result_raw = ""
            result_parsed = ""
            try:
                result_raw = client.generate(
                    model=model_name,
                    prompt=formatted_prompt,
                    temperature=0.7,
                    max_tokens=2048
                )
                
                # Parse solution if it's wrapped in tags
                result_parsed = parse_solution(result_raw)
            except Exception as e:
                print(f"      ERROR generating plan: {e}")
                result_raw = f"ERROR: {e}"
                result_parsed = f"ERROR: {e}"

            # 2. Evaluate Plan
            print("      Evaluating plan...")
            eval_prompt = eval_prompt_template.format(
                scenario=scenario,
                rubric_items=rubric_items_str,
                reference_solution=reference_solution_str,
                proposed_plan=result_parsed
            )
            
            score = 0.0
            evaluation_raw = ""
            try:
                evaluation_raw = client.generate(
                    model=FINAL_EVAL_MODEL,
                    prompt=eval_prompt,
                    temperature=0.0,
                    max_tokens=8192
                )
                
                parsed_items = parse_evaluation_xml(evaluation_raw)
                
                total_items = 0
                passed_items = 0
                
                for item in parsed_items:
                    total_items += 1
                    errors = item['errors']
                    # Check if errors is empty, "None", "none", or just whitespace
                    if not errors or errors.lower() == "none" or not errors.strip():
                        passed_items += 1
                
                if total_items > 0:
                    score = passed_items / total_items
                print(f"      Score: {passed_items}/{total_items} ({score:.2%})")
                
            except Exception as e:
                print(f"      ERROR evaluating plan: {e}")
                evaluation_raw = f"ERROR: {e}"
                score = 0.0

            # Store Result
            all_results.append({
                "run_id": run_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "model_name": model_name,
                "sample_id": row.get('q_id'),
                "scenario": scenario,
                "generated_plan_raw": result_raw,
                "generated_plan_parsed": result_parsed,
                "evaluation_raw": evaluation_raw,
                "score": score
            })

            # Save incrementally
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    print(f"\nBenchmark Complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
