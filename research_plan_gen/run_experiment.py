
# Auto-generated Script: Run Refactored API
import os
import sys
import datetime
import uuid
import re
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_api import APIClient, prompts

# --- Configuration ---
# Models
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507")
GENERATOR_MODEL_PATH = os.getenv("GENERATOR_MODEL_PATH", None)

FEEDBACK_MODEL = os.getenv("FEEDBACK_MODEL", "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507")
FEEDBACK_MODEL_PATH = os.getenv("FEEDBACK_MODEL_PATH", None)

FINAL_EVAL_MODEL = os.getenv("FINAL_EVAL_MODEL", "anthropic/claude-sonnet-4-5-20250929")

# Settings
FEEDBACK_ROUNDS = 2
DATASET_NAME = "facebook/research-plan-gen"
CONFIG_NAME = "arxiv"
SPLIT = "train"
NUM_SAMPLES = 20

from datasets import load_dataset
import random
import json

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
    print("--- Initializing API Client ---")
    client = APIClient()

    print(f"\nGenerative Model: {GENERATOR_MODEL}")
    if GENERATOR_MODEL_PATH:
        print(f"Gen Model Path:   {GENERATOR_MODEL_PATH}")
    print(f"Feedback Model:   {FEEDBACK_MODEL}")
    if FEEDBACK_MODEL_PATH:
        print(f"FB Model Path:    {FEEDBACK_MODEL_PATH}")
    print(f"Feedback Rounds:  {FEEDBACK_ROUNDS}")
    print(f"Dataset:          {DATASET_NAME} ({CONFIG_NAME})")

    # Generate Run ID
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    print(f"Run ID:           {run_id}")

    # Load Dataset
    print(f"Loading dataset {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT, streaming=True)
    
    print(f"Selecting {NUM_SAMPLES} random samples...")
    # Buffer a bit
    buffer = []
    MAX_BUFFER = 100
    for i, row in enumerate(ds):
        if i >= MAX_BUFFER:
            break
        buffer.append(row)
    
    # Fixed random seed for reproducible sample selection across runs
    random.seed(42)
    selected_indices = sorted(random.sample(range(len(buffer)), NUM_SAMPLES))
    samples = [buffer[i] for i in selected_indices]

    # Define custom templates for this specific task
    refinement_gen_template = """
I will provide you a research scenario and two draft research plans (Plan A and Plan B) along with a critique comparing them. You have to provide an improved, concise yet thoughtful research plan based on the winner.

Here is the research scenario.
Scenario: {scenario}

{rubric_section}

{history}

Here is Draft Plan A:
{plan_a}

Here is Draft Plan B:
{plan_b}

Here is the Critique and Verdict:
Verdict: {verdict}
Rationale: {rationale}

INSTRUCTION: Revise the winning plan ({verdict}) to further improve it based on the feedback. Keep the good parts but address any weaknesses mentioned in the rationale. You can also incorporate any strong points from the losing plan if relevant.
The solution inside <solution></solution> tags should be readable for humans, and not in XML itself.
Output Format: Put the final improved solution inside <solution> </solution> XML tags.
"""

    refinement_feedback_template = """
Evaluate which of the two Proposed Research Plans is better for the Research Scenario based on the provided evaluation criteria.

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

# Proposed Research Plan A
{plan_a}

# Proposed Research Plan B
{plan_b}

**DESIDERATA**
1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item?
2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it? There should be no self-proclaimed claims of handling something without doing so.
3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses?
4. WELL JUSTIFIED RATIONALE: Is the plan well-motivated and justified?
5. COST AND EFFORT EFFICIENT: Does the plan handle this cost efficiently?
6. NO ETHICAL ISSUES: Does the plan have any potential for negative consequences?
7. CONSISTENT WITH OVERALL PLAN: Is the plan consistent?

# Instructions
First, compare the two plans against the rubric and desiderata. Identify the strengths and weaknesses of each.
Then, provide a verdict indicating which plan is better, and a clear text rationale explaining why.
IMPORTANT: In your rationale, do NOT quote the exact rubric items. Discuss the concepts they represent, but do not leak the specific wording of the rubric.

Please output your response in the following format:
Rationale: [Your detailed comparison and justification]
Verdict: [Plan A or Plan B]
"""

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

    
    try:
        # Load template from prompts.yaml
        gen_template = prompts.get_prompt("generation_template")
        
        results = []

        for i, row in enumerate(samples):
            print(f"\n==================================================")
            print(f"Processing Sample {i+1}/{NUM_SAMPLES} (ID: {row.get('q_id', 'N/A')})")
            
            gen_str = f"Gen={GENERATOR_MODEL}" + (f" (Path: {GENERATOR_MODEL_PATH})" if GENERATOR_MODEL_PATH else "")
            fb_str = f"Feedback={FEEDBACK_MODEL}" + (f" (Path: {FEEDBACK_MODEL_PATH})" if FEEDBACK_MODEL_PATH else "")
            print(f"Models: {gen_str}, {fb_str}")
            
            print(f"==================================================")
            
            scenario = row.get('Goal', '')
            rubric_list = row.get('Rubric', [])
            reference_solution_str = row.get('Reference solution', '')
            
            # Format Rubric String
            rubric_items_str = ""
            for idx, r_item in enumerate(rubric_list):
                rubric_items_str += f"Item {idx+1}: {r_item}\n"
            
            if not gen_template:
                 print("Warning: generation_template not found. Using raw topic.")
                 formatted_prompt = scenario
            else:
                 # Fill in the template
                 formatted_prompt = gen_template.format(
                     scenario=scenario,
                     rubric_section=rubric_items_str
                 )

            print("\n--- Starting Generation Loop ---")
            
            # 1. Generate & Refine
            result = client.generate(
                model=GENERATOR_MODEL,
                model_path=GENERATOR_MODEL_PATH,
                prompt=formatted_prompt,
                feedback_rounds=FEEDBACK_ROUNDS,
                feedback_model=FEEDBACK_MODEL,
                feedback_model_path=FEEDBACK_MODEL_PATH,
                temperature=0.7,
                max_tokens=1024,
                run_id=run_id,
                parsing_function=parse_solution,
                # Explicitly pass templates to decouple from custom_api/prompts.py
                refinement_prompt_template=refinement_gen_template,
                feedback_prompt_template=refinement_feedback_template,
                # Pass context for refinement templates
                rubric_items=rubric_items_str,
                rubric_section=rubric_items_str,
                reference_section=reference_solution_str,
                reference_solution=reference_solution_str
            )
            
            if FEEDBACK_ROUNDS > 0:
                print("\n--- Refined Result (Snippet) ---")
            else:
                print("\n--- Generated Result (Snippet) ---")
            print(result[:500] + "..." if len(result) > 500 else result)

            # 2. Final Evaluation
            print("\n--- Running Final Evaluation ---")
            
            eval_prompt = eval_prompt_template.format(
                scenario=scenario,
                rubric_items=rubric_items_str,
                reference_solution=reference_solution_str,
                proposed_plan=result
            )
            
            evaluation = client.generate(
                model=FINAL_EVAL_MODEL,
                prompt=eval_prompt,
                temperature=0.0,
                max_tokens=8192
            )
            
            print("\n--- Parsed Evaluation ---")
            parsed_items = parse_evaluation_xml(evaluation)
            
            total_items = 0
            passed_items = 0
            
            for item in parsed_items:
                total_items += 1
                errors = item['errors']
                # Check if errors is empty, "None", "none", or just whitespace
                if not errors or errors.lower() == "none" or not errors.strip():
                    passed_items += 1
                    
                print(f"Item {item['item_num']}: Criteria: {item['criteria'][:50]}... | Errors: {item['errors']}")

            score = 0
            if total_items > 0:
                score = passed_items / total_items
                print(f"\nScore: {passed_items}/{total_items} ({score:.2%})")
            else:
                print("\nScore: N/A (No rubric items parsed)")
                
            results.append({
                "sample_id": row.get('q_id'),
                "scenario": scenario,
                "score": score,
                "result": result,
                "evaluation": evaluation
            })
            
            # Save results to JSON file incrementally
            output_file = f"experiment_results_{run_id}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"\nERROR: An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
