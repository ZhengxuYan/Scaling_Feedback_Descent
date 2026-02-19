import os
import sys
import datetime
import uuid
import re
import json
import random
import asyncio
import argparse
from dotenv import load_dotenv
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_api import APIClient, prompts

# --- Configuration ---
# Models
GENERATOR_MODEL = "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507"
FEEDBACK_MODEL = "tinker/Qwen/Qwen3-30B-A3B-Instruct-2507" 
JUDGE_MODEL = "openai/gpt-5.2"

def parse_solution(text):
    match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def parse_evaluation_xml(text):
    """
    Parses the XML output from the evaluation model.
    Returns a score between 0.0 and 1.0
    """
    # Regex to capture each item block
    item_pattern = re.compile(r'<item num=(\d+)>(.*?)</item>', re.DOTALL)
    errors_pattern = re.compile(r'<errors>(.*?)</errors>', re.DOTALL)

    total_items = 0
    passed_items = 0

    for match in item_pattern.finditer(text):
        total_items += 1
        content = match.group(2)
        
        errors = ""
        errors_match = errors_pattern.search(content)
        if errors_match:
            errors = errors_match.group(1).strip()
            
        # Check if errors is empty, "None", "none", or just whitespace
        if not errors or errors.lower() == "none" or not errors.strip():
            passed_items += 1
            
            
    if total_items == 0:
        print(f"DEBUG: No items found in XML response!")
        return 0.0
        
    print(f"DEBUG: Parsed {passed_items}/{total_items} items passed. Score: {passed_items / total_items:.2f}")
    return passed_items / total_items

# --- Templates (Copied from run_experiment.py) ---
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

# Rubric
{rubric_items}

{reference_section}

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

class DPOCollector:
    def __init__(self):
        self.client = APIClient()
    
    def generate_draft(self, prompt):
        # Initial draft generation
        return self.client.generate(
            model=GENERATOR_MODEL,
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
            parsing_function=parse_solution
        )

    def generate_feedback(self, prompt, challenger, incumbent, context):
        # Generate feedback (Critique)
        # Using high temperature to get diverse feedbacks
        
        # Prepare feedback prompt
        feedback_prompt = refinement_feedback_template.format(
            scenario=prompt,
            plan_a=challenger,
            plan_b=incumbent,
            **context
        )
        
        # messages = [{"role": "user", "content": feedback_prompt}]
        
        return self.client.generate(
            model=FEEDBACK_MODEL,
            prompt=feedback_prompt, # APIClient handles messages if this is string
            temperature=1.0, # High temp for diversity
            max_tokens=1024
        ), feedback_prompt

    def generate_revision(self, prompt, challenger, incumbent, feedback, context, history=""):
        # Parse verdict from feedback
        cleaned_feedback = feedback.strip().lower()
        winning_label = "Draft B"
        if "draft a" in cleaned_feedback or "response a" in cleaned_feedback:
            winning_label = "Draft A"
        
        # Prepare revision prompt
        revision_prompt = refinement_gen_template.format(
            scenario=prompt,
            plan_a=challenger,
            plan_b=incumbent,
            verdict=winning_label,
            rationale=feedback,
            history=history,
            **context
        )
        
        revision = self.client.generate(
            model=GENERATOR_MODEL,
            prompt=revision_prompt,
            temperature=0.7,
            max_tokens=1024,
            parsing_function=parse_solution
        )
        return revision

    def evaluate_plan(self, prompt, plan, context):
        eval_prompt = eval_prompt_template.format(
            scenario=prompt,
            proposed_plan=plan,
            **context
        )
        
        evaluation = self.client.generate(
            model=JUDGE_MODEL,
            prompt=eval_prompt,
            temperature=0.0,
            max_tokens=8192
        )
        
        return parse_evaluation_xml(evaluation)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--feedback_rounds", type=int, default=1)
    parser.add_argument("--output_file", type=str, default="dpo_feedback_training_data.jsonl")
    args = parser.parse_args()
    
    DATASET_CONFIGS = ["arxiv", "ml", "pubmed"]

    print("--- Initializing DPO Data Collector ---")
    load_dotenv()
    collector = DPOCollector()
    
    # Calculate samples per config to reach total num_samples roughly
    samples_per_config = (args.num_samples + len(DATASET_CONFIGS) - 1) // len(DATASET_CONFIGS)
    
    gen_template = prompts.get_prompt("generation_template") or "{scenario}"
    
    total_pairs_collected = 0
    
    # Ensure output file is empty if it's a new run (or should we append? User said "add a pair... not output all at the end")
    # Usually safer to append if it exists, or user can clear it. 
    # But for a specific "run", maybe we want to start fresh or continue?
    # Let's assuming appending is fine, but maybe print a warning or separation.
    print(f"Output will be appended to {args.output_file}")

    for config_name in DATASET_CONFIGS:
        print(f"\n--- Processing Config: {config_name} ---")
        try:
            ds = load_dataset("facebook/research-plan-gen", config_name, split="train", streaming=True)
            
            buffer = []
            MAX_BUFFER = 200 # increased buffer for better randomization
            for i, row in enumerate(ds):
                if i >= MAX_BUFFER:
                    break
                buffer.append(row)
            
            current_samples_needed = min(samples_per_config, len(buffer))
            if current_samples_needed == 0:
                print(f"No samples found for {config_name}")
                continue
                
            selected_indices = sorted(random.sample(range(len(buffer)), current_samples_needed))
            samples = [buffer[i] for i in selected_indices]
            
            for i, row in enumerate(samples):
                print(f"\nProcessing Sample {i+1}/{len(samples)} from {config_name}")
        
                scenario = row.get('Goal', '')
                rubric_list = row.get('Rubric', [])
                reference_solution_str = row.get('Reference solution', '')
                
                rubric_items_str = ""
                for idx, r_item in enumerate(rubric_list):
                    rubric_items_str += f"Item {idx+1}: {r_item}\n"
                    
                context = {
                    "rubric_items": rubric_items_str,
                    "rubric_section": rubric_items_str,
                    "reference_section": reference_solution_str,
                    "reference_solution": reference_solution_str
                }
                
                # 0. Initial Drafts (Draft A and Draft B)
                formatted_prompt = gen_template.format(scenario=scenario, rubric_section=rubric_items_str)
                
                draft_a = collector.generate_draft(formatted_prompt)
                draft_b = collector.generate_draft(formatted_prompt)
                
                current_challenger = draft_a
                current_incumbent = draft_b
                history_buffer = []    
                
                # Branching Loop 
                for round_idx in range(args.feedback_rounds):
                    print(f"--- Round {round_idx+1} ---")
                    
                    # 1. Generate Two Feedbacks
                    print("Generating Feedback 1...")
                    feedback_1, feedback_prompt = collector.generate_feedback(scenario, current_challenger, current_incumbent, context)
                    
                    print("Generating Feedback 2...")
                    feedback_2, _ = collector.generate_feedback(scenario, current_challenger, current_incumbent, context)
                    
                    # 2. Generate Revisions based on Feedbacks
                    history_text = "\n\n".join(history_buffer) if history_buffer else ""
                    if history_text:
                        history_text = f"Previous Rounds History:\n{history_text}"

                    print("Generating Revision 1...")
                    revision_1 = collector.generate_revision(scenario, current_challenger, current_incumbent, feedback_1, context, history=history_text)
                    
                    print("Generating Revision 2...")
                    revision_2 = collector.generate_revision(scenario, current_challenger, current_incumbent, feedback_2, context, history=history_text)
                    
                    # 3. Evaluate Revisions
                    print("Evaluating Revision 1...")
                    score_1 = collector.evaluate_plan(scenario, revision_1, context)
                    
                    print("Evaluating Revision 2...")
                    score_2 = collector.evaluate_plan(scenario, revision_2, context)
                    
                    print(f"Score 1: {score_1:.2f} | Score 2: {score_2:.2f}")
                    
                    # 4. Construct DPO Pair
                    if score_1 != score_2:
                        if score_1 > score_2:
                            chosen_feedback = feedback_1
                            rejected_feedback = feedback_2
                            winner_revision = revision_1
                            winner_feedback = feedback_1
                        else:
                            chosen_feedback = feedback_2
                            rejected_feedback = feedback_1
                            winner_revision = revision_2
                            winner_feedback = feedback_2

                        
                        # Format: interleaved chosen/rejected
                        # Chosen
                        chosen_entry = {
                            "messages": [
                                {"role": "user", "content": feedback_prompt},
                                {"role": "assistant", "content": chosen_feedback}
                            ],
                            "score": max(score_1, score_2),
                            "source": config_name
                        }
                        # Rejected
                        rejected_entry = {
                            "messages": [
                                {"role": "user", "content": feedback_prompt},
                                {"role": "assistant", "content": rejected_feedback}
                            ],
                            "score": min(score_1, score_2),
                            "source": config_name
                        }
                        
                        print(f"Pair added. Winner: {'Feedback 1' if score_1 > score_2 else 'Feedback 2'}. Scores: {score_1:.2f} vs {score_2:.2f}")

                        # Immediate Save
                        with open(args.output_file, 'a') as f:
                            json.dump(chosen_entry, f)
                            f.write('\n')
                            json.dump(rejected_entry, f)
                            f.write('\n')
                        
                        total_pairs_collected += 1

                        
                        # Update history
                        history_entry = (
                            f"--- Round {round_idx + 1} ---\n"
                            f"Draft A:\n{current_challenger}\n\n"
                            f"Draft B:\n{current_incumbent}\n\n"
                            f"Feedback:\n{winner_feedback}\n"
                            f"Selected Revision Score: {max(score_1, score_2):.2f}\n"
                        )
                        history_buffer.append(history_entry)

                        current_incumbent = winner_revision
                        current_challenger = collector.generate_draft(formatted_prompt)
                        
                    else:
                        print(f"Tie. Skipping pair. Scores: {score_1:.2f} vs {score_2:.2f}")
                        current_incumbent = revision_1
                        current_challenger = collector.generate_draft(formatted_prompt)

        except Exception as e:
            print(f"Error processing config {config_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Collected {total_pairs_collected} pairs (x2 rows) in {args.output_file}")

if __name__ == "__main__":
    main()
