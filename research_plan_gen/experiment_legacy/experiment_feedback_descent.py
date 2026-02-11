import os
import sys
import json
import random
import asyncio
import re
from datasets import load_dataset
from dotenv import load_dotenv

# Import local custom API client
from custom_api import APIClient

# --- Configuration ---
DATASET_NAME = "facebook/research-plan-gen"
CONFIG_NAME = "arxiv"
SPLIT = "train"
NUM_SAMPLES = 5
OUTPUT_FILE = "feedback_descent_results.json"

# Models
MODEL_GENERATOR = "Qwen/Qwen3-4B-Instruct-2507" 
MODEL_JUDGE = "gpt-5.2" 
MODEL_FINAL_JUDGE = "claude-sonnet-4-5-20250929" 


# --- Prompts ---

def build_generation_prompt(row):
    scenario = row.get('Goal', '')
    rubric_list = row.get('Rubric', [])
    doc = row.get('doc', '') 

    prompt = f"I will provide you a research scenario. You have to provide me a concise yet thoughtful research plan with all details needed to execute it.\n\n"
    
    prompt += f"Here is the research scenario.\n"
    prompt += f"Scenario: {scenario}\n\n"

    if doc:
        prompt += f"If a research paper is provided:\n"
        prompt += f"Now I will provide you with the research document you have to use to answer the scenario.\n"
        prompt += f"**DOCUMENT**\n{doc}\n\n"
        prompt += f"INSTRUCTION: You have to stick to exactly how the document solves the scenario, and not change any details. Include all motivation, justification, and details of the plan the researchers used to approach the scenario provided above. Do not omit any information.\n\n"

    if rubric_list:
        prompt += f"If a grading rubric is provided:\n"
        prompt += f"**GRADING RUBRIC**\n"
        prompt += f"Your proposed plan will be evaluated according to the following rubric:\n"
        for i, item in enumerate(rubric_list):
            prompt += f"Item {i+1}: {item}\n"
        prompt += "\nMake sure your research plan addresses all aspects mentioned in the grading rubric above, but do not directly mention the rubric items in your response.\n\n"

    prompt += f"Overall Solution Guidelines:\n"
    prompt += f"• The plan should address the goals of the scenario, and account for all constraints and confounders.\n"
    prompt += f"• Do NOT just say WHAT you will do. Explain HOW you will do it and WHY it is needed. Provide clear explanation and justification for each proposed step. The solution inside <solution></solution> tags should be readable for humans, and not in XML itself.\n"
    prompt += f"• The phrasing should NOT be verbose, and NOT be in past tense, as in 'the author's approach' but rather in present tense, as how you would approach the problem.\n"
    prompt += f"• Do not claim to have done any experiments or have results, just provide the plan.\n"
    prompt += f"• Do not add self-proclaimed praises of your solution. For example do NOT say yourself it satisfies some desiderata, we will let the evaluator decide that.\n\n"
    
    prompt += f"Output Format\n\n"
    prompt += f"You can think before you give the final solution, but only the final solution will be judged so make sure to include (potentially repeat) all details in it.\n"
    prompt += f"Put the final solution, the full detailed research plan, inside <solution> </solution> XML tags. This should be information dense, maximum 750 words."

    return prompt

def build_judge_prompt(row, plan_a, plan_b):
    scenario = row.get('Goal', '')
    rubric_list = row.get('Rubric', [])
    reference = row.get('Reference solution', '')

    prompt = f"Evaluate which of the two Proposed Research Plans is better for the Research Scenario based on the provided evaluation criteria.\n\n"
    
    prompt += f"# Research Scenario\n{scenario}\n\n"
    
    prompt += f"# Rubric\n"
    for i, item in enumerate(rubric_list):
        prompt += f"Item {i+1}: {item}\n"
    prompt += "\n"

    if reference:
        prompt += f"# Reference Solution\n"
        prompt += f"Here is a reference solution written by an expert:\n{reference}\n"
        prompt += f"• It is just meant to demonstrate one possible approach that satisfies the scenario. It is not necessary for the proposed research plan you are grading to match all details in the reference solution.\n"
        prompt += f"• The Research Plan you have to grade might have different design choices. This is okay, if the choices are valid, and supported with correct rationale.\n\n"

    prompt += f"# Proposed Research Plan A\n{plan_a}\n\n"
    prompt += f"# Proposed Research Plan B\n{plan_b}\n\n"

    prompt += f"**DESIDERATA**\n"
    prompt += "1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item?\n"
    prompt += "2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it? There should be no self-proclaimed claims of handling something without doing so.\n"
    prompt += "3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses?\n"
    prompt += "4. WELL JUSTIFIED RATIONALE: Is the plan well-motivated and justified?\n"
    prompt += "5. COST AND EFFORT EFFICIENT: Does the plan handle this cost efficiently?\n"
    prompt += "6. NO ETHICAL ISSUES: Does the plan have any potential for negative consequences?\n"
    prompt += "7. CONSISTENT WITH OVERALL PLAN: Is the plan consistent?\n\n"

    prompt += f"# Instructions\n"
    prompt += f"First, compare the two plans against the rubric and desiderata. Identify the strengths and weaknesses of each.\n"
    prompt += f"Then, provide a verdict indicating which plan is better, and a clear text rationale explaining why.\n"
    prompt += f"IMPORTANT: In your rationale, do NOT quote the exact rubric items. Discuss the concepts they represent, but do not leak the specific wording of the rubric.\n\n"
    prompt += f"Please output your response in the following format:\n"
    prompt += f"Rationale: [Your detailed comparison and justification]\n"
    prompt += f"Verdict: [Plan A or Plan B]\n"

    return prompt

def build_improvement_prompt(row, plan_a, plan_b, verdict, rationale):
    scenario = row.get('Goal', '')
    rubric_list = row.get('Rubric', [])

    prompt = f"I will provide you a research scenario and two draft research plans (Plan A and Plan B) along with a critique comparing them. You have to provide an improved, concise yet thoughtful research plan based on the winner.\n\n"
    
    prompt += f"Here is the research scenario.\n"
    prompt += f"Scenario: {scenario}\n\n"
    
    if rubric_list:
        prompt += f"**GRADING RUBRIC**\n"
        for i, item in enumerate(rubric_list):
            prompt += f"Item {i+1}: {item}\n"
        prompt += "\n"

    prompt += f"Here is Draft Plan A:\n{plan_a}\n\n"
    prompt += f"Here is Draft Plan B:\n{plan_b}\n\n"
    
    prompt += f"Here is the Critique and Verdict:\n"
    prompt += f"Verdict: {verdict}\n"
    prompt += f"Rationale: {rationale}\n\n"

    prompt += f"INSTRUCTION: Revise the winning plan ({verdict}) to further improve it based on the feedback. Keep the good parts but address any weaknesses mentioned in the rationale. You can also incorporate any strong points from the losing plan if relevant.\n"
    prompt += f"The solution inside <solution></solution> tags should be readable for humans, and not in XML itself.\n"
    prompt += f"Output Format: Put the final improved solution inside <solution> </solution> XML tags."

    return prompt

def extract_solution(text):
    match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip() # Fallback

def parse_verdict(text):
    # Matches "Verdict" followed optionally by ":" and whitespace/newlines, then "Plan A" or "Plan B"
    # Handles: "**Verdict**: Plan A", "# Verdict\nPlan B", "Verdict: Plan A"
    match = re.search(r"(?:#|\*\*|^)\s*Verdict[:\s]*(?:\*\*|)\s*(Plan [AB])", text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).title() # Returns "Plan A" or "Plan B"

    # Fallback to simple containment if regex fails
    text_lower = text.lower()
    if "plan a" in text_lower and "plan b" not in text_lower:
        return "Plan A"
    if "plan b" in text_lower and "plan a" not in text_lower:
        return "Plan B"
    return "Unknown"

async def run_experiment():
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
    
    selected_indices = sorted(random.sample(range(len(buffer)), NUM_SAMPLES))
    samples = [buffer[i] for i in selected_indices]
    
    results = []
    
    # Initialize Clients
    # client_gen and client_gpt5 can share the default configuration (OpenAI key + Tinker lazy load)
    client_gen = APIClient() 
    client_gpt5 = APIClient()
    
    # client_final_judge uses the "judge" configuration (mapped to Anthropic/Claude by user)
    client_final_judge = APIClient(model_type="judge") 
    
    for i, row in enumerate(samples):
        print(f"\nProcessing Sample {i+1}/{NUM_SAMPLES} (ID: {row.get('q_id', 'N/A')})...")
        
        # 1. Generate 2 Plans
        gen_prompt = build_generation_prompt(row)
        print(f"  Generating Plan A ({MODEL_GENERATOR})...")
        # Use "tinker" as model name to force APIClient to use Tinker backend (configured via env TINKER_BASE_MODEL)
        plan_a_raw = client_gen.generate("tinker", gen_prompt, temperature=0.7, feedback_rounds=0)
        plan_a = extract_solution(plan_a_raw)
        
        print(f"  Generating Plan B ({MODEL_GENERATOR})...")
        plan_b_raw = client_gen.generate("tinker", gen_prompt, temperature=0.7, feedback_rounds=0)
        plan_b = extract_solution(plan_b_raw)
        
        # 2. Judge
        print(f"  Judging ({MODEL_JUDGE})...")
        judge_prompt = build_judge_prompt(row, plan_a, plan_b)
        # Use client_gpt5 for the intermediate judge
        judge_output = client_gpt5.generate(MODEL_JUDGE, judge_prompt, temperature=0.0)
        
        verdict = parse_verdict(judge_output)
        print(f"  Verdict: {verdict}")
        
        # Extract Rationale
        if "Rationale:" in judge_output:
            rationale_part = judge_output.split("Rationale:", 1)[1]
            if "Verdict:" in rationale_part:
                rationale = rationale_part.split("Verdict:", 1)[0].strip()
            else:
                rationale = rationale_part.strip()
        else:
            rationale = judge_output
            
        # 3. Improve
        print(f"  Improving Winner ({MODEL_GENERATOR})...")
        # Now passing all required context: Plan A, Plan B, Verdict, Rationale
        improve_prompt = build_improvement_prompt(row, plan_a, plan_b, verdict, rationale)
        improved_raw = client_gen.generate(MODEL_GENERATOR, improve_prompt, temperature=0.7, feedback_rounds=0)
        improved_plan = extract_solution(improved_raw)
        
        winner_plan = plan_a if verdict == "Plan A" else plan_b

        # 4. Final Judge
        print(f"  Final Evaluation ({MODEL_FINAL_JUDGE})...")
        final_judge_prompt = build_judge_prompt(row, winner_plan, improved_plan)
        # Use client_final_judge (Anthropic) for the final step
        final_output = client_final_judge.generate(MODEL_FINAL_JUDGE, final_judge_prompt, temperature=0.0)
        final_verdict = parse_verdict(final_output)
        print(f"  Final Verdict: {final_verdict}")
        
        results.append({
            "sample_id": row.get('q_id'),
            "scenario": row.get('Goal'),
            "plan_a": plan_a,
            "plan_b": plan_b,
            "judge_rationale": rationale,
            "judge_verdict": verdict,
            "winner_plan": winner_plan,
            "improved_plan": improved_plan,
            "final_verdict_output": final_output,
            "final_verdict": final_verdict
        })
        
        # Incremental save
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
            
    print(f"\nExperiment complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(run_experiment())
