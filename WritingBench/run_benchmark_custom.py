
import os
import sys
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to sys.path to access custom_api
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from custom_api import APIClient
except ImportError:
    # Fallback if running from within WritingBench
    sys.path.append(os.path.join(parent_dir, '..'))
    from custom_api import APIClient

# Import prompt templates
try:
    from WritingBench.prompt import evaluate_system, evaluate_prompt
except ImportError:
    try:
        from prompt import evaluate_system, evaluate_prompt
    except ImportError:
        # Fallback definition if import fails
        evaluate_system = "You are an expert evaluator with extensive experience in evaluating response of given query."
        evaluate_prompt = """
Evaluate the Response based on the Query and Criteria provided following the Scoring Rules.

** Scoring Rules **

"1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
"3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
"5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
"7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
"9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."

-Provide reasons for each score by indicating specific strengths or deficiencies within the Response. Reference exact text passages to justify the score, ensuring that each reason is concrete and aligns with the criteria requirements while highlighting key gaps from the ideal answer.

-Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.

-Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.

-Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.

-Scoring Range: Assign an integer score between 1 to 10

** Output format ** 
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}

** Criteria **
```{criteria}```

** Query **
```{query}```

** Response **
```{response}```

Provide your evaluation based on the criteria restated below:

```{criteria}```

** Output format ** 
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}
```
""".strip()

def load_jsonl(file_path):
    """Loads JSONL file."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {file_path}")
    return data

def save_jsonl(data, file_path, mode='a'):
    """Saves data to JSONL file."""
    with open(file_path, mode, encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def process_gen_field(gen_content):
    marker = "</think>\n\n"
    marker_pos = gen_content.find(marker)
    
    if marker_pos != -1:
        return gen_content[marker_pos + len(marker):]
    else:
        return gen_content

# Refinement Templates
refinement_gen_template = """
I will provide you a query and two draft responses (Response A and Response B) along with a critique comparing them based on specific criteria. You have to provide an improved, concise yet thoughtful response based on the winner.

Here is the Query:
{scenario}

Here is the Evaluation Criteria:
{rubric_section}

Here is Draft Response A:
{plan_a}

Here is Draft Response B:
{plan_b}

Here is the Critique and Verdict:
Verdict: {verdict}
Rationale: {rationale}

INSTRUCTION: Revise the winning response ({verdict}) to further improve it based on the feedback. Keep the good parts but address any weaknesses mentioned in the rationale. The goal is to maximize the score on the Evaluation Criteria.
Output Format: Direct text of the response. Do not use XML tags for the solution unless requested by the query.
"""

refinement_feedback_template = """
Evaluate which of the two Proposed Responses is better for the Query based on the provided Evaluation Criteria.

# Query
{scenario}

# Evaluation Criteria
{rubric_items}

# Proposed Response A
{plan_a}

# Proposed Response B
{plan_b}

# Instructions
1. Compare Response A and Response B against each item in the Evaluation Criteria.
2. Identify the strengths and weaknesses of each response relative to the criteria.
3. Determine which response better satisfies the criteria overall.
4. Provide a verdict (Response A or Response B) and a clear rationale explaining why it is better, citing specific examples.

Please output your response in the following format:
Rationale: [Your detailed comparison and justification]
Verdict: [Response A or Response B]
"""

def parse_solution(text):
    # No XML parsing needed for general writing bench
    return text.strip()

def run_generation(client, queries, output_file, model, feedback_rounds, feedback_model=None, max_samples=None):
    """Runs the generation phase."""
    print(f"\n--- Starting Generation Phase (Model: {model}, Feedback Rounds: {feedback_rounds}, Feedback Model: {feedback_model}) ---")
    
    # Load existing to skip
    existing_indices = set()
    existing_data = load_jsonl(output_file)
    for obj in existing_data:
        existing_indices.add(obj['index'])
    
    count = 0
    generated_count = 0
    
    pbar = tqdm(queries, desc="Generating")
    for query_data in pbar:
        if max_samples and generated_count >= max_samples:
            break
            
        idx = query_data['index']
        if idx in existing_indices:
            continue
            
        query = query_data['query']
        checklist = query_data.get('checklist', [])
        
        # Format Rubric String
        rubric_items_str = ""
        for i, item in enumerate(checklist):
            rubric_items_str += f"Item {i+1}: {item['name']} - {item['criteria_description']}\n"
        
        try:
            # Determine effective feedback model
            effective_feedback_model = feedback_model if feedback_model else model

            # Use custom_api to generate
            response = client.generate(
                model=model,
                prompt=query,
                feedback_rounds=feedback_rounds,
                feedback_model=effective_feedback_model,
                temperature=0.7,
                max_tokens=2048,
                parsing_function=parse_solution,
                # Pass templates
                refinement_prompt_template=refinement_gen_template,
                feedback_prompt_template=refinement_feedback_template,
                # Pass context for templates
                scenario=query,
                rubric_items=rubric_items_str,
                rubric_section=rubric_items_str, # Template uses rubric_section in one place
            )
            
            result = {
                "index": idx,
                "query": query,
                "response": response
            }
            save_jsonl([result], output_file)
            generated_count += 1
            
        except Exception as e:
            print(f"Error generating for index {idx}: {e}")
            import traceback
            traceback.print_exc()

def run_evaluation(client, response_file, query_file, output_file, model, eval_times=1, max_samples=None):
    """Runs the evaluation phase."""
    print(f"\n--- Starting Evaluation Phase (Model: {model}) ---")
    
    # Load responses
    if not os.path.exists(response_file):
        print(f"Response file not found: {response_file}. Skipping evaluation.")
        return

    # Map index to query data (including criteria)
    query_map = {}
    queries = load_jsonl(query_file)
    for obj in queries:
        query_map[obj['index']] = obj

    # Load existing evaluations
    existing_indices = set()
    existing_evals = load_jsonl(output_file)
    for obj in existing_evals:
        existing_indices.add(obj['index'])

    responses = load_jsonl(response_file)
        
    evaluated_count = 0
    pbar = tqdm(responses, desc="Evaluating")
    
    for response_data in pbar:
        if max_samples and evaluated_count >= max_samples:
            break
            
        idx = response_data['index']
        if idx in existing_indices:
            continue
            
        if idx not in query_map:
            print(f"Warning: Index {idx} not found in query file.")
            continue
            
        criteria_list = query_map[idx].get('checklist', [])
        query_text = query_map[idx]['query']
        response_text = process_gen_field(response_data['response'])
        
        eval_result = {
            "index": idx,
            "scores": {}
        }
        
        for criteria in criteria_list:
            c_name = criteria["name"]
            eval_result["scores"][c_name] = []
            
            prompt_data = {
                "query": query_text,
                "response": response_text,
                "criteria": criteria,
            }
            
            formatted_prompt = evaluate_prompt.format(**prompt_data)
            
            for _ in range(eval_times):
                try:
                    # Using client.generate for evaluation
                    eval_response_str = client.generate(
                        model=model,
                        prompt=formatted_prompt,
                        system=evaluate_system,
                        temperature=0.0,
                        max_tokens=1024
                    )
                    
                    # Clean up response for JSON parsing
                    cleaned_response = eval_response_str.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.startswith('```'):
                        cleaned_response = cleaned_response[3:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                    cleaned_response = cleaned_response.strip()
                    
                    try:
                        score_data = json.loads(cleaned_response)
                        if "score" in score_data and "reason" in score_data:
                             eval_result["scores"][c_name].append(score_data)
                        else:
                             print(f"Invalid JSON structure for index {idx}, criteria {c_name}")
                    except json.JSONDecodeError:
                         print(f"JSON Decode Error for index {idx}, criteria {c_name}")

                except Exception as e:
                    print(f"Error evaluating index {idx}, criteria {c_name}: {e}")
        
        save_jsonl([eval_result], output_file)
        evaluated_count += 1

def main():
    parser = argparse.ArgumentParser(description="Run WritingBench with Custom API")
    parser.add_argument("--query_file", type=str, default="WritingBench/benchmark_query/benchmark_all.jsonl", help="Path to query file")
    parser.add_argument("--output_dir", type=str, default="WritingBench/results", help="Directory to save results")
    parser.add_argument("--generator_model", type=str, default="tinker/meta-llama/Llama-3.1-8B-Instruct", help="Model for generation")
    parser.add_argument("--feedback_model", type=str, default=None, help="Model for feedback (default: same as generator)")
    parser.add_argument("--evaluator_model", type=str, default="openai/gpt-4.1-mini", help="Model for evaluation")
    parser.add_argument("--feedback_rounds", type=int, default=0, help="Number of feedback rounds")
    parser.add_argument("--mode", choices=['generation', 'evaluation', 'all'], default='all', help="Mode of operation")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Sanitize model names for filenames
    gen_model_name_safe = args.generator_model.replace('/', '_')
    eval_model_name_safe = args.evaluator_model.replace('/', '_')
    
    # Define output paths
    responses_dir = os.path.join(args.output_dir, "responses")
    scores_dir = os.path.join(args.output_dir, "scores")
    
    if not os.path.exists(responses_dir):
        os.makedirs(responses_dir)
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)
        
    responses_file = os.path.join(responses_dir, f"{gen_model_name_safe}.jsonl")
    scores_file = os.path.join(scores_dir, f"{gen_model_name_safe}.jsonl") # Use generator model name for scores file so calculator uses it as label
    
    try:
        if args.mode in ['generation', 'all']:
            queries = load_jsonl(args.query_file)
            client = APIClient() # Re-instantiate client inside usage to avoid import side-effects if any
            run_generation(client, queries, responses_file, args.generator_model, args.feedback_rounds, feedback_model=args.feedback_model, max_samples=args.max_samples)
            
        if args.mode in ['evaluation', 'all']:
            client = APIClient()
            run_evaluation(client, responses_file, args.query_file, scores_file, args.evaluator_model, max_samples=args.max_samples)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
