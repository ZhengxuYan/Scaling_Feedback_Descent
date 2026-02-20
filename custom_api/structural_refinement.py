import re
import time
import uuid
import os
import json
import logging
from typing import Callable, Any, Dict, List, Optional
from .prompts import prompts

STRUCTURAL_FEEDBACK_TEMPLATE = """You are an expert evaluator. Your task is to evaluate two drafts side-by-side. 

# Task Context
{scenario}

{rubric_and_reference}

# Draft A
{plan_a}

# Draft B
{plan_b}

Please provide your evaluation using exactly the following structured format. Do not omit any sections.

1) Task and evaluation criteria:
[State the goal, constraints, and a list of evaluation criteria.]

2) Head-to-head comparison by criterion:
[For each criterion, specify the winner (A, B, or tie), provide 1-2 concrete references as evidence, and note what to borrow from the loser.]

3) Draft-specific notes:
[For Draft A: Strengths, Weaknesses, Highest-impact fix. For Draft B: Strengths, Weaknesses, Highest-impact fix. Keep these short and actionable.]

4) Synthesis plan (best-of-both):
[Step-by-step merge instructions, ordered by impact. Include exact edits when possible.]

5) Verification and acceptance tests:
[Define what "done" means for the next revision.]

6) Output format for next revision:
[What you want back in the next iteration.]

7) Verdict:
[You MUST conclude your evaluation with a final verdict in exactly this format: "Verdict: Draft A" or "Verdict: Draft B".]
"""

class StructuralRefinementPipeline:
    def __init__(self, 
                 generator: Callable, 
                 feedback_model: Callable,
                 prompts_manager = prompts,
                 parsing_function: Optional[Callable[[str], str]] = None):
        self.generator = generator
        self.feedback_model = feedback_model
        self.prompts = prompts_manager
        self.parsing_function = parsing_function

    def run_feedback_loop(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        feedback_rounds: int = 1,
        system: str = None,
        **kwargs
    ) -> str:
        
        current_prompt = prompt
        history_buffer = []
        structured_history = []
        current_best_response = None
        challenger_wins = 0
        
        print(f"\n[Structural Refinement] Starting feedback loop for {feedback_rounds} rounds...")
        
        try:
            for round_idx in range(1, feedback_rounds + 1):
                if round_idx == 1:
                    print(f"[Structural Refinement] Round {round_idx}/{feedback_rounds}: Evaluating two initial drafts...")
                else:
                    print(f"[Structural Refinement] Round {round_idx}/{feedback_rounds}: Generating and evaluating improved draft...")
                # 1. Generation
                round_context_prompt = current_prompt
                if history_buffer:
                    history_text = "\n\n".join(history_buffer)
                    refinement_template = kwargs.get("refinement_prompt_template")
                    
                    if refinement_template:
                         try:
                             # Try to format custom template
                             format_ctx = {
                                 "scenario": current_prompt,
                                 "prompt": current_prompt,
                                 "history": history_text,
                                 "plan_a": challenger,     # From previous round
                                 "plan_b": incumbent,      # From previous round
                                 "verdict": winning_label, # From previous round
                                 "rationale": feedback,    # From previous round
                             }
                             format_ctx.update(kwargs)
                             round_context_prompt = refinement_template.format(**format_ctx)
                         except Exception as e:
                             logging.warning(f"Failed to format refinement_prompt_template. Using default: {e}")
                             refinement_template = None
                    
                    if not refinement_template:
                        # Use default from prompts manager
                        default_template = self.prompts.get_prompt("refinement_generation_template")
                        if default_template:
                            round_context_prompt = default_template.format(prompt=current_prompt, history=history_text)
                        else:
                             # Hard fallback if prompt is missing
                            raise ValueError("Missing required prompt: 'refinement_generation_template'. Please ensure it is defined in prompts.py or passed in kwargs.")

                messages = [{"role": "user", "content": round_context_prompt}]
                if system:
                    messages.insert(0, {"role": "system", "content": system})

                if round_idx == 1:
                    draft_a = self.generator(messages, temperature, max_tokens)
                    draft_b = self.generator(messages, temperature, max_tokens)
                    
                    if self.parsing_function:
                        draft_a = self.parsing_function(draft_a)
                        draft_b = self.parsing_function(draft_b)

                    challenger = draft_a
                    incumbent = draft_b 
                else:
                    draft_new = self.generator(messages, temperature, max_tokens)
                    if self.parsing_function:
                        draft_new = self.parsing_function(draft_new)
                    challenger = draft_new
                    incumbent = current_best_response

                # 2. Feedback (Structural)
                rubric_and_ref = ""
                if "rubric_items" in kwargs and kwargs["rubric_items"]:
                    rubric_and_ref += f"\n# Rubric\n{kwargs['rubric_items']}\n"
                if "reference_section" in kwargs and kwargs["reference_section"]:
                    if not kwargs["reference_section"].startswith("# Reference Solution"):
                        rubric_and_ref += f"\n# Reference Solution\n{kwargs['reference_section']}\n"
                    else:
                        rubric_and_ref += f"\n{kwargs['reference_section']}\n"

                # Prepare context for formatting
                format_context = {
                    "prompt": current_prompt,
                    "challenger": challenger,
                    "incumbent": incumbent,
                    "plan_a": challenger,      # Alias
                    "plan_b": incumbent,       # Alias
                    "scenario": current_prompt,# Alias/Fallback
                    "rubric_and_reference": rubric_and_ref.strip()
                }
                format_context.update(kwargs)

                try:
                     feedback_prompt = STRUCTURAL_FEEDBACK_TEMPLATE.format(**format_context)
                except Exception as e:
                     logging.warning(f"Failed to format structural feedback template: {e}. Falling back to simple concatenation.")
                     feedback_prompt = f"Here is a writing prompt:\n{current_prompt}\n\nDraft A:\n{challenger}\n\nDraft B:\n{incumbent}\n\nWhich draft is better and why?"

                feedback_messages = [{"role": "user", "content": feedback_prompt}]
                feedback = self.feedback_model(feedback_messages, 0.0, 1024)

                # 3. Verdict
                cleaned_feedback = feedback.strip().lower()
                winning_label = "Draft B"
                winner_is_challenger = False
                
                verdict_match = re.search(r"verdict\*?\*?:\s*(.*)", cleaned_feedback, re.IGNORECASE)
                if verdict_match:
                    verdict_text = verdict_match.group(1).strip().lower()
                    if "draft a" in verdict_text or "response a" in verdict_text or "plan a" in verdict_text:
                        winner_is_challenger = True
                        winning_label = "Draft A"
                    elif "draft b" in verdict_text or "response b" in verdict_text or "plan b" in verdict_text:
                        winner_is_challenger = False
                        winning_label = "Draft B"
                    else:
                        if "draft a" in cleaned_feedback or "plan a" in cleaned_feedback:
                            if "draft b" not in cleaned_feedback and "plan b" not in cleaned_feedback:
                                winner_is_challenger = True
                                winning_label = "Draft A"
                else:
                    if "draft a" in cleaned_feedback or "plan a" in cleaned_feedback:
                        if "draft b" not in cleaned_feedback and "plan b" not in cleaned_feedback:
                            winner_is_challenger = True
                            winning_label = "Draft A"

                if winner_is_challenger:
                    current_best_response = challenger
                    if round_idx > 1:
                        challenger_wins += 1
                        print(f"[Structural Refinement]   -> Improved draft (Challenger) beat the old draft (Incumbent).")
                    else:
                        print(f"[Structural Refinement]   -> Draft A selected as initial best.")
                else:
                    current_best_response = incumbent
                    if round_idx > 1:
                        print(f"[Structural Refinement]   -> Old draft retained (improved draft did not win).")
                    else:
                        print(f"[Structural Refinement]   -> Draft B selected as initial best.")

                # 4. History Update
                if round_idx == 1:
                    history_entry = (
                        f"--- Round 1 ---\n"
                        f"Draft A:\n{challenger}\n\n"
                        f"Draft B:\n{incumbent}\n\n"
                        f"Feedback:\n{feedback}\n"
                        f"Selected Winner: {winning_label}\n"
                    )
                else:
                    history_entry = (
                        f"--- Round {round_idx} ---\n"
                        f"New Challenger (Draft A):\n{challenger}\n\n"
                        f"Previous Best (Draft B):\n{incumbent}\n\n"
                        f"Feedback:\n{feedback}\n"
                        f"Selected Winner: {winning_label}\n"
                    )
                history_buffer.append(history_entry)
                
                structured_history.append({
                    "round": round_idx,
                    "draft_a_challenger": challenger,
                    "draft_b_incumbent": incumbent,
                    "feedback": feedback,
                    "verdict": winning_label,
                    "winner_content": current_best_response
                })

            # Save history
            self._save_history(kwargs, prompt, structured_history, current_best_response)
            
            if feedback_rounds > 1:
                improvement_rounds = feedback_rounds - 1
                print(f"[Structural Refinement] Loop completed. Improved draft win rate: {challenger_wins}/{improvement_rounds} ({challenger_wins/improvement_rounds:.0%})")
            else:
                print(f"[Structural Refinement] Loop completed.")
            
            return current_best_response
        except Exception as e:
            logging.error(f"Error in feedback loop: {e}")
            raise e

    def _save_history(self, kwargs, prompt, structured_history, final_response):
        try:
            history_dir = "structural_refinement_history"
            run_id = kwargs.get('run_id')
            if run_id:
                 run_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(run_id))
                 history_dir = os.path.join(history_dir, run_id)

            if not os.path.exists(history_dir):
                os.makedirs(history_dir)
            
            p_id = kwargs.get('prompt_id', 'unknown_prompt')
            p_id = re.sub(r'[^a-zA-Z0-9_-]', '_', p_id)
            
            timestamp = int(time.time())
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{history_dir}/history_{p_id}_{timestamp}_{unique_id}.json"
            
            save_data = {
                "timestamp": timestamp,
                "prompt_id": kwargs.get('prompt_id'),
                "seed_modifier": kwargs.get('seed_modifier'),
                "original_prompt": prompt,
                "rounds": structured_history,
                "final_response": final_response
            }
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save structural refinement history: {e}")
