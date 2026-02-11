
import re
import time
import uuid
import os
import json
import logging
from typing import Callable, Any, Dict, List, Optional
from .prompts import prompts

class RefinementPipeline:
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
        
        try:
            for round_idx in range(1, feedback_rounds + 1):
                # 1. Generation
                round_context_prompt = current_prompt
                if history_buffer:
                    history_text = "\n\n".join(history_buffer)
                    refinement_template = kwargs.get("refinement_prompt_template")
                    
                    if refinement_template:
                         try:
                             # Try to format custom template
                             round_context_prompt = refinement_template.replace("{prompt}", current_prompt).replace("{history}", history_text)
                         except:
                             logging.warning("Failed to format refinement_prompt_template. Using default.")
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

                # 2. Feedback
                feedback_template = kwargs.get("feedback_prompt_template")
                if not feedback_template:
                    feedback_template = self.prompts.get_prompt("refinement_feedback_template")

                else:
                    pass 
                    
                if not feedback_template:
                     raise ValueError("Missing required prompt: 'refinement_feedback_template'.")

                # Prepare context for formatting
                format_context = {
                    "prompt": current_prompt,
                    "challenger": challenger,
                    "incumbent": incumbent,
                    "plan_a": challenger,      # Alias
                    "plan_b": incumbent,       # Alias
                    "scenario": current_prompt # Alias/Fallback
                }
                # Merge kwargs into context to allow passing 'rubric_items', 'reference_section' etc.
                format_context.update(kwargs)

                try:
                     feedback_prompt = feedback_template.format(**format_context)
                except KeyError as e:
                     logging.warning(f"Failed to format 'refinement_feedback_template': Missing key {e}. Falling back to simple concatenation.")
                     feedback_prompt = f"Here is a writing prompt:\n{current_prompt}\n\nDraft A:\n{challenger}\n\nDraft B:\n{incumbent}\n\nWhich draft is better and why?"
                except Exception as e:
                     logging.warning(f"Failed to format 'refinement_feedback_template': {e}. Falling back to simple concatenation.")
                     feedback_prompt = f"Here is a writing prompt:\n{current_prompt}\n\nDraft A:\n{challenger}\n\nDraft B:\n{incumbent}\n\nWhich draft is better and why?"

                feedback_messages = [{"role": "user", "content": feedback_prompt}]
                feedback = self.feedback_model(feedback_messages, 0.0, 1024)

                # 3. Verdict
                cleaned_feedback = feedback.strip().lower()
                winning_label = "Draft B"
                winner_is_challenger = False
                
                verdict_match = re.search(r"\*\*Verdict\*\*:\s*(.*)", cleaned_feedback, re.IGNORECASE)
                if verdict_match:
                    verdict_text = verdict_match.group(1).strip().lower()
                    if "draft a" in verdict_text or "response a" in verdict_text:
                        winner_is_challenger = True
                        winning_label = "Draft A"
                    elif "draft b" in verdict_text or "response b" in verdict_text:
                         winner_is_challenger = False
                         winning_label = "Draft B"
                    else:
                         if "draft a" in cleaned_feedback and "draft b" not in cleaned_feedback:
                              winner_is_challenger = True
                              winning_label = "Draft A"
                else:
                    match = re.search(r"Verdict\**:\s*(Draft|Response) [AB]", cleaned_feedback, re.IGNORECASE)
                    if match:
                         if " a" in match.group(0).lower():
                             winner_is_challenger = True
                             winning_label = "Draft A"
                    else:
                         if "draft a" in cleaned_feedback and "draft b" not in cleaned_feedback:
                              winner_is_challenger = True
                              winning_label = "Draft A"

                if winner_is_challenger:
                    current_best_response = challenger
                else:
                    current_best_response = incumbent

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
            
            return current_best_response
        except Exception as e:
            logging.error(f"Error in feedback loop: {e}")
            raise e

    def _save_history(self, kwargs, prompt, structured_history, final_response):
        try:
            history_dir = "refinement_history"
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
            logging.error(f"Failed to save refinement history: {e}")
