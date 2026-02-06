import os
import re
import time
import logging
import json
import requests
import random
import string
import asyncio
import uuid
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Tinker imports
try:
    import tinker
    from tinker_cookbook import renderers, model_info, tokenizer_utils

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

try:
    from model_name_subs import MODEL_NAME_SUBS
except ImportError:
    MODEL_NAME_SUBS = {}

load_dotenv()


class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI or other).
    Mimics eqbench usage: we have 'test' vs 'judge' model_type references.
    Also supports 'tinker' models directly via the tinker library.
    """

    def __init__(
        self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5
    ):
        self.model_type = model_type or "default"
        self.is_tinker = False
        self.tinker_client = None
        self.tinker_renderer = None
        self.tinker_critic_renderer = None

        if model_type == "test":
            self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv(
                "TEST_API_URL",
                os.getenv(
                    "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
                ),
            )

            # Check for Tinker configuration - REMOVED eager check relying on env var
            # We will do lazy init in generate()

        elif model_type == "judge":
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv(
                "JUDGE_API_URL",
                os.getenv(
                    "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
                ),
            )
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv(
                "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
            )

        # Detect Anthropic Key
        self.is_anthropic = (self.api_key and self.api_key.startswith("sk-ant-")) or (
            self.model_type == "judge" and "anthropic" in (os.getenv("JUDGE_API_URL") or "")
        )
        if self.is_anthropic:
            logging.info(f"Detected Anthropic Configuration for {self.model_type}")

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if not self.is_tinker:
            logging.debug(
                f"Initialized {self.model_type} API client with URL: {self.base_url}"
            )

        # Gemini API Key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.is_gemini = False
        if self.gemini_api_key and (
            "gemini" in (self.model_type or "").lower() or "google" in (self.base_url or "").lower()
        ):
             # Basic heuristic, though usually we check model name at generate time
             self.is_gemini = True

        # Create a shared session for connection pooling with robust retries
        self.session = requests.Session()
        
        # Define a retry strategy that handles connection errors (RemoteDisconnected)
        from urllib3.util.retry import Retry
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
            raise_on_status=False # We handle status codes in our wrapper
        )
        
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def setup_tinker(self):
        # ... (keep existing implementation) ...
        with open("debug_api.txt", "a") as f:
            f.write("DEBUG: setup_tinker called\n")
        if not TINKER_AVAILABLE:
            logging.error("Tinker library not found. Cannot use Tinker models.")
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: Tinker not available\n")
            return

        self.is_tinker = True
        tinker_api_key = os.getenv("TINKER_API_KEY")
        base_model_name = os.getenv(
            "TINKER_BASE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"
        )
        critic_base_model_name = os.getenv(
            "TINKER_CRITIC_BASE_MODEL", base_model_name
        )
        model_path = os.getenv("TINKER_MODEL_PATH")  # Optional, for trained weights

        logging.info(
            f"Initializing Tinker client with Base: {base_model_name}, Critic Base: {critic_base_model_name}, Path: {model_path}"
        )
        with open("debug_api.txt", "a") as f:
            f.write(f"DEBUG: Init Tinker with {base_model_name}, {critic_base_model_name}, {model_path}\n")

        try:
            service_client = tinker.ServiceClient(api_key=tinker_api_key)
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: ServiceClient created\n")
            
            # --- Generator Setup ---
            tokenizer = tokenizer_utils.get_tokenizer(base_model_name)
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: Generator Tokenizer created\n")

            # Base Client (Generator)
            self.tinker_base_client = service_client.create_sampling_client(
                base_model=base_model_name
            )
            
            renderer_name = model_info.get_recommended_renderer_name(base_model_name)
            self.tinker_renderer = renderers.get_renderer(
                renderer_name, tokenizer=tokenizer
            )
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: Base SamplingClient & Renderer created\n")

            # --- Critic Setup ---
            
            # If critic base model is different, we need a new tokenizer/renderer
            if critic_base_model_name != base_model_name:
                 critic_tokenizer = tokenizer_utils.get_tokenizer(critic_base_model_name)
                 critic_renderer_name = model_info.get_recommended_renderer_name(critic_base_model_name)
                 self.tinker_critic_renderer = renderers.get_renderer(
                     critic_renderer_name, tokenizer=critic_tokenizer
                 )
                 with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Critic Tokenizer & Renderer created for {critic_base_model_name}\n")
            else:
                 self.tinker_critic_renderer = self.tinker_renderer

            # Critic Client (Evaluator)
            if model_path:
                # If we have a specific trained path, we use that on top of the CRITIC base model
                self.tinker_critic_client = service_client.create_sampling_client(
                    base_model=critic_base_model_name, model_path=model_path
                )
                with open("debug_api.txt", "a") as f:
                    f.write("DEBUG: Critic SamplingClient (Custom Path) created\n")
            elif critic_base_model_name != base_model_name:
                # No trained path, but different base model
                self.tinker_critic_client = service_client.create_sampling_client(
                     base_model=critic_base_model_name
                )
                with open("debug_api.txt", "a") as f:
                    f.write("DEBUG: Critic SamplingClient (Different Base) created\n")
            else:
                # Same base model, no trained path -> fallback to same client
                logging.warning(
                    "No TINKER_MODEL_PATH set for critic and same base model. Using base client as critic."
                )
                with open("debug_api.txt", "a") as f:
                    f.write("DEBUG: No Critic path/diff base, using base as critic\n")
                self.tinker_critic_client = self.tinker_base_client

            # Maintain backward compatibility if needed, though we use base/critic explicitly now
            self.tinker_client = self.tinker_base_client

            logging.info("Tinker client initialized successfully.")
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: Tinker setup complete.\n")
        except Exception as e:
            logging.error(f"Failed to initialize Tinker client: {e}")
            with open("debug_api.txt", "a") as f:
                f.write(f"DEBUG: Failed to init Tinker: {e}\n")
            self.is_tinker = False

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8096,
        include_seed=True,
        min_p=0.1,
        system=None,
        use_rationale=True,
        **kwargs
    ) -> str:
        """
        Generic chat-completion style call.
        """
        # Lazy init Tinker if needed
        # We now also check TINKER_BASE_MODEL env var to support custom model names (e.g. "kimi-k2")
        tinker_env_model = os.getenv("TINKER_BASE_MODEL")
        if not self.is_tinker and (
            "tinker" in model.lower() or "tinker" in self.base_url or tinker_env_model
        ):
            self.setup_tinker()

        # Apply Model Name Substitution (Global)
        # This allows users to use aliases (e.g. gemini-2.5-flash-vanilla -> gemini-2.5-flash)
        # to separate ELO results while calling the correct API.
        if model in MODEL_NAME_SUBS:
             logging.info(f"Substituting model alias: {model} -> {MODEL_NAME_SUBS[model]}")
             model = MODEL_NAME_SUBS[model]

        # Route to Anthropic if configured
        if self.is_anthropic or model.startswith("anthropic/"):
             return self.generate_anthropic(model, prompt, temperature, max_tokens, system)

        # Route to Tinker if configured
        # We check explicitly for tinker prefix OR if TINKER_BASE_MODEL is set and aligns with intent
        # If the user sets TINKER_BASE_MODEL, we assume they want to use Tinker for this generation unless it matches another specific handler
        # To avoid hijacking, we might want to be careful, but here we assume if TINKER_BASE_MODEL is present, we prefer Tinker
        # UNLESS the model name explicitly points to another known provider (like gemini/anthropic)
        is_explicit_tinker = model.startswith("tinker") or "tinker" in self.base_url
        is_implicit_tinker = self.is_tinker and tinker_env_model and not (
            model.startswith("gemini/") or model.startswith("google/") or "gemini" in model.lower() or
            model.startswith("anthropic/") or "claude" in model.lower() or
            model.startswith("gpt-") or "openai" in model.lower()
        )
        
        if self.is_tinker and (is_explicit_tinker or is_implicit_tinker):
            return self.generate_tinker(prompt, temperature, max_tokens, system, use_rationale=use_rationale, **kwargs)

        # Route to Gemini if configured
        if model.lower().startswith("gemini/") or model.lower().startswith("google/") or "gemini" in model.lower():
             feedback_rounds = kwargs.get("feedback_rounds", 0)
             
             if feedback_rounds >= 1:
                 # Check if we should/can run the hybrid loop
                 # We attempt it for any rounds >= 1 to ensure consistent behavior (Best of 2 + History)
                 # IF Tinker is available.
                 loop_possible = False
                 try:
                     if not self.is_tinker:
                         self.setup_tinker()
                     if self.is_tinker:
                         loop_possible = True
                 except Exception:
                     pass
             
                 if loop_possible:
                     # Hybrid Loop: Gemini Generator + Tinker Critic
                     
                     # Define wrappers
                     def gemini_generator(messages, temp, max_tok):
                         # Extract user prompt from messages
                         user_text = messages[-1]["content"] if messages else ""
                         sys_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else system
                         return self.generate_gemini(model, user_text, temp, max_tok, sys_text)
                     
                     critic_client = self.tinker_critic_client if self.tinker_critic_client else self.tinker_base_client
                     def tinker_critic(messages, temp, max_tok):
                         return self._tinker_sample_sync(critic_client, self.tinker_critic_renderer, messages, temp, max_tok)
                         
                     # Filter kwargs to remove feedback_rounds if present to avoid dup
                     kwargs_filtered = {k: v for k, v in kwargs.items() if k != "feedback_rounds"}
                     
                     return self._run_feedback_loop(
                         prompt, temperature, max_tokens, 
                         gemini_generator, tinker_critic,
                         system=system, use_rationale=use_rationale, 
                         feedback_rounds=feedback_rounds, **kwargs_filtered
                     )
             
             # Fallback to simple generation if loop not requested OR Tinker not available
             if feedback_rounds >= 1 and not loop_possible:
                  pass # Fall through to simple generation below
             elif feedback_rounds > 1:
                  raise RuntimeError("Tinker is required for feedback loop (rounds > 1) but failed to initialize.")
                 
             return self.generate_gemini(model, prompt, temperature, max_tokens, system)

        messages = [{"role": "user", "content": prompt}]
        if system:
            messages = [{"role": "system", "content": system}] + messages

        # Optionally add random seed block as a system message for judging tasks.
        # This allows us to get variation between iterations without using temp > 0 which compromises judging performance.
        # The reason for doing this is to understand *judging* variance from the same inputs, i.e. when
        # using --redo-judging. In most use cases you won't need to worry about this and can leave it disabled.
        if False:
            if include_seed:
                seed_lines = [
                    "".join(random.choices(string.ascii_letters + string.digits, k=80))
                    for _ in range(5)
                ]
                random_seed_block = (
                    "<RANDOM SEED PLEASE IGNORE>\n"
                    + "\n".join(seed_lines)
                    + "\n</RANDOM SEED>"
                )
                messages = [{"role": "system", "content": random_seed_block}] + messages

        for attempt in range(self.max_retries):
            response = {}
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if (
                    min_p != None
                    and model != "o3"
                    and self.base_url != "https://api.openai.com/v1/chat/completions"
                ):
                    # Only use min_p for the test model (not judge).
                    # If your test model doesn't support min_p, you may need to
                    # disable this here. Alternatively you could use openrouter
                    # which will automatically omit unsupported params.
                    payload["min_p"] = min_p
                if self.base_url == "https://api.openai.com/v1/chat/completions":
                    try:
                        del payload["min_p"]
                    except:
                        pass

                if model == "o3":
                    # o3 has special reqs via the openai api
                    del payload["max_tokens"]
                    payload["max_completion_tokens"] = max_tokens
                    payload["temperature"] = 1
                if model in [
                    "gpt-5-2025-08-07",
                    "gpt-5-mini-2025-08-07",
                    "gpt-5-nano-2025-08-07",
                ]:
                    payload["reasoning_effort"] = "minimal"
                    del payload["max_tokens"]
                    payload["max_completion_tokens"] = max_tokens
                    payload["temperature"] = 1

                if model in ["gpt-5-chat-latest"]:
                    del payload["max_tokens"]
                    payload["max_completion_tokens"] = max_tokens
                    payload["temperature"] = 1
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # strip out any <think> blocks if the model yields that
                if "<think>" in content and "</think>" in content:
                    post_think = content.find("</think>") + len("</think>")
                    content = content[post_think:]
                if "<reasoning>" in content and "</reasoning>" in content:
                    post_think = content.find("</reasoning>") + len("</reasoning>")
                    content = content[post_think:]
                return content
            except requests.exceptions.Timeout:
                logging.warning(
                    f"Request timed out on attempt {attempt+1}/{self.max_retries}"
                )
            except requests.exceptions.HTTPError as e:
                logging.error(e)
                if response:
                    print(response)
                if e.response.status_code == 429:
                    logging.warning("Rate limit. Backing off.")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
            except Exception as e:
                logging.error(e)
                logging.warning(
                    f"Error on attempt {attempt+1}/{self.max_retries}: {str(e)}"
                )

            time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to generate text after {self.max_retries} attempts")

    def generate_anthropic(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        system: str = None,
    ) -> str:
        """
        Direct call to Anthropic API.
        """
        with open("debug_log.txt", "a") as f:
            f.write(f"DEBUG: Calling Anthropic API for model {model}\n")

        # Handle substitutions
        if model in MODEL_NAME_SUBS:
             model = MODEL_NAME_SUBS[model]

        # Handle 'anthropic/' prefix if present (common in some configs)
        real_model = model.replace("anthropic/", "")
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        messages = [{"role": "user", "content": prompt}]
        
        # Anthropic system prompts are a top-level parameter, not a message role
        payload = {
            "model": real_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                if response.status_code != 200:
                   logging.error(f"Anthropic API Error: {response.text}")
                   
                response.raise_for_status()
                data = response.json()
                content = data["content"][0]["text"]
                return content
                
            except Exception as e:
                logging.warning(
                    f"Anthropic Request failed on attempt {attempt+1}: {e}"
                )
                # Exponential backoff
                wait_time = self.retry_delay * (2 ** attempt)
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to generate text from Anthropic after {self.max_retries} attempts")

    def generate_gemini(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system: str = None,
    ) -> str:
        """
        Direct call to Google Gemini API via REST.
        """
        # Strip prefixes if present
        real_model = model.replace("gemini/", "").replace("google/", "")
        
        # https://ai.google.dev/api/rest/v1beta/models/generateContent
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{real_model}:generateContent"
        
        params = {"key": self.gemini_api_key}
        headers = {
            "Content-Type": "application/json"
        }

        # Gemini Content Structure
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        
        # System instructions (Gemini 1.5+ supports this via system_instruction)
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        if system:
             payload["system_instruction"] = {
                 "parts": [{"text": system}]
             }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    params=params, 
                    json=payload,
                    timeout=self.request_timeout,
                )
                
                if response.status_code != 200:
                    logging.error(f"Gemini API Error: {response.text}")
                
                response.raise_for_status()
                data = response.json()
                
                # Handling safe path
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                         return candidate["content"]["parts"][0]["text"]
                    elif "finishReason" in candidate and candidate["finishReason"] != "STOP":
                         logging.warning(f"Gemini finish reason: {candidate['finishReason']}")
                         if "content" not in candidate:
                             raise ValueError("Empty response from Gemini (filtered?)")
                else:
                    raise ValueError(f"Unrecognized Gemini response format: {str(data)[:200]}...")
                    
            except Exception as e:
                logging.warning(
                    f"Gemini Request failed on attempt {attempt+1}: {e}"
                )
                wait_time = self.retry_delay * (2 ** attempt)
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to generate text from Gemini after {self.max_retries} attempts")

    def _tinker_sample_sync(self, client, renderer, messages, temperature, max_tokens):
        model_input = renderer.build_generation_prompt(
            [renderers.Message(role=m["role"], content=m["content"]) for m in messages]
        )
        temp = max(0.01, temperature)
        future = client.sample(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=temp,
                max_tokens=max_tokens,
                stop=renderer.get_stop_sequences(),
            ),
        )
        if hasattr(future, "result"):
            try:
                response = future.result(timeout=300)
            except Exception as e:
                raise e
        else:
            response = future

        parsed_message, _ = renderer.parse_response(
            response.sequences[0].tokens
        )
        return parsed_message["content"]

    def generate_tinker(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str = None,
        use_rationale: bool = True,
        feedback_rounds: int = 1,
        **kwargs
    ) -> str:
        # Wrapper for pure Tinker loop
        
        def tinker_generator(messages, temp, max_tok):
            return self._tinker_sample_sync(self.tinker_base_client, self.tinker_renderer, messages, temp, max_tok)
            
        critic_client = self.tinker_critic_client if self.tinker_critic_client else self.tinker_base_client
        def tinker_critic(messages, temp, max_tok):
            return self._tinker_sample_sync(critic_client, self.tinker_critic_renderer, messages, temp, max_tok)
            
        if feedback_rounds < 1:
            # Vanilla generation
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            return tinker_generator(messages, temperature, max_tokens)

        return self._run_feedback_loop(
            prompt, temperature, max_tokens,
            tinker_generator, tinker_critic,
            system, use_rationale, feedback_rounds, **kwargs
        )

    def _run_feedback_loop(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        generator_func,
        critic_func,
        system: str = None,
        use_rationale: bool = True,
        feedback_rounds: int = 1,
        **kwargs
    ) -> str:
        with open("debug_api.txt", "a") as f:
            f.write(
                f"DEBUG: _run_feedback_loop called (rounds={feedback_rounds}, rationale={use_rationale})\n"
            )
        
        current_prompt = prompt
        history_buffer = []
        structured_history = []
        
        # Track the best response found so far
        current_best_response = None
        
        try:
            for round_idx in range(1, feedback_rounds + 1):
                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Starting Round {round_idx}/{feedback_rounds}\n")
                
                # =======================================================
                # 1. GENERATION PHASE
                # =======================================================
                
                # Context construction
                round_context_prompt = current_prompt
                if history_buffer:
                    history_text = "\n\n".join(history_buffer)
                    
                    refinement_template = kwargs.get("refinement_prompt_template")
                    if refinement_template:
                        # Support simplified format requiring {prompt} and {history}
                        # We allow tolerant formatting to avoid crashes
                        try:
                            round_context_prompt = refinement_template.replace("{prompt}", current_prompt).replace("{history}", history_text)
                        except Exception as e:
                            logging.warning(f"Failed to format custom template: {e}. Using default.")
                            refinement_template = None
                    
                    if not refinement_template:
                        # Default Prompt (Optimization: 'fix_weaknesses' - 60% Win Rate)
                        round_context_prompt = (
                            f"{current_prompt}\n\n"
                            f"HISTORY:\n{history_text}\n"
                            f"TASK: Fix the weaknesses identified in the critique.\n"
                            f"Instruction: Address every point in the feedback history. "
                            f"Ensure the new draft has no flaws and fully refines the areas that were criticized. "
                            f"Do not add unnecessary flair; just make it solid and correct.\n"
                            f"Output the final story text only."
                        )
                
                messages = [{"role": "user", "content": round_context_prompt}]
                if system:
                    messages.insert(0, {"role": "system", "content": system})

                if round_idx == 1:
                    # Round 1: Generate TWO drafts (Cold Start)
                    with open("debug_api.txt", "a") as f:
                        f.write(f"DEBUG: Round 1 - Generating Draft 1 & 2\n")
                    
                    draft_a = generator_func(messages, temperature, max_tokens)
                    draft_b = generator_func(messages, temperature, max_tokens)
                    
                    challenger = draft_a
                    incumbent = draft_b # In Round 1, incumbent is just the second draft
                    
                else:
                    # Round N: Generate ONE draft (Challenger) using History
                    # The incumbent is the current_best_response from previous rounds
                    with open("debug_api.txt", "a") as f:
                        f.write(f"DEBUG: Round {round_idx} - Generating Challenger\n")
                    
                    draft_new = generator_func(messages, temperature, max_tokens)
                    
                    challenger = draft_new
                    incumbent = current_best_response

                # =======================================================
                # 2. COMPARISON PHASE (Critic)
                # =======================================================
                # Convention: Draft A is Challenger, Draft B is Incumbent
                
                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Round {round_idx} - Criticizing\n")
                    
                critic_prompt = (
                    f"Here is a writing prompt:\n{current_prompt}\n\n"
                    f"Draft A:\n{challenger}\n\n"
                    f"Draft B:\n{incumbent}\n\n"
                    "Which draft is better and why?"
                )
                critic_messages = [{"role": "user", "content": critic_prompt}]
                
                evaluation = critic_func(critic_messages, 0.0, 1024)

                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Round {round_idx} - Eval length: {len(evaluation)}\n")

                # =======================================================
                # 3. VERDICT PARSING
                # =======================================================
                
                cleaned_eval = evaluation.strip().lower()
                winning_label = "Draft B" # Default to Incumbent (Safe choice)
                winner_is_challenger = False 

                # Parsing logic trying to find explicit Verdict first
                verdict_match = re.search(r"\*\*Verdict\*\*:\s*(.*)", evaluation, re.IGNORECASE)
                
                if verdict_match:
                    verdict_text = verdict_match.group(1).strip().lower()
                    if "draft a" in verdict_text or "response a" in verdict_text:
                        winner_is_challenger = True
                        winning_label = "Draft A"
                    elif "draft b" in verdict_text or "response b" in verdict_text:
                        winner_is_challenger = False
                        winning_label = "Draft B"
                    else:
                        # Ambiguous verdict text, check body
                         if "draft a" in cleaned_eval and "draft b" not in cleaned_eval:
                             winner_is_challenger = True
                             winning_label = "Draft A"
                else:
                    # Fallback strategies
                    match = re.search(r"Verdict\**:\s*(Draft|Response) [AB]", evaluation, re.IGNORECASE)
                    if match:
                        if " a" in match.group(0).lower():
                            winner_is_challenger = True
                            winning_label = "Draft A"
                        else:
                            winner_is_challenger = False
                            winning_label = "Draft B"
                    else:
                        # Logic: if A is mentioned but B isn't, maybe A won? 
                        # This is risky, but let's stick to safe default (B) unless A is explicitly favored?
                        if "draft a" in cleaned_eval and "draft b" not in cleaned_eval:
                             winner_is_challenger = True
                             winning_label = "Draft A"

                # Update Best Response
                if winner_is_challenger:
                    current_best_response = challenger
                else:
                    current_best_response = incumbent

                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Round {round_idx} Winner: {winning_label}\n")

                # =======================================================
                # 4. HISTORY UPDATE
                # =======================================================
                
                # Helper description for history
                if round_idx == 1:
                    history_entry = (
                        f"--- Round 1 ---\n"
                        f"Draft A:\n{challenger}\n\n"
                        f"Draft B:\n{incumbent}\n\n"
                        f"Critic Evaluation:\n{evaluation}\n"
                        f"Selected Winner: {winning_label}\n"
                    )
                else:
                    # For future rounds, Draft A is the NEW attempt, Draft B was the PREVIOUS Best
                    history_entry = (
                        f"--- Round {round_idx} ---\n"
                        f"New Challenger (Draft A):\n{challenger}\n\n"
                        f"Previous Best (Draft B):\n{incumbent}\n\n"
                        f"Critic Evaluation:\n{evaluation}\n"
                        f"Selected Winner: {winning_label}\n"
                    )
                
                history_buffer.append(history_entry)
                
                # Structured data
                round_data = {
                    "round": round_idx,
                    "draft_a_challenger": challenger,
                    "draft_b_incumbent": incumbent,
                    "evaluation": evaluation,
                    "verdict": winning_label,
                    "winner_content": current_best_response
                }
                structured_history.append(round_data)

            # END LOOP
            
            # Save history to JSON
            try:
                history_dir = "tinker_history"
                
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
                    "final_response": current_best_response
                }
                
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2)
                
                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Saved tinker history to {filename}\n")
                    
            except Exception as e:
                logging.error(f"Failed to save tinker history: {e}")
                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Failed to save history: {e}\n")

            return current_best_response

        except Exception as e:
            with open("debug_api.txt", "a") as f:
                f.write(f"DEBUG: Error in generate_tinker: {e}\n")
            raise e
