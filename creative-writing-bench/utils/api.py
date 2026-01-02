import os
import re
import time
import logging
import json
import requests
import random
import string
import asyncio
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
        model_path = os.getenv("TINKER_MODEL_PATH")  # Optional, for trained weights

        logging.info(
            f"Initializing Tinker client with Base: {base_model_name}, Path: {model_path}"
        )
        with open("debug_api.txt", "a") as f:
            f.write(f"DEBUG: Init Tinker with {base_model_name}, {model_path}\n")

        try:
            service_client = tinker.ServiceClient(api_key=tinker_api_key)
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: ServiceClient created\n")
            tokenizer = tokenizer_utils.get_tokenizer(base_model_name)
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: Tokenizer created\n")

            # Base Client (Generator)
            self.tinker_base_client = service_client.create_sampling_client(
                base_model=base_model_name
            )
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: Base SamplingClient created\n")

            # Critic Client (Evaluator)
            if model_path:
                self.tinker_critic_client = service_client.create_sampling_client(
                    base_model=base_model_name, model_path=model_path
                )
                with open("debug_api.txt", "a") as f:
                    f.write("DEBUG: Critic SamplingClient created\n")
            else:
                logging.warning(
                    "No TINKER_MODEL_PATH set for critic. Using base model as critic."
                )
                with open("debug_api.txt", "a") as f:
                    f.write("DEBUG: No Critic path, using base as critic\n")
                self.tinker_critic_client = self.tinker_base_client

            # Maintain backward compatibility if needed, though we use base/critic explicitly now
            self.tinker_client = self.tinker_base_client

            renderer_name = model_info.get_recommended_renderer_name(base_model_name)
            self.tinker_renderer = renderers.get_renderer(
                renderer_name, tokenizer=tokenizer
            )
            logging.info("Tinker client initialized successfully.")
            with open("debug_api.txt", "a") as f:
                f.write("DEBUG: Renderer created. Success.\n")
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
        if not self.is_tinker and (
            "tinker" in model.lower() or "tinker" in self.base_url
        ):
            self.setup_tinker()

        # Route to Anthropic if configured
        if self.is_anthropic or model.startswith("anthropic/"):
             return self.generate_anthropic(model, prompt, temperature, max_tokens, system)

        # Route to Tinker if configured
        if self.is_tinker and (model.startswith("tinker") or "tinker" in self.base_url):
            return self.generate_tinker(prompt, temperature, max_tokens, system, use_rationale=use_rationale, **kwargs)

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

    def _tinker_sample_sync(self, client, messages, temperature, max_tokens):
        model_input = self.tinker_renderer.build_generation_prompt(
            [renderers.Message(role=m["role"], content=m["content"]) for m in messages]
        )
        temp = max(0.01, temperature)
        future = client.sample(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=temp,
                max_tokens=max_tokens,
                stop=self.tinker_renderer.get_stop_sequences(),
            ),
        )
        if hasattr(future, "result"):
            try:
                response = future.result(timeout=300)
            except Exception as e:
                raise e
        else:
            response = future

        parsed_message, _ = self.tinker_renderer.parse_response(
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
    ) -> str:
        with open("debug_api.txt", "a") as f:
            f.write(
                f"DEBUG: generate_tinker called (rounds={feedback_rounds}, rationale={use_rationale})\n"
            )
        
        current_prompt = prompt
        history_buffer = []
        final_response = ""

        try:
            for round_idx in range(1, feedback_rounds + 1):
                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Starting Round {round_idx}/{feedback_rounds}\n")
                
                # If we have history, append it to the prompt for this round
                # User requested: (draft A, draft B, verdict, rationale) roughly
                round_context_prompt = current_prompt
                if history_buffer:
                    history_text = "\n\n".join(history_buffer)
                    # We append history to the user prompt to inform the model of previous attempts
                    round_context_prompt = (
                        f"{current_prompt}\n\n"
                        f"--- Previous Attempts History ---\n"
                        f"{history_text}\n"
                        f"---------------------------------\n"
                        f"Using the above history/feedback to improve, please respond to the original request."
                    )
                
                # 1. Generate two drafts (Base Model)
                messages = [{"role": "user", "content": round_context_prompt}]
                if system:
                    messages.insert(0, {"role": "system", "content": system})

                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Round {round_idx} - Generating Draft 1\n")
                draft1 = self._tinker_sample_sync(
                    self.tinker_base_client, messages, temperature, max_tokens
                )

                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Round {round_idx} - Generating Draft 2\n")
                draft2 = self._tinker_sample_sync(
                    self.tinker_base_client, messages, temperature, max_tokens
                )

                # 2. Criticize (Critic Model)
                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Round {round_idx} - Criticizing\n")
                    
                critic_prompt = (
                    f"Here is a request:\n{current_prompt}\n\n" # Use original prompt for context
                    f"Response A:\n{draft1}\n\n"
                    f"Response B:\n{draft2}\n\n"
                    "Which response is better and why?"
                )
                critic_messages = [{"role": "user", "content": critic_prompt}]
                # Use low temp for critic
                evaluation = self._tinker_sample_sync(
                    self.tinker_critic_client, critic_messages, 0.0, 1024
                )

                # 3. Parse Verdict
                winner = draft1
                cleaned_eval = evaluation.strip().lower()
                winning_response_str = "Response A"
                
                # Simple parsing logic
                if "response b" in cleaned_eval and "response a" not in cleaned_eval:
                    winner = draft2
                    winning_response_str = "Response B"
                elif "response a" in cleaned_eval and "response b" not in cleaned_eval:
                    winner = draft1
                else:
                    # Regex Fallback
                    match = re.search(
                        r"Verdict\**:\s*(Response [AB])", evaluation, re.IGNORECASE
                    )
                    if match and match.group(1).lower() == "response b":
                        winner = draft2
                        winning_response_str = "Response B"

                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Round {round_idx} - Winner selected ({winning_response_str})\n")

                # 4. Refine (Base Model)
                if use_rationale:
                    refine_prompt_text = (
                        f"Here is a request:\n{current_prompt}\n\n"
                        f"Draft Response:\n{winner}\n\n"
                        f"Feedback:\n{evaluation}\n\n"
                        "Please rewrite the response to address the feedback."
                    )
                else:
                    refine_prompt_text = (
                        f"Here is a request:\n{current_prompt}\n\n"
                        f"Draft Response:\n{winner}\n\n"
                        f"Feedback: {winning_response_str} was selected as the better response.\n\n"
                        "Please rewrite the response to be even better."
                    )
                
                refine_messages = [{"role": "user", "content": refine_prompt_text}]
                refined = self._tinker_sample_sync(
                    self.tinker_base_client, refine_messages, temperature, max_tokens
                )
                
                final_response = refined # specific round result
                
                # Add to history for next round
                round_history_entry = (
                    f"--- Round {round_idx} ---\n"
                    f"Draft 1:\n{draft1}\n\n"
                    f"Draft 2:\n{draft2}\n\n"
                    f"Critic Evaluation:\n{evaluation}\n"
                    f"Selected Winner: {winning_response_str}\n" 
                    f"Refined Output:\n{refined}\n"
                )
                history_buffer.append(round_history_entry)

                with open("debug_api.txt", "a") as f:
                    f.write(f"DEBUG: Round {round_idx} Complete\n")

            return final_response

        except Exception as e:
            with open("debug_api.txt", "a") as f:
                f.write(f"DEBUG: Error in generate_tinker: {e}\n")
            raise e
