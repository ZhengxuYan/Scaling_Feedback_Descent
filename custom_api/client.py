import os
import logging
from typing import Optional, Dict, Any, Union
import re
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.tinker import TinkerProvider
from .config import config, ModelConfig
from .prompts import prompts
from .refinement import RefinementPipeline

MODEL_NAME_SUBS = {}

class APIClient:
    """
    Client for interacting with LLM API endpoints.
    Routes requests to appropriate providers based on model configuration or naming conventions.
    """

    def __init__(
        self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5
    ):
        self.model_type = model_type or "default"
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._providers = {}


    def _get_provider(self, model: str) -> Any:
        # 1. Explicit provider config
        if model.startswith("anthropic/") or "claude" in model.lower():
            if "anthropic" not in self._providers:
                self._providers["anthropic"] = AnthropicProvider(
                    request_timeout=self.request_timeout,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay
                )
            return self._providers["anthropic"]

        if model.lower().startswith("gemini/") or model.lower().startswith("google/") or "gemini" in model.lower():
            if "google" not in self._providers:
                self._providers["google"] = GoogleProvider(
                    request_timeout=self.request_timeout,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay
                )
            return self._providers["google"]

        if model.startswith("tinker") or "tinker" in model.lower():
             if "tinker" not in self._providers:
                 self._providers["tinker"] = TinkerProvider()
             return self._providers["tinker"]
             
        # Fallback to OpenAI compatible
        
        # Default keys if not using specific provider class
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        if "openai" not in self._providers:
            self._providers["openai"] = OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
                request_timeout=self.request_timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay
            )
        return self._providers["openai"]

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
        
        # Global Substitutions
        if model in MODEL_NAME_SUBS:
             logging.info(f"Substituting model alias: {model} -> {MODEL_NAME_SUBS[model]}")
             model = MODEL_NAME_SUBS[model]

        # Extract model paths for routing
        generator_model_path = kwargs.pop("model_path", None)
        feedback_model_path = kwargs.pop("feedback_model_path", None)

        # Determine actual provider
        provider = self._get_provider(model)
        
        # --- Handle Feedback Loop ---
        feedback_rounds = kwargs.get("feedback_rounds", 0)
        
        if feedback_rounds > 0:
            # We need a generator and a feedback model
            
            # Generator Function Wrapper
            def generator_func(messages, temp, max_tok):
                # Extract text prompt (assuming last message is user prompt, valid for this loop)
                user_text = messages[-1]["content"] if messages else ""
                sys_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else system
                # Call provider directly to avoid recursion or re-routing logic
                kwargs_safe = {k: v for k, v in kwargs.items() if k != "feedback_rounds"}
                if generator_model_path:
                    kwargs_safe["model_path"] = generator_model_path
                return provider.generate(model, user_text, temp, max_tok, sys_text, **kwargs_safe)

            # Feedback Function Wrapper
            feedback_model = kwargs.get("feedback_model", model) # Default to same model if not specified
            
            feedback_prov = self._get_provider(feedback_model)
            def feedback_func(messages, temp, max_tok):
                user_text = messages[-1]["content"] if messages else ""
                kwargs_safe = {k: v for k, v in kwargs.items() if k != "feedback_rounds"}
                if feedback_model_path:
                    kwargs_safe["model_path"] = feedback_model_path
                return feedback_prov.generate(
                    feedback_model, 
                    user_text, 
                    temp, 
                    max_tok, 
                    **kwargs_safe
                )

            # Run Pipeline
            parsing_function = kwargs.get("parsing_function", None)
            pipeline = RefinementPipeline(generator_func, feedback_func, parsing_function=parsing_function)
            
            # Remove parsing_function from kwargs before passing to run_feedback_loop to avoid issues if any
            run_kwargs = {k: v for k, v in kwargs.items() if k != "feedback_rounds" and k != "parsing_function"}
            
            return pipeline.run_feedback_loop(
                prompt, temperature, max_tokens, 
                feedback_rounds=feedback_rounds, 
                system=system, 
                **run_kwargs
            )

        # Standard Generation
        kw = {**kwargs}
        if generator_model_path:
            kw["model_path"] = generator_model_path
        return provider.generate(
            model, 
            prompt, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            system=system, 
            min_p=min_p, 
            **kw
        )
