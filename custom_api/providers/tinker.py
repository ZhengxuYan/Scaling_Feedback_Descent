import os
import re
import time
import json
import uuid
import logging
from typing import Optional, Dict, Any, List
from .base import ModelProvider

# Try imports
try:
    import tinker
    from tinker_cookbook import renderers, model_info, tokenizer_utils
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

class TinkerProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key, base_url)
        self.api_key = api_key or os.getenv("TINKER_API_KEY")
        # Cache for initialized models: {(model_name, model_path): {'client': client, 'renderer': renderer}}
        self.models_cache = {}
        
    def _initialize(self, model_name: str, model_path: Optional[str] = None):
        # Clean model name if it comes with prefix
        original_name = model_name
        if model_name.lower().startswith("tinker/"):
            model_name = model_name[7:]
            
        # Check if already initialized with this exact config
        cache_key = (model_name, model_path)
        if cache_key in self.models_cache:
            return

        if not TINKER_AVAILABLE:
            raise ImportError("Tinker library not found. Cannot use Tinker models.")

        try:
            service_client = tinker.ServiceClient(api_key=self.api_key)
            
            # Use passed model name, fallback to env if somehow empty
            target_model = model_name or os.getenv("TINKER_BASE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
            
            log_msg = f"Initializing Tinker provider with model: {target_model}"
            if model_path:
                log_msg += f" (checkpoint: {model_path})"
            logging.info(log_msg)

            # Generator Setup
            tokenizer = tokenizer_utils.get_tokenizer(target_model)
            
            # Pass model_path if provided
            if model_path:
                base_client = service_client.create_sampling_client(base_model=target_model, model_path=model_path)
            else:
                base_client = service_client.create_sampling_client(base_model=target_model)
            
            renderer_name = model_info.get_recommended_renderer_name(target_model)
            renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

            # Store in cache
            self.models_cache[cache_key] = {
                'client': base_client,
                'renderer': renderer
            }
            
        except Exception as e:
            logging.error(f"Failed to initialize Tinker provider with {model_name} (path={model_path}): {e}")
            raise e

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

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 8096, system: Optional[str] = None, **kwargs) -> str:
        # Extract model_path if provided in kwargs
        model_path = kwargs.pop('model_path', None)
        
        self._initialize(model, model_path)
        
        # Determine strict model name key
        model_key = model
        if model_key.lower().startswith("tinker/"):
            model_key = model_key[7:]
            
        # Retrieve from cache
        cache_key = (model_key, model_path)
        model_data = self.models_cache[cache_key]
        client = model_data['client']
        renderer = model_data['renderer']
        
        messages = [{"role": "user", "content": prompt}]
        
        # Simple generation
        return self._tinker_sample_sync(client, renderer, messages, temperature, max_tokens)
