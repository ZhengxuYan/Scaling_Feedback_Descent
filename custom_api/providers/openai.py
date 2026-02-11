import os
import requests
import time
import logging
from typing import Optional, Dict, Any
from .base import ModelProvider

class OpenAIProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, request_timeout=240, max_retries=3, retry_delay=5):
        super().__init__(api_key, base_url)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 4096, system: Optional[str] = None, min_p: Optional[float] = 0.1, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        
        # Seed logic (omitted for now based on legacy code check - it was disabled)

        for attempt in range(self.max_retries):
            response = {}
            try:
                # Strip openai/ prefix if present
                if model.startswith("openai/"):
                    model = model.replace("openai/", "")
                    
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                # Check for min_p support
                if min_p is not None and model != "o3" and "openai.com" not in self.base_url:
                     payload["min_p"] = min_p

                # Handle model specific tweaks (o3, gpt-5 variants)
                if model == "o3" or model == "gpt-5-chat-latest" or "gpt-5" in model:
                     if "max_tokens" in payload:
                         del payload["max_tokens"]
                     payload["max_completion_tokens"] = max_tokens
                     # Force temp 1 for o3/gpt-5 reasoning models if needed, following legacy code
                     if model == "o3" or "gpt-5" in model:
                         payload["temperature"] = 1
                         if "gpt-5" in model:
                            payload["reasoning_effort"] = "low"

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Strip thinking blocks
                if "<think>" in content and "</think>" in content:
                    post_think_index = content.find("</think>") + len("</think>")
                    stripped_content = content[post_think_index:].strip()
                    if stripped_content:
                        content = stripped_content
                    else:
                        logging.warning("Model output only contained <think> block. Returning full content.")

                if "<reasoning>" in content and "</reasoning>" in content:
                    post_think_index = content.find("</reasoning>") + len("</reasoning>")
                    stripped_content = content[post_think_index:].strip()
                    if stripped_content:
                        content = stripped_content
                    else:
                        logging.warning("Model output only contained <reasoning> block. Returning full content.")
                    
                return content

            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries}")
            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP Error: {e}")
                if response is not None and hasattr(response, 'text'):
                     logging.error(f"Error Response: {response.text}")
                if e.response.status_code == 429:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
            except Exception as e:
                logging.error(f"Error on attempt {attempt+1}: {e}")

            time.sleep(self.retry_delay)
            
        raise RuntimeError(f"Failed to generate text from OpenAI after {self.max_retries} attempts")
