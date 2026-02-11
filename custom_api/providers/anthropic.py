import os
import requests
import time
import logging
from typing import Optional, Dict, Any
from .base import ModelProvider

class AnthropicProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, request_timeout=240, max_retries=3, retry_delay=5):
        super().__init__(api_key, base_url)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 4096, system: Optional[str] = None, **kwargs) -> str:
        real_model = model.replace("anthropic/", "")
        
        messages = [{"role": "user", "content": prompt}]
        
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
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
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
                logging.warning(f"Anthropic Request failed on attempt {attempt+1}: {e}")
                wait_time = self.retry_delay * (2 ** attempt)
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to generate text from Anthropic after {self.max_retries} attempts")
