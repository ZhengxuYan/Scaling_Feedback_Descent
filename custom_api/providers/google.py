import os
import requests
import time
import logging
from typing import Optional, Dict, Any
from .base import ModelProvider

class GoogleProvider(ModelProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, request_timeout=240, max_retries=3, retry_delay=5):
        super().__init__(api_key, base_url)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 4096, system: Optional[str] = None, **kwargs) -> str:
        real_model = model.replace("gemini/", "").replace("google/", "")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{real_model}:generateContent"
        
        params = {"key": self.api_key}
        headers = {
            "Content-Type": "application/json"
        }

        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        
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
                response = requests.post(
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
                logging.warning(f"Gemini Request failed on attempt {attempt+1}: {e}")
                wait_time = self.retry_delay * (2 ** attempt)
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to generate text from Gemini after {self.max_retries} attempts")
