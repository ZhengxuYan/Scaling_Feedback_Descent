from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ModelProvider(ABC):
    """
    Abstract base class for Model Providers.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 1024, system: Optional[str] = None, **kwargs) -> str:
        """
        Generate text completion for the given prompt.
        """
        pass
