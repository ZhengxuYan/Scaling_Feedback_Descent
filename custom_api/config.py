from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class ModelConfig:
    provider: str
    model_name: str
    api_key_env: Optional[str] = None
    base_url_env: Optional[str] = None
    parameters: Dict = field(default_factory=dict)

@dataclass
class RoleConfig:
    generator: ModelConfig
    critic: ModelConfig
    judge: ModelConfig

class ConfigManager:
    """
    Manages configuration for models and roles.
    """
    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._roles: Dict[str, str] = {} # role -> model_alias

    def register_model(self, alias: str, config: ModelConfig):
        self._models[alias] = config

    def set_role_model(self, role: str, model_alias: str):
        if model_alias not in self._models:
            raise ValueError(f"Model alias '{model_alias}' not registered.")
        self._roles[role] = model_alias

    def get_model_config(self, alias: str) -> Optional[ModelConfig]:
        return self._models.get(alias)

    def get_role_config(self, role: str) -> Optional[ModelConfig]:
        alias = self._roles.get(role)
        if alias:
            return self.get_model_config(alias)
        return None

# Default Instance
config = ConfigManager()
