from typing import Dict, Optional

class PromptManager:
    def __init__(self):
        self._prompts: Dict[str, str] = {}

    def set_prompt(self, name: str, prompt: str):
        self._prompts[name] = prompt

    def get_prompt(self, name: str) -> Optional[str]:
        return self._prompts.get(name)

    def load_defaults(self):
        import os
        import yaml
        
        # Load from YAML file in the same directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "prompts.yaml")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_prompts = yaml.safe_load(f)
                
            if yaml_prompts:
                for name, content in yaml_prompts.items():
                    self.set_prompt(name, content.strip())
            
        except FileNotFoundError:
            print(f"Warning: prompts.yaml not found at {yaml_path}")
        except Exception as e:
            print(f"Error loading prompts.yaml: {e}")


prompts = PromptManager()
prompts.load_defaults()
