import os
from abc import ABC, abstractmethod
from typing import Optional

class PromptLoaderStrategy(ABC):
    @abstractmethod
    def load(self, key: str, fallback: Optional[str] = None) -> str:
        """
        Loads prompt content by key.
        If file reading fails or file is not found, returns fallback content.
        """
        pass

class FilePromptLoaderStrategy(PromptLoaderStrategy):
    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            # Locate prompts directory relative to loader.py
            self.prompts_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.prompts_dir = prompts_dir

    def load(self, key: str, fallback: Optional[str] = None) -> str:
        file_path = os.path.join(self.prompts_dir, f"{key}.txt")
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
        except Exception:
            # Gracefully handle any read/open issues and return the fallback
            pass
        return fallback if fallback is not None else ""
