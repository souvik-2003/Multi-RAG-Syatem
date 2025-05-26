from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import openai
from config.settings import settings
import json

class BaseAgent(ABC):
    def __init__(self, model_name: str = None, temperature: float = 0.1):
        self.model_name = model_name or settings.PRIMARY_MODEL
        self.temperature = temperature
        self.client = self._setup_openrouter_client()
        
    def _setup_openrouter_client(self):
        """Setup OpenRouter client"""
        return openai.OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY
        )
    
    def _make_api_call(self, messages: list, max_tokens: int = 1000) -> str:
        """Make API call to OpenRouter"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed: {e}")
            return None
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input and return results"""
        pass
    
    def get_confidence_score(self, response: str) -> float:
        """Calculate confidence score for response"""
        confidence_indicators = [
            "I'm confident", "clearly shows", "definitely", 
            "certainly", "without doubt"
        ]
        uncertainty_indicators = [
            "might", "possibly", "unclear", "uncertain",
            "I'm not sure", "it appears", "seems like"
        ]
        
        confidence_count = sum(1 for indicator in confidence_indicators if indicator.lower() in response.lower())
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator.lower() in response.lower())
        
        # Simple scoring mechanism
        base_score = 0.5
        confidence_boost = confidence_count * 0.1
        uncertainty_penalty = uncertainty_count * 0.15
        
        return min(max(base_score + confidence_boost - uncertainty_penalty, 0.0), 1.0)
