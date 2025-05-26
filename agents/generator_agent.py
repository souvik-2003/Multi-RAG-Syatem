from agents.base_agent import BaseAgent
from typing import Dict, Any
from config.settings import settings

class GeneratorAgent(BaseAgent):
    def __init__(self, model_name: str = None):
        super().__init__(model_name or settings.PRIMARY_MODEL)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on input data"""
        context = input_data.get('context', [])
        query = input_data.get('query', '')
        multimodal_context = input_data.get('multimodal_context', {})
        
        # Build context text
        context_text = '\n'.join([chunk.get('content', '') for chunk in context])
        
        # Adjust prompt based on multimodal context
        uncertainty_note = ""
        if multimodal_context.get('has_multimodal_content'):
            uncertainty_note = "\n\nIMPORTANT: The source documents contain images that may contain additional relevant information not captured in this text. Please acknowledge this limitation in your response if the images might be relevant to the question."
        
        prompt = f"""
        Answer the following question based on the provided context. Be accurate and acknowledge any limitations.
        
        Context:
        {context_text}
        
        Question: {query}
        
        Instructions:
        1. Base your answer strictly on the provided context
        2. If information is incomplete, acknowledge the gaps
        3. If you're uncertain, express that uncertainty
        4. Do not make up information not present in the context
        5. Cite specific parts of the context when possible
        {uncertainty_note}
        
        Answer:
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self._make_api_call(messages, max_tokens=1500)
        
        return {
            'response': response,
            'confidence': self.get_confidence_score(response)
        }
