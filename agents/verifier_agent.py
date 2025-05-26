from agents.base_agent import BaseAgent
from typing import Dict, Any, List
import json

class VerifierAgent(BaseAgent):
    def __init__(self):
        super().__init__(model_name="openai/gpt-4-turbo-preview", temperature=0.0)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify response quality and detect hallucinations"""
        response = input_data.get('response', '')
        retrieved_context = input_data.get('context', [])
        query = input_data.get('query', '')
        
        # Perform multiple verification checks
        factual_check = self._check_factual_consistency(response, retrieved_context)
        context_check = self._check_context_grounding(response, retrieved_context)
        uncertainty_check = self._check_uncertainty_handling(response, query)
        
        overall_confidence = min(
            factual_check['confidence'],
            context_check['confidence'],
            uncertainty_check['confidence']
        )
        
        return {
            'verified': overall_confidence >= 0.7,
            'confidence_score': overall_confidence,
            'factual_consistency': factual_check,
            'context_grounding': context_check,
            'uncertainty_handling': uncertainty_check,
            'recommendation': self._get_recommendation(overall_confidence),
            'flags': self._get_flags(factual_check, context_check, uncertainty_check)
        }
    
    def _check_factual_consistency(self, response: str, context: List[Dict]) -> Dict[str, Any]:
        """Check if response is factually consistent with context"""
        context_text = '\n'.join([chunk.get('content', '') for chunk in context])
        
        prompt = f"""
        Compare the response with the provided context and check for factual consistency.
        
        Context:
        {context_text}
        
        Response:
        {response}
        
        Check for:
        1. Any claims not supported by the context
        2. Contradictions with the context
        3. Invented details or hallucinations
        
        Respond in JSON:
        {{
            "is_consistent": true|false,
            "confidence": 0.0-1.0,
            "issues": ["list of specific issues"],
            "unsupported_claims": ["list of unsupported claims"]
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        result = self._make_api_call(messages, max_tokens=800)
        
        try:
            return json.loads(result)
        except:
            return {
                "is_consistent": False,
                "confidence": 0.3,
                "issues": ["Could not parse verification result"],
                "unsupported_claims": []
            }
    
    def _check_context_grounding(self, response: str, context: List[Dict]) -> Dict[str, Any]:
        """Check if response is properly grounded in context"""
        if not context:
            return {
                "is_grounded": False,
                "confidence": 0.0,
                "coverage": 0.0,
                "notes": "No context provided"
            }
        
        context_text = '\n'.join([chunk.get('content', '') for chunk in context])
        
        prompt = f"""
        Evaluate how well the response is grounded in the provided context.
        
        Context:
        {context_text}
        
        Response:
        {response}
        
        Rate:
        1. Grounding quality (0.0-1.0)
        2. Context coverage utilization
        3. Any gaps between context and response
        
        Respond in JSON:
        {{
            "is_grounded": true|false,
            "confidence": 0.0-1.0,
            "coverage": 0.0-1.0,
            "notes": "explanation"
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        result = self._make_api_call(messages, max_tokens=600)
        
        try:
            return json.loads(result)
        except:
            return {
                "is_grounded": True,
                "confidence": 0.5,
                "coverage": 0.5,
                "notes": "Could not evaluate grounding"
            }
    
    def _check_uncertainty_handling(self, response: str, query: str) -> Dict[str, Any]:
        """Check if response appropriately handles uncertainty"""
        prompt = f"""
        Evaluate how well the response handles uncertainty and knowledge gaps.
        
        Query: {query}
        Response: {response}
        
        Check if the response:
        1. Acknowledges limitations when appropriate
        2. Avoids overconfident claims
        3. Indicates when information is incomplete
        
        Respond in JSON:
        {{
            "handles_uncertainty": true|false,
            "confidence": 0.0-1.0,
            "overconfidence_detected": true|false,
            "notes": "explanation"
        }}
        """
        
        messages = [{"role": "user", "content": prompt}]
        result = self._make_api_call(messages, max_tokens=500)
        
        try:
            return json.loads(result)
        except:
            return {
                "handles_uncertainty": True,
                "confidence": 0.5,
                "overconfidence_detected": False,
                "notes": "Could not evaluate uncertainty handling"
            }
    
    def _get_recommendation(self, confidence: float) -> str:
        """Get recommendation based on confidence score"""
        if confidence >= 0.8:
            return "accept"
        elif confidence >= 0.6:
            return "review"
        else:
            return "reject"
    
    def _get_flags(self, factual: Dict, context: Dict, uncertainty: Dict) -> List[str]:
        """Generate warning flags"""
        flags = []
        
        if not factual.get('is_consistent', True):
            flags.append("factual_inconsistency")
        
        if not context.get('is_grounded', True):
            flags.append("poor_grounding")
        
        if uncertainty.get('overconfidence_detected', False):
            flags.append("overconfidence")
        
        if factual.get('unsupported_claims'):
            flags.append("unsupported_claims")
        
        return flags
