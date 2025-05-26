from agents.base_agent import BaseAgent
from typing import Dict, Any, List
import json

class ImageClassifierAgent(BaseAgent):
    def __init__(self):
        super().__init__(model_name="anthropic/claude-3-sonnet")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify and analyze images in documents"""
        images = input_data.get('images', [])
        document_context = input_data.get('text_content', '')
        
        if not images:
            return {
                'has_images': False,
                'image_analysis': [],
                'routing_decision': 'text_only'
            }
        
        analysis_results = []
        for image in images:
            if image.get('base64'):
                analysis = self._analyze_image_content(image, document_context)
                analysis_results.append(analysis)
            else:
                analysis_results.append({
                    'type': 'unprocessable',
                    'importance': 'unknown',
                    'recommendation': 'flag_for_human_review'
                })
        
        # Determine routing decision
        routing_decision = self._determine_routing(analysis_results)
        
        return {
            'has_images': True,
            'image_analysis': analysis_results,
            'routing_decision': routing_decision,
            'requires_human_review': any(
                result.get('importance') == 'critical' 
                for result in analysis_results
            )
        }
    
    def _analyze_image_content(self, image: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Analyze individual image content"""
        prompt = f"""
        Analyze this image in the context of the document. 
        
        Document context: {context[:500]}...
        
        Please determine:
        1. Image type (chart, diagram, photo, table, etc.)
        2. Importance level (critical, moderate, low)
        3. Whether the image contains essential information not available in text
        4. Recommended action (process_with_text, defer_to_human, flag_uncertainty)
        
        Respond in JSON format:
        {{
            "type": "chart|diagram|photo|table|other",
            "importance": "critical|moderate|low",
            "contains_essential_info": true|false,
            "recommendation": "process_with_text|defer_to_human|flag_uncertainty",
            "description": "brief description",
            "confidence": 0.0-1.0
        }}
        """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image['base64']}"
                        }
                    }
                ]
            }
        ]
        
        response = self._make_api_call(messages, max_tokens=500)
        
        try:
            return json.loads(response)
        except:
            return {
                'type': 'unknown',
                'importance': 'moderate',
                'contains_essential_info': True,
                'recommendation': 'flag_uncertainty',
                'description': 'Could not analyze image',
                'confidence': 0.1
            }
    
    def _determine_routing(self, analyses: List[Dict[str, Any]]) -> str:
        """Determine how to route the document based on image analysis"""
        if not analyses:
            return 'text_only'
        
        critical_count = sum(1 for a in analyses if a.get('importance') == 'critical')
        essential_info_count = sum(1 for a in analyses if a.get('contains_essential_info'))
        
        if critical_count > 0 or essential_info_count > len(analyses) / 2:
            return 'multimodal_required'
        elif essential_info_count > 0:
            return 'hybrid_processing'
        else:
            return 'text_primary'
