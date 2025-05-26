from typing import Dict, Any, List
import uuid
from agents.image_classifier import ImageClassifierAgent
from agents.verifier_agent import VerifierAgent
from agents.base_agent import BaseAgent
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from config.settings import settings
from agents.generator_agent import GeneratorAgent  # Import the new generator


class RAGOrchestrator:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.image_classifier = ImageClassifierAgent()
        self.verifier = VerifierAgent()
        self.generator = GeneratorAgent()  # Use concrete implementation
        
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process and index a new document"""
        try:
            # Step 1: Parse document
            doc_data = self.document_processor.process_document(file_path)
            document_id = str(uuid.uuid4())
            
            # Step 2: Analyze images if present
            image_analysis = {}
            if doc_data.get('images'):
                image_analysis = self.image_classifier.process({
                    'images': doc_data['images'],
                    'text_content': self._extract_text_summary(doc_data)
                })
            
            # Step 3: Create chunks for indexing
            chunks = self._create_chunks(doc_data, image_analysis)
            
            # Step 4: Store in vector database
            self.vector_store.add_document_chunks(chunks, document_id)
            
            return {
                'success': True,
                'document_id': document_id,
                'type': doc_data['type'],
                'chunks_created': len(chunks),
                'has_images': bool(doc_data.get('images')),
                'image_analysis': image_analysis,
                'metadata': doc_data.get('metadata', {})
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Process a query through the multi-agent pipeline"""
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.vector_store.similarity_search(question, k=k)
            
            if not retrieved_chunks:
                return {
                    'success': False,
                    'response': "I don't have enough information to answer this question. Please upload relevant documents first.",
                    'confidence': 0.0,
                    'flags': ['no_relevant_context']
                }
            
            # Step 2: Check for multimodal content
            multimodal_context = self._analyze_multimodal_context(retrieved_chunks)
            
            # Step 3: Generate response
            response = self._generate_response(question, retrieved_chunks, multimodal_context)
            
            # Step 4: Verify response
            verification = self.verifier.process({
                'response': response,
                'context': retrieved_chunks,
                'query': question
            })
            
            # Step 5: Handle verification results
            final_response = self._handle_verification(response, verification, multimodal_context)
            
            return {
                'success': True,
                'response': final_response['response'],
                'confidence': verification['confidence_score'],
                'verified': verification['verified'],
                'flags': verification.get('flags', []),
                'multimodal_context': multimodal_context,
                'sources_used': len(retrieved_chunks),
                'verification_details': verification
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': "An error occurred while processing your question."
            }
    
    def _extract_text_summary(self, doc_data: Dict[str, Any]) -> str:
        """Extract a summary of text content for image analysis"""
        text_content = doc_data.get('text_content', [])
        if not text_content:
            return ""
        
        # Combine first few chunks for context
        summary_parts = []
        char_count = 0
        max_chars = 1000
        
        for content in text_content:
            if char_count >= max_chars:
                break
            
            text = content.get('content', '')
            remaining_chars = max_chars - char_count
            
            if len(text) <= remaining_chars:
                summary_parts.append(text)
                char_count += len(text)
            else:
                summary_parts.append(text[:remaining_chars] + "...")
                break
        
        return '\n'.join(summary_parts)
    
    def _create_chunks(self, doc_data: Dict[str, Any], image_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks for vector storage"""
        chunks = []
        text_content = doc_data.get('text_content', [])
        
        for content in text_content:
            chunk = {
                'content': content.get('content', ''),
                'type': 'text',
                'page': content.get('page'),
                'paragraph': content.get('paragraph'),
                'char_count': content.get('char_count', 0),
                'has_images': False,
                'confidence': 1.0
            }
            
            # Check if this chunk is associated with images
            if image_analysis.get('has_images'):
                chunk['has_images'] = True
                chunk['image_context'] = image_analysis.get('routing_decision')
            
            chunks.append(chunk)
        
        # Add image placeholder chunks if images are critical
        if image_analysis.get('requires_human_review'):
            chunks.append({
                'content': f"[IMPORTANT: This document contains {len(doc_data.get('images', []))} images that may contain critical information not available in text. Human review recommended for complete understanding.]",
                'type': 'image_placeholder',
                'has_images': True,
                'confidence': 0.5
            })
        
        return chunks
    
    def _analyze_multimodal_context(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multimodal context in retrieved chunks"""
        has_images = any(chunk.get('metadata', {}).get('has_images', False) for chunk in chunks)
        image_chunks = [chunk for chunk in chunks if chunk.get('metadata', {}).get('has_images', False)]
        
        return {
            'has_multimodal_content': has_images,
            'image_chunk_count': len(image_chunks),
            'requires_special_handling': len(image_chunks) > len(chunks) / 2,
            'uncertainty_level': 'high' if image_chunks else 'low'
        }
    
    def _generate_response(self, question: str, context: List[Dict[str, Any]], multimodal_context: Dict[str, Any]) -> str:
        """Generate response using the generator agent"""
        result = self.generator.process({
            'context': context,
            'query': question,
            'multimodal_context': multimodal_context
        })
        return result['response']
    
    def _handle_verification(self, response: str, verification: Dict[str, Any], multimodal_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle verification results and adjust response if needed"""
        if verification['verified']:
            return {'response': response}
        
        # If verification failed, modify response
        flags = verification.get('flags', [])
        
        if 'factual_inconsistency' in flags or 'unsupported_claims' in flags:
            modified_response = f"""
            I need to express some uncertainty about my previous response. Based on the available text context, {response}
            
            However, verification indicates there may be some inconsistencies or unsupported claims in my response. Please verify this information independently or consult the original documents directly.
            """
        elif multimodal_context.get('has_multimodal_content'):
            modified_response = f"""
            {response}
            
            Note: The source documents contain images that may provide additional context relevant to your question. For a complete understanding, you may want to review the visual elements in the original documents.
            """
        else:
            modified_response = f"""
            {response}
            
            Please note: I have moderate confidence in this response. You may want to verify this information from the original sources.
            """
        
        return {'response': modified_response}
