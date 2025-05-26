import streamlit as st
import os
import tempfile
from pathlib import Path
from orchestrator.rag_orchestrator import RAGOrchestrator
from config.settings import settings

# Initialize the orchestrator
@st.cache_resource
def get_orchestrator():
    return RAGOrchestrator()

def main():
    st.set_page_config(
        page_title="Multi-Agent RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Agent RAG System")
    st.markdown("### Upload documents and ask questions with advanced multimodal processing")
    
    # Initialize session state
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = get_orchestrator()
    
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Check API key
    if not settings.OPENROUTER_API_KEY:
        st.error("⚠️ Please set your OPENROUTER_API_KEY in the environment variables")
        st.stop()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, DOC, TXT, PNG, JPG, JPEG, GIF"
        )
        
        if uploaded_files:
            process_button = st.button("🔄 Process Documents", type="primary")
            
            if process_button:
                process_documents(uploaded_files)
        
        # Display processed documents
        if st.session_state.processed_documents:
            st.subheader("✅ Processed Documents")
            for doc in st.session_state.processed_documents:
                with st.expander(f"📋 {doc['filename']}"):
                    st.json({
                        'Type': doc['type'],
                        'Chunks': doc['chunks_created'],
                        'Has Images': doc['has_images'],
                        'Status': '✅ Processed'
                    })
        
        # Settings
        with st.expander("⚙️ Settings"):
            retrieval_k = st.slider("Retrieval Results", 3, 10, 5)
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic discussed in the documents?",
            key="user_input"
        )
        
        if user_question and st.button("🚀 Ask", type="primary"):
            process_question(user_question, retrieval_k)
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📝 Chat History")
            for i, interaction in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.container():
                    st.markdown(f"**Question {len(st.session_state.chat_history) - i}:** {interaction['question']}")
                    
                    # Create columns for response and metadata
                    resp_col, meta_col = st.columns([3, 1])
                    
                    with resp_col:
                        st.markdown(f"**Answer:** {interaction['response']}")
                    
                    with meta_col:
                        confidence = interaction.get('confidence', 0.0)
                        verified = interaction.get('verified', False)
                        
                        # Confidence indicator
                        if confidence >= 0.8:
                            confidence_color = "🟢"
                        elif confidence >= 0.6:
                            confidence_color = "🟡"
                        else:
                            confidence_color = "🔴"
                        
                        st.markdown(f"**Confidence:** {confidence_color} {confidence:.2f}")
                        st.markdown(f"**Verified:** {'✅' if verified else '❌'}")
                        
                        if interaction.get('flags'):
                            st.markdown(f"**Flags:** {', '.join(interaction['flags'])}")
                    
                    st.divider()
    
    with col2:
        st.header("📊 System Status")
        
        # System metrics
        if st.session_state.processed_documents:
            total_docs = len(st.session_state.processed_documents)
            total_chunks = sum(doc['chunks_created'] for doc in st.session_state.processed_documents)
            multimodal_docs = sum(1 for doc in st.session_state.processed_documents if doc['has_images'])
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Documents", total_docs)
                st.metric("Chunks", total_chunks)
            with col_b:
                st.metric("Multimodal", multimodal_docs)
                st.metric("Queries", len(st.session_state.chat_history))
        
        # Recent flags and warnings
        if st.session_state.chat_history:
            recent_flags = []
            for interaction in st.session_state.chat_history[-3:]:
                if interaction.get('flags'):
                    recent_flags.extend(interaction['flags'])
            
            if recent_flags:
                st.subheader("⚠️ Recent Warnings")
                for flag in set(recent_flags):
                    st.warning(f"🚨 {flag.replace('_', ' ').title()}")
        
        # Help section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Upload Documents**: Use the sidebar to upload PDFs, Word docs, images, or text files
            2. **Process**: Click "Process Documents" to analyze and index your files
            3. **Ask Questions**: Type questions about your documents in the chat interface
            4. **Review Responses**: Check confidence scores and verification flags
            
            **Features:**
            - 🖼️ Intelligent image handling
            - 🔍 Multi-agent verification
            - 🚫 Hallucination detection
            - 📊 Confidence scoring
            """)

def process_documents(uploaded_files):
    """Process uploaded documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            result = st.session_state.orchestrator.process_document(tmp_file_path)
            
            if result['success']:
                # Add to processed documents list
                doc_info = {
                    'filename': uploaded_file.name,
                    'document_id': result['document_id'],
                    'type': result['type'],
                    'chunks_created': result['chunks_created'],
                    'has_images': result['has_images'],
                    'metadata': result.get('metadata', {})
                }
                st.session_state.processed_documents.append(doc_info)
                
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                
                # Show image analysis if available
                if result.get('image_analysis') and result['image_analysis'].get('has_images'):
                    with st.expander(f"🖼️ Image Analysis for {uploaded_file.name}"):
                        st.json(result['image_analysis'])
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}: {result.get('error')}")
        
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("✅ All documents processed!")

def process_question(question, k):
    """Process user question"""
    with st.spinner("🤔 Thinking..."):
        result = st.session_state.orchestrator.query(question, k=k)
        
        if result['success']:
            # Add to chat history
            interaction = {
                'question': question,
                'response': result['response'],
                'confidence': result['confidence'],
                'verified': result['verified'],
                'flags': result.get('flags', []),
                'multimodal_context': result.get('multimodal_context', {}),
                'sources_used': result.get('sources_used', 0)
            }
            st.session_state.chat_history.append(interaction)
            
            # Show detailed results
            if result.get('flags') or result['confidence'] < 0.7:
                if result['confidence'] < 0.5:
                    st.error(f"⚠️ Low confidence response: {result['response']}")
                elif result['confidence'] < 0.7:
                    st.warning(f"⚠️ Moderate confidence response: {result['response']}")
                else:
                    st.info(result['response'])
            else:
                st.success(result['response'])
            
            # Show verification details in expander
            if result.get('verification_details'):
                with st.expander("🔍 Verification Details"):
                    st.json(result['verification_details'])
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    
    main()
