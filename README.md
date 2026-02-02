# RAG

# pdf-RAG
A Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about their content. The system extracts text from PDFs, creates semantic embeddings, and uses AI to generate context-aware answers with page references.
# Technologies Used
Core Dependencies
      
      Python Libraries
      
      streamlit==1.28.0          # Web application framework
      pyngrok==7.0.0             # Tunneling for public URL access
      pdfplumber==0.10.3         # PDF text extraction
      faiss-cpu==1.7.4           # Vector similarity search
      sentence-transformers==2.2.2  # Text embeddings
      transformers==4.35.0       # NLP models for Q&A
      torch==2.0.1               # Deep learning framework
      numpy==1.24.3              # Numerical computing
      
      System Dependencies
      
      poppler-utils             # PDF processing utilities

# System Architecture
1. Data Flow Pipeline
   
       PDF Upload → Text Extraction → Chunking → Embedding → FAISS Indexing → Query Processing → Answer Generation

2. Component Breakdown
   
        A. Text Extraction Module
              pdfplumber: Extracts selectable text from PDFs
              Text Cleaning: Removes special characters, normalizes whitespace
              Fallback Mechanism: Provides sample text if extraction fails
        
        B. Embedding System
              Model: all-MiniLM-L6-v2 (Sentence Transformers)
              Vector Dimension: 384 dimensions 
              Normalization: L2 normalization for cosine similarity
        
        C. Vector Database
              FAISS: Facebook AI Similarity Search
              Index Type: IndexFlatL2 (Euclidean distance)
              Search: k-nearest neighbors (k=5)
        
        D. Retrieval System
              Query Embedding: Same model for consistency
              Semantic Search: Finds most relevant text chunks
              Similarity Scoring: Distance to similarity conversion
        
        E. Answer Generation
              Primary Model: GPT-2 (via Hugging Face Transformers)
              Fallback: Template-based responses for common topics
              Context Integration: Uses retrieved text chunks as context
        
        F. Web Interface
              Streamlit: Interactive web app
              Real-time Chat: Streaming responses
              PDF Preview: Embedded PDF viewer
              Session Management: Caches processed documents
    
# Janus Multimodal AI Streamlit Demo
A comprehensive web application showcasing Janus-1.3B, a state-of-the-art multimodal AI model that combines:

      Visual Question Answering - Understand and answer questions about images
      Text-to-Image Generation - Create images from textual descriptions

# Features
      Dual-Mode Capabilities:
            Multimodal Understanding: Upload images and ask questions about them
            Text-to-Image Generation: Generate multiple images from text prompts
      
      Interactive Web Interface:
            Clean Streamlit interface with tabbed navigation
            Sidebar controls for parameters
            Real-time image display and generation
      
      Advanced Controls:
            Adjustable generation parameters (temperature, top-p, CFG weight)
            Seed control for reproducible results
            Example prompts for inspiration

# Technical Architecture
Core Dependencies
      
      System Dependencies:
      libgl1-mesa-glx - OpenGL support for image processing
      
      Python Packages:
      Deep Learning Frameworks:
      torch, torchvision, torchaudio - PyTorch ecosystem
      flash-attn - Optimized attention mechanisms
      accelerate - Distributed training/inference
      
      AI/ML Libraries:
      transformers - Hugging Face transformers
      janus - Janus multimodal model framework
      
      Web/UI:
      streamlit - Interactive web application framework
      pyngrok - Secure tunnel for exposing local servers
      
      Image Processing:
      PIL (via pillow) - Image manipulation
      numpy - Numerical operations
      
      Utilities:
      protobuf - Protocol buffers
      huggingface-hub - Model downloading

Code Flow
1. Setup & Installation:
   
            # 1. Install system dependencies
            # 2. Clone Janus repository
            # 3. Install Python packages
            # 4. Download the 1.3B model
2. Model Loading (@st.cache_resource):

            Loads Janus-1.3B model with eager attention implementation
            Initializes processor and tokenizer
            Handles CUDA/CPU device placement
3. Multimodal Understanding Pipeline:

         Input Image + Question → Processor → Model → Generated Text Response
   
            Processes image and text into multimodal embeddings
            Uses causal language modeling for generation
            Implements controlled sampling with temperature/top-p
4. Web Interface (Streamlit):

            Tab 1: Image upload + chat interface
            Tab 2: Text-to-image generation
            Sidebar: Parameter controls and settings

5. Deployment (ngrok):

            Creates secure tunnel for public access
            Manages background processes
            Provides public URL

