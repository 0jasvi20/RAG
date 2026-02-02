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
