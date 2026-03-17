"""
RAG-enabled Health Agent
Uses vector database to retrieve relevant health/time management knowledge
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from loguru import logger

# Load environment variables
load_dotenv()



class RAGHealthKnowledge:
    """RAG system for health and time management knowledge"""
    
    def __init__(self, pdf_path: str, persist_directory: str = "./health_knowledge_db"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.vectordb = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load or create vector database
        self._initialize_vectordb()
    
    def _initialize_vectordb(self):
        """Initialize or load existing vector database"""
        
        # Check if DB already exists
        if os.path.exists(self.persist_directory):
            logger.info(f"📚 Loading existing vector database from {self.persist_directory}")
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.success(f"✅ Vector DB loaded with {self.vectordb._collection.count()} documents")
        else:
            logger.info(f"📖 Creating new vector database from {self.pdf_path}")
            self._create_vectordb()
    
    def _create_vectordb(self):
        """Create vector database from PDF"""
        
        # Load PDF
        logger.info(f"Loading PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Create vector store
        logger.info("Creating embeddings and vector store...")
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.success(f"✅ Vector DB created with {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 3) -> list:
        """Retrieve top-k relevant documents for query"""
        
        if self.vectordb is None:
            logger.error("Vector database not initialized")
            return []
        
        logger.info(f"🔍 RAG Query: {query[:100]}...")
        results = self.vectordb.similarity_search(query, k=k)
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        return results
    
    def get_context(self, query: str, k: int = 3) -> str:
        """Get formatted context string from retrieved documents"""
        
        docs = self.retrieve(query, k=k)
        
        if not docs:
            return "No relevant knowledge found."
        
        context = "📚 RELEVANT RESEARCH & KNOWLEDGE:\n" + "="*50 + "\n\n"
        
        for i, doc in enumerate(docs, 1):
            context += f"Source {i} (Page {doc.metadata.get('page', 'unknown')}):\n"
            context += f"{doc.page_content}\n\n"
            context += "-"*50 + "\n\n"
        
        return context


# Global RAG instance (lazy loaded)
_rag_instance = None


def get_rag_health_knowledge():
    """Get or create RAG health knowledge instance"""
    global _rag_instance
    
    if _rag_instance is None:
        pdf_path = os.getenv('HEALTH_KNOWLEDGE_PDF', './Module_1_Effective_Time_Management_in_Personal_Development.pdf')
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            return None
        
        _rag_instance = RAGHealthKnowledge(pdf_path)
    
    return _rag_instance