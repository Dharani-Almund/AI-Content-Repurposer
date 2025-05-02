import os
import logging
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it in your .env file.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGroq(
    model_name="groq/gemma-7b-it",
    temperature=0.7,
    max_tokens=500,
    api_key=groq_api_key
)

# Define functions for tools (not decorated with @tool)
def chroma_retriever(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant content from ChromaDB vector store."""
    try:
        docs = chroma_db.similarity_search(query, k=top_k)
        return [{
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in docs]
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        return [{"error": str(e)}]

def content_generator(context: str, output_format: str) -> str:
    """Generate content in specified format based on context."""
    try:
        prompt = f"""
        Convert the following information into a professional {output_format}:
        {context}

        Guidelines:
        - Use appropriate tone for {output_format}
        - Keep it concise (under 300 words)
        - Include key points only
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return f"Error generating content: {str(e)}"