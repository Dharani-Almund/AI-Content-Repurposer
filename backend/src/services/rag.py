import os
import logging
import json
import re
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it in your .env file.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize HuggingFace embeddings explicitly (forcing local computation)
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
        encode_kwargs={'normalize_embeddings': True}  # Ensure embeddings are normalized
    )
    logger.info("Successfully initialized HuggingFace embeddings")
except Exception as e:
    logger.error(f"Error initializing embeddings: {str(e)}")
    raise

# Initialize Chroma with better error handling
try:
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    logger.info("Successfully connected to ChromaDB")
except Exception as e:
    logger.error(f"Error connecting to ChromaDB: {str(e)}")
    raise


# Define CrewAI custom tools with robust error handling
class ChromaRetrieverTool(BaseTool):
    name: str = "ChromaRetrieverTool"
    description: str = "Retrieve relevant content from ChromaDB vector store by formulating effective search queries."

    def _run(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Retrieving content for query: {query}")
            docs = chroma_db.similarity_search(query, k=top_k)
            results = [{
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs]
            logger.info(f"Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"ChromaRetrieverTool error: {str(e)}")
            # Return a valid response even in error cases
            return [{"content": "Error retrieving content. Please try a different query.", "metadata": {}}]


class ContentGeneratorTool(BaseTool):
    name: str = "ContentGeneratorTool"
    description: str = "Generate content in specified format based on context."

    def _run(self, context: str, output_format: str) -> str:
        try:
            logger.info(f"Generating content in format: {output_format}")
            # Create a new LLM instance for each generation to avoid conflicts
            llm = ChatGroq(
                model_name="groq/deepseek-r1-distill-qwen-32b",
                temperature=0.7,
                max_tokens=500,  # Increased token limit
                api_key=groq_api_key
            )

            prompt = f"""
            Convert the following information into a professional {output_format}:
            {context}

            Guidelines:
            - Use appropriate tone for {output_format}
            - Keep it concise (under 500 words)
            - Include key points only
            - Return complete and well-formatted content
            """

            # Add timeout and retries
            response = llm.invoke(prompt)

            if not response or not response.content:
                logger.warning("Empty response from LLM")
                return "Unable to generate content. Please try again with more specific instructions."

            return response.content
        except Exception as e:
            logger.error(f"ContentGeneratorTool error: {str(e)}")
            return f"Error generating content: {str(e)}"


class RAGOrchestrator:
    def __init__(self, model_name="groq/deepseek-r1-distill-qwen-32b"):
        try:
            self.llm = ChatGroq(
                model_name=model_name,
                temperature=0.5,
                max_tokens=500,  # Increased token limit
                api_key=os.getenv("GROQ_API_KEY")
            )
            logger.info(f"Successfully initialized Groq LLM with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {str(e)}")
            raise

        # Initialize tools
        self.retriever_tool = ChromaRetrieverTool()
        self.generator_tool = ContentGeneratorTool()

        self._create_agents()
        self._create_tasks()
        self.crew = Crew(
            agents=[self.router_agent, self.retriever_agent, self.generator_agent],
            tasks=[self.routing_task, self.retrieval_task, self.generation_task],
            verbose=True,
            memory=True,
            cache=True
        )

    def _create_agents(self):
        # Router agent (using existing tools)
        self.router_agent = Agent(
            role="Query Router",
            goal="Analyze user queries and transform them into optimal search terms for retrieval",
            backstory="Expert in query understanding and information flow management. Ensures that search queries capture all relevant aspects of the user's request.",
            llm=self.llm,
            max_iter=2,
            max_rpm=60,
            max_execution_time=150,
            verbose=True
        )

        self.retriever_agent = Agent(
            role="Information Retriever",
            goal="Retrieve comprehensive and relevant information from ChromaDB to support content creation",
            backstory="Specialized in semantic search and contextual understanding to extract precisely what's needed from large knowledge bases.",
            llm=self.llm,
            max_iter=2,
            max_rpm=60,
            max_execution_time=150,
            verbose=True
        )

        self.generator_agent = Agent(
            role="Content Generator",
            goal="Determine the appropriate format, and create professional, engaging content based on retrieved information",
            backstory="Expert copywriter and content strategist with experience across multiple platforms and content types",
            tools=[self.generator_tool],
            llm=self.llm,
            max_iter=2,
            max_rpm=60,
            max_execution_time=150,
            verbose=True,
        )

    def _create_tasks(self):
        # Router task (simplified)
        self.routing_task = Task(
            description="Analyze the user query and extract optimal search terms to use for document retrieval",
            agent=self.router_agent,
            expected_output="""
            Provide a list of 3-5 optimized search terms that:

            1. Capture the core topics and entities from the user query
            2. Include domain-specific terminology (like e-commerce terms for retail queries)
            3. Extract specific metrics or statistics if mentioned in the query
            4. Consider the content format being requested

            Return exact search terms that can be passed directly to a retrieval system.

            For example:
            - Original query: "Prime Big Deal Days performance with US traffic drop stats"
            - Optimized terms: ["Prime Big Deal Days sales", "Amazon traffic metrics", "e-commerce performance statistics"]
            """
        )

        self.retrieval_task = Task(
            description="Retrieve comprehensive and relevant information for the user query from ChromaDB",
            agent=self.retriever_agent,
            expected_output="""To create effective search queries for your retrieval agent system, follow these instructions:

                    Parse the user request carefully to identify the core topic and related subtopics
                    Generate 3-5 specific search terms or phrases based on your analysis
                    Use synonyms, industry terminology, and related concepts to create varied search queries
                    Keep queries focused and concise (not full sentences)
                    Ensure each query captures a different angle or aspect of the user's request
                    Include industry-specific terminology where relevant
                    Avoid overly broad or generic terms that might return irrelevant results
                    Include location-specific terms when the request mentions geographic areas
                    Consider timeframes if the request is about current events or trends
                    Make sure queries align with the type of content being requested (LinkedIn post, blog, etc.)

            When implementing this system, pass the generated search queries to your document retrieval system, then use the retrieved documents as 
            contextual background for generating the requested content without directly returning the documents to the user.""",
            context=[self.routing_task]
        )

        self.generation_task = Task(
            description="Create professional content in the exact format requested by the user (LinkedIn post, tweet, email, etc.) using the retrieved information as context. Format the output appropriately for the platform specified in the user query.",
            agent=self.generator_agent,
            expected_output="""
            Generate ready-to-use content that:

            1. Matches the requested platform format exactly (LinkedIn, Tweet, Email, Blog, etc.)
            2. Follows platform-specific best practices:
               - LinkedIn: Professional tone, 1300 chars max, 3-5 hashtags, clear CTA
               - Tweet: 280 chars max, punchy, 1-2 hashtags, consider thread format
               - Email: Compelling subject line, clear greeting/sign-off, short paragraphs, specific CTA
               - Blog: Engaging headline, structured sections, mixed paragraph lengths, 500-1500 words
               - Instagram: Attention-grabbing opener, 2200 chars max, 5-10 hashtags, visual suggestions
               - Press Release: Standard header, inverted pyramid structure, quotes, formal tone

            3. Incorporates key facts and insights from retrieved information
            4. Uses appropriate length, tone, and formatting for the specific platform
            5. Includes all platform-specific elements (headers, hashtags, greetings, etc.)
            6. Delivers a complete, polished product ready for immediate use
            """,
            context=[self.retrieval_task]
        )

    def process_query(self, query: str, output_format: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: '{query}' for output format: '{output_format}'")
            result = self.crew.kickoff(inputs={
                "query": query,
                "output_format": output_format
            })
            logger.info("Successfully processed query")
            return {
                "content": result,
                "format": output_format,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }


# Create module-level instance with better singleton pattern
_rag_instance = None


def get_rag_instance(model_name: str = "groq/deepseek-r1-distill-qwen-32b") -> RAGOrchestrator:
    global _rag_instance
    if _rag_instance is None:
        try:
            _rag_instance = RAGOrchestrator(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize RAG instance: {str(e)}")
            raise
    return _rag_instance


def process_query(query: str, output_format: str) -> Dict[str, Any]:
    """Public interface for processing queries"""
    try:
        rag = get_rag_instance()
        return rag.process_query(query, output_format)
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


# Explicitly export the public interface
__all__ = ['process_query', 'RAGOrchestrator']