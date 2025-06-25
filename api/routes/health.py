from fastapi import APIRouter, HTTPException, Depends
from utils.dttm import current_utc_str
from pydantic import BaseModel
from typing import Dict, Optional
from db.ag_node import create_knowledge_base, get_vector_db
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.agent import Agent
from fastapi.concurrency import run_in_threadpool
from agents.settings import agent_settings
import inspect

######################################################
## Request model
######################################################
class ChatRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, str]] = None

######################################################
## Router for health checks and chat
######################################################
health_check_router = APIRouter(tags=["Health"])

@health_check_router.get("/health")
def get_health():
    """Check the health of the API"""
    return {
        "status": "success",
        "router": "health",
        "path": "/health",
        "utc": current_utc_str(),
    }

async def get_kb():
    urls = ["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"]
    return await create_knowledge_base(urls)


@health_check_router.post("/search")
async def chat_with_DB(chat: ChatRequest, kb=Depends(get_kb)):
    try:
        search_results = kb.search(chat.prompt, 3)
        return {"results": [getattr(r, "content", str(r)) for r in search_results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

@health_check_router.post("/store")
async def save_DB(chat: ChatRequest, kb=Depends(get_kb)):
    try:
        if kb is None:
            raise HTTPException(status_code=500, detail="Knowledge base was not initialized.")

        url = chat.prompt.strip()

        if url and url not in kb.urls:
            kb.urls.append(url)

        # Safely handle async or sync loading
        if hasattr(kb, "aload") and inspect.iscoroutinefunction(kb.aload):
            await kb.aload(recreate=False)
        elif hasattr(kb, "load"):
            kb.load(recreate=False)
        else:
            raise HTTPException(status_code=500, detail="No valid load method found on knowledge base.")

        return {
            "status": "success",
            "message": f"Added {url} to knowledge base."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Store error: {e}")


@health_check_router.get("/weaviate-health")
async def weaviate_health_check():
    try:
        vector_db = await get_vector_db()
        schema = vector_db.client.collections.list_all()

        return {
            "status": "success",
            "message": "Connected to Weaviate successfully.",
            "schema": schema
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Weaviate: {e}")

@health_check_router.post("/scrape")
async def scrape(chat: ChatRequest):
    try:
        logger.info(f"Scraping: {chat.prompt}")
        knowledge_base = WebsiteKnowledgeBase(
            urls=["https://docs.agno.com/introduction", "https://docs.agno.com/introduction/agents"],
            max_links=2,
            vector_db=await get_vector_db("website"),
        )

        agent = Agent(
            knowledge=knowledge_base,
            search_knowledge=True,
            model=OpenAIChat(
                id=agent_settings.gpt_4o_mini,
                api_key=agent_settings.openai_api_key,
                max_tokens=agent_settings.default_max_completion_tokens,
                temperature=agent_settings.default_temperature,
            ),
        )

        await knowledge_base.aload(recreate=False)

        response = agent.run(chat.prompt)

        return {
            "status": "success",
            "question": chat.prompt,
            "answer": response.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# custom_qa_mcp_server.py
import os
import asyncio
from mcp.server.fastmcp import FastMCP
from agno.agent import Agent
from agno.knowledge import WebsiteKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.vectordb.weaviate import Weaviate
from agno.vectordb.distance import Distance
from agno.vectordb.index import VectorIndex
from agno.vectordb.search import SearchType
import weaviate

# Initialize your MCP server
mcp = FastMCP("knowledge_qa_assistant")

async def get_vector_db(collection: str = None):
    """Your existing get_vector_db function"""
    host = os.getenv("WEAVIATE_HOST")
    port = 8080
    api_key = os.getenv("WEAVIATE_API_KEY", "rushabh-api")
    collection = collection or os.getenv("WEAVIATE_COLLECTION", "recipes")
    
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=False,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
    )

    # Check client health
    try:
        if not client.is_ready():
            raise RuntimeError("Weaviate client is not ready or connection failed")
    except Exception as e:
        raise RuntimeError(f"Weaviate health check failed: {str(e)}")

    return Weaviate(
        collection=collection,
        vector_index=VectorIndex.HNSW,
        distance=Distance.COSINE,
        search_type=SearchType.hybrid,
        local=True,
        client=client,
    )

@mcp.tool()
async def ask_question(prompt: str) -> str:
    """
    Ask a question using the knowledge base and get an AI-powered answer.
    
    Args:
        prompt: The question to ask
        
    Returns:
        The AI-generated answer based on the knowledge base
    """
    try:
        # Initialize knowledge base (similar to your existing code)
        kb = WebsiteKnowledgeBase(
            urls=[],  # Not loading from URLs
            max_links=1,
            vector_db=await get_vector_db("website")
        )

        # Create agent with knowledge base
        agent = Agent(
            knowledge=kb,
            search_knowledge=True,
            model=OpenAIChat(
                id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
            ),
        )

        # Get response from agent
        response = agent.run(prompt)
        return response.content

    except Exception as e:
        return f"Error processing question: {str(e)}"

@mcp.tool()
async def get_knowledge_base_info() -> str:
    """
    Get information about the current knowledge base configuration.
    
    Returns:
        Information about the knowledge base setup
    """
    try:
        vector_db = await get_vector_db("website")
        return f"Knowledge base connected to Weaviate at {os.getenv('WEAVIATE_HOST')}. Collection: website"
    except Exception as e:
        return f"Error getting knowledge base info: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server with SSE transport
    mcp.run(transport="sse", port=8001)



async def run_agent(message: str) -> None:
    """Run the filesystem agent with the given message."""

    file_path = str(Path(__file__).parent.parent.parent.parent)

    # MCP server to access the filesystem (via `npx`)
    async with MCPTools(f"npx -y @modelcontextprotocol/server-filesystem {file_path}") as mcp_tools:
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[mcp_tools],
            instructions=dedent("""\
                You are a filesystem assistant. Help users explore files and directories.

                - Navigate the filesystem to answer questions
                - Use the list_allowed_directories tool to find directories that you can access
                - Provide clear context about files you examine
                - Use headings to organize your responses
                - Be concise and focus on relevant information\
            """),
            markdown=True,
            show_tool_calls=True,
        )

        # Run the agent
        await agent.aprint_response(message, stream=True)


# Example usage
if __name__ == "__main__":
    # Basic example - exploring project license
    asyncio.run(run_agent("What is the license for this project?"))