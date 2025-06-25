from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.weaviate import Weaviate, VectorIndex, Distance
from agno.vectordb.search import SearchType
import os
from dotenv import load_dotenv
import weaviate
import logging

logger = logging.getLogger("candidate_search")
logger.setLevel(logging.INFO)

load_dotenv()

async def get_vector_db(collection: str = None):
    host = os.getenv("WEAVIATE_HOST")
    port = 8080
    api_key = os.getenv("WEAVIATE_API_KEY", "rushabh-api")
    collection = collection or os.getenv("WEAVIATE_COLLECTION", "recipes")
    logger.info(f"Connecting to Weaviate at {host}:{port}: {api_key}: {collection}")
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=False,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
        # headers={"X-API-KEY": "rushabh-api"},
    )

    # ✅ Check client health BEFORE passing into Agno's vector wrapper
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
        local=False,
        client=client,
    )


async def create_knowledge_base(urls=None, collection: str = None):
    """
    Create a PDFUrlKnowledgeBase backed by Weaviate.
    """
    collection = collection or "recipes"
    if urls is None:
        urls = []
    vector_db = await get_vector_db(collection)
    return PDFUrlKnowledgeBase(
        urls=urls,
        vector_db=vector_db
    )