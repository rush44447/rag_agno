import weaviate
from agno.vectordb.weaviate import Distance, VectorIndex, Weaviate
from agno.vectordb.search import SearchType

async def get_vector_db(collection: str = None):
    host = os.getenv("WEAVIATE_HOST")
    port = 8080
    api_key = os.getenv("WEAVIATE_API_KEY", "rushabh-api")
    collection = collection or os.getenv("WEAVIATE_COLLECTION")
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

    # 1. Define optimized HNSW config
    hnsw_config = {
        "hnswConfig": {
            "efConstruction": 128,  # build-time graph complexity
            "M": 32                 # number of bi-directional links per node
        }
    }

    # 2. Create or update the index with this config
    if client.schema.contains({"class": collection}):
        client.schema.delete_class(collection)
    client.schema.create_class({
        "class": collection,
        "vectorIndexConfig": hnsw_config,
        "properties": [{"name": "text", "dataType": ["text"]}]
    })

    # 3. Wrap it in Agno’s Weaviate connector
    return Weaviate(
        collection=collection,
        vector_index=VectorIndex.HNSW,
        distance=Distance.COSINE,
        search_type=SearchType.hybrid,
        local=True,
        client=client,
    )

















