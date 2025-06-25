from agno.vectordb.search import SearchType
from agno.agent import Agent
from agno.knowledge.base import AgentKnowledge
from agno.vectordb.pgvector import PgVector  # or any other supported vector DB
from agno.models.openai import OpenAIChat

# 1. Setup Vector Database with Hybrid Search
vector_db = PgVector(
    table_name="your_table_name",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    search_type=SearchType.hybrid  # Enable hybrid search
)

# 2. Create Knowledge Base
knowledge_base = AgentKnowledge(
    vector_db=vector_db,
    # You can add documents here or load them separately
)

@router.post("/question")
async def question(chat: ChatRequest):
    prompt = chat.prompt.strip()
    
    # 1. Initial retrieve with hybrid search using Agno's search method
    # Note: Agno uses 'limit' instead of 'top_k'
    results = await vector_db.search(
        query=prompt,
        limit=20,  # Agno uses 'limit' instead of 'top_k'
        # search_type is already set in vector_db initialization
    )

    # 2. For each doc, compute extra signals (custom re-ranking)
    re_ranked = []
    for doc in results:
        # Access document metadata and scores
        bm25_score = doc.meta.get("bm25_score", 0) if hasattr(doc, 'meta') else 0
        embed_score = doc.score if hasattr(doc, 'score') else 0
        feedback_score = await get_feedback_score(doc.id)  # Your custom function

        # 3. Combine with weights
        combined = (
            0.5 * embed_score +
            0.3 * bm25_score +
            0.2 * feedback_score
        )
        re_ranked.append((combined, doc))

    # 4. Sort by combined score descending
    re_ranked.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in re_ranked[:5]]

    # 5. Synthesize answer using Agno Agent
    # Instead of build_synthesizer, use an Agno Agent
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        knowledge=knowledge_base,
        search_knowledge=False,  # We're providing docs directly
        instructions=f"""
        Answer the user's question based on the provided context documents.
        Question: {prompt}
        
        Use the following documents as context:
        {format_documents_for_context(top_docs)}
        """
    )
    
    # Generate response
    response = agent.run(prompt)
    return {"answer": response.content}

def format_documents_for_context(docs):
    """Helper function to format documents for the agent context"""
    context = ""
    for i, doc in enumerate(docs, 1):
        context += f"\nDocument {i}:\n{doc.content}\n"
    return context