from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# ——— FastAPI setup ———
app = FastAPI()

class SynonymExpansionRequest(BaseModel):
    query: str

# ——— Constitution Laws Database (Sample) ———
CONSTITUTION_LAWS = """
Article 14: Right to Equality - The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.

Article 19: Protection of certain rights regarding freedom of speech, etc. - All citizens shall have the right to freedom of speech and expression; to assemble peaceably and without arms; to form associations or unions; to move freely throughout the territory of India; to reside and settle in any part of the territory of India; and to practice any profession, or to carry on any occupation, trade or business.

Article 21: Protection of life and personal liberty - No person shall be deprived of his life or personal liberty except according to procedure established by law.

Article 32: Remedies for enforcement of rights conferred by this Part - The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed.

Article 44: Uniform civil code for the citizens - The State shall endeavour to secure for the citizens a uniform civil code throughout the territory of India.

Article 51A: Fundamental Duties - It shall be the duty of every citizen of India to abide by the Constitution and respect its ideals and institutions, the National Flag and the National Anthem; to cherish and follow the noble ideals which inspired our national struggle for freedom; to uphold and protect the sovereignty, unity and integrity of India.

Article 356: Provisions in case of failure of constitutional machinery in States - If the President, on receipt of a report from the Governor of a State or otherwise, is satisfied that a situation has arisen in which the government of the State cannot be carried on in accordance with the provisions of this Constitution, the President may by Proclamation assume to himself all or any of the functions of the Government of the State.
"""

# ——— API endpoint ———
@app.post("/synonym-expansion")
async def synonym_expansion_endpoint(req: SynonymExpansionRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty")

    try:
        # Create an AI Agent with cached system prompt for constitution laws
        agent = Agent(
            model=OpenAIChat(
                id="gpt-4o-mini",
                cache_system_prompt=True,
            ),
            instructions=[
                "You are a Constitutional Law Expert AI Agent specialized in synonym expansion and precise legal interpretation.",
                "Your task is to:",
                "1. Analyze the user's query and identify synonyms, related terms, and legal concepts",
                "2. Expand the query to include relevant constitutional articles and provisions",
                "3. Return the exact text of relevant constitutional laws in precise words",
                "4. Provide context on how the synonyms relate to the constitutional provisions",
                "",
                "Constitution Laws Database:",
                CONSTITUTION_LAWS,
                "",
                "Instructions:",
                "- Quote constitutional articles verbatim (exact words from the database)",
                "- Identify synonym variations (e.g., 'freedom' → 'liberty', 'equality' → 'equal protection')",
                "- Explain the legal significance of synonym expansion",
                "- Be precise and authoritative in your response",
            ],
        )
        
        # Run the agent with the user's query
        response = agent.run(
            f"Query: {query}\n\nExpand this query with synonyms and return relevant constitutional laws in precise words."
        )
        
        return {
            "status": "success",
            "query": query,
            "expanded_results": response.content,
            "agent_type": "Synonym Expansion - Constitutional Law Expert"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
