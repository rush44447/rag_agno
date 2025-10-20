from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from typing import List, Optional

# ——— FastAPI setup ———
app = FastAPI()

class QueryExpansionRequest(BaseModel):
    query: str
    max_expansions: Optional[int] = 5

class ExpansionResult(BaseModel):
    original_query: str
    expanded_queries: List[str]
    relevant_laws: str
    explanation: str

# ——— Constitution Laws Database (Comprehensive) ———
CONSTITUTION_LAWS = """
Article 14: Right to Equality - The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.

Article 15: Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth - The State shall not discriminate against any citizen on grounds only of religion, race, caste, sex, place of birth or any of them.

Article 16: Equality of opportunity in matters of public employment - There shall be equality of opportunity for all citizens in matters relating to employment or appointment to any office under the State.

Article 19: Protection of certain rights regarding freedom of speech, etc. - All citizens shall have the right to freedom of speech and expression; to assemble peaceably and without arms; to form associations or unions; to move freely throughout the territory of India; to reside and settle in any part of the territory of India; and to practice any profession, or to carry on any occupation, trade or business.

Article 20: Protection in respect of conviction for offences - No person shall be convicted of any offence except for violation of a law in force at the time of the commission of the act charged as an offence, nor be subjected to a penalty greater than that which might have been inflicted under the law in force at the time of the commission of the offence.

Article 21: Protection of life and personal liberty - No person shall be deprived of his life or personal liberty except according to procedure established by law.

Article 21A: Right to education - The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine.

Article 22: Protection against arrest and detention in certain cases - No person who is arrested shall be detained in custody without being informed, as soon as may be, of the grounds for such arrest nor shall he be denied the right to consult, and to be defended by, a legal practitioner of his choice.

Article 25: Freedom of conscience and free profession, practice and propagation of religion - Subject to public order, morality and health and to the other provisions of this Part, all persons are equally entitled to freedom of conscience and the right freely to profess, practice and propagate religion.

Article 32: Remedies for enforcement of rights conferred by this Part - The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed.

Article 44: Uniform civil code for the citizens - The State shall endeavour to secure for the citizens a uniform civil code throughout the territory of India.

Article 51A: Fundamental Duties - It shall be the duty of every citizen of India to abide by the Constitution and respect its ideals and institutions, the National Flag and the National Anthem; to cherish and follow the noble ideals which inspired our national struggle for freedom; to uphold and protect the sovereignty, unity and integrity of India; to defend the country and render national service when called upon to do so; to promote harmony and the spirit of common brotherhood amongst all the people of India; to value and preserve the rich heritage of our composite culture; to protect and improve the natural environment including forests, lakes, rivers and wild life, and to have compassion for living creatures; to develop the scientific temper, humanism and the spirit of inquiry and reform; to safeguard public property and to abjure violence; to strive towards excellence in all spheres of individual and collective activity; and to provide opportunities for education to his child or ward between the age of six and fourteen years.

Article 226: Power of High Courts to issue certain writs - Every High Court shall have power, throughout the territories in relation to which it exercises jurisdiction, to issue to any person or authority, including in appropriate cases, any Government, within those territories directions, orders or writs, including writs in the nature of habeas corpus, mandamus, prohibitions, quo warranto and certiorari, or any of them, for the enforcement of any of the rights conferred by Part III and for any other purpose.

Article 356: Provisions in case of failure of constitutional machinery in States - If the President, on receipt of a report from the Governor of a State or otherwise, is satisfied that a situation has arisen in which the government of the State cannot be carried on in accordance with the provisions of this Constitution, the President may by Proclamation assume to himself all or any of the functions of the Government of the State.

Article 370: Temporary provisions with respect to the State of Jammu and Kashmir - [Historical provision regarding special status]
"""

# ——— API endpoint ———
@app.post("/query-expansion", response_model=dict)
async def query_expansion_endpoint(req: QueryExpansionRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty")

    try:
        # Create an AI Agent with cached system prompt for query expansion
        agent = Agent(
            model=OpenAIChat(
                id="gpt-4o-mini",
                cache_system_prompt=True,
            ),
            instructions=[
                "You are a Query Expansion AI Agent specialized in constitutional law research.",
                "Your task is to:",
                "1. Analyze the user's query and expand it with related legal concepts, terms, and variations",
                f"2. Generate up to {req.max_expansions} expanded query variations that capture different aspects of the original query",
                "3. Identify and return ALL relevant constitutional articles in their EXACT, PRECISE wording",
                "4. Explain how each expansion relates to constitutional provisions",
                "5. Provide comprehensive coverage of related legal concepts",
                "",
                "Query Expansion Techniques:",
                "- Semantic expansion (e.g., 'privacy' → 'personal liberty', 'right to be left alone')",
                "- Legal terminology (e.g., 'fairness' → 'natural justice', 'due process')",
                "- Related concepts (e.g., 'speech' → 'expression', 'press freedom', 'media rights')",
                "- Broader/narrower terms (e.g., 'rights' → 'fundamental rights', 'civil liberties')",
                "- Contextual variations (e.g., 'arrest' → 'detention', 'custody', 'preventive detention')",
                "",
                "Constitution Laws Database:",
                CONSTITUTION_LAWS,
                "",
                "Output Format:",
                "1. List the expanded queries (numbered)",
                "2. Quote ALL relevant constitutional articles VERBATIM",
                "3. Provide a brief explanation of the expansion strategy",
                "",
                "Be thorough, precise, and authoritative. Always quote constitutional text exactly.",
            ],
        )
        
        # Run the agent with the user's query
        response = agent.run(
            f"Original Query: {query}\n\nExpand this query and return all relevant constitutional laws in precise words."
        )
        
        return {
            "status": "success",
            "original_query": query,
            "max_expansions": req.max_expansions,
            "expansion_results": response.content,
            "agent_type": "Query Expansion - Constitutional Law Research"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Query Expansion AI Agent"}
