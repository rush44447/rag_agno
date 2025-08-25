import logging
from typing import Optional

from agno.models.openai import OpenAIChat
from agno.agent import Agent
from agno.team import Team
from agno.knowledge.website import WebsiteKnowledgeBase

from agents.settings import agent_settings
from db.ag_node import get_vector_db

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# ---------------- Logging ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Router ----------------- #
health_check_router = APIRouter()


# ---------------- Request Schema --------- #
class CodeRequest(BaseModel):
    question: str


# ---------------- Agents ---------------- #
def build_clarifier() -> Agent:
    return Agent(
        name="ClarifierAgent",
        role="Confirm defaults or ask only if truly missing",
        model=OpenAIChat(
            id=agent_settings.gpt_4o_mini,
            api_key=agent_settings.openai_api_key,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        instructions=[
            "If the user's question contains the term 'Agno,' assume they want Agno-based code and do NOT ask any follow-up.",
            "Otherwise, ask exactly one clarifying question if they haven’t named a language or framework."
        ],
    )


def build_requirements() -> Agent:
    return Agent(
        name="RequirementsAgent",
        role="List specific code-example needs from prompt",
        model=OpenAIChat(
            id=agent_settings.gpt_4o_mini,
            api_key=agent_settings.openai_api_key,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        instructions=[
            "Return up to three bullet points of the code examples we must retrieve "
            "(e.g., image upload, collection insert, comparison)."
        ],
    )


async def build_rag(collection: Optional[str] = "website") -> Agent:
    kb = WebsiteKnowledgeBase(
        urls=[],
        max_links=2,
        vector_db=await get_vector_db(collection),
    )
    return Agent(
        name="RAGAgent",
        role="Fetch code snippets for specified topics",
        knowledge=kb,
        search_knowledge=True,
        model=OpenAIChat(
            id=agent_settings.gpt_4o_mini,
            api_key=agent_settings.openai_api_key,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        instructions=[
            "Given a bullet list of topics, pull verbatim code blocks from the KB."
        ],
    )


def build_synthesizer() -> Agent:
    return Agent(
        name="SynthesizerAgent",
        role="Combine clarifier answers & retrieved snippets into final code",
        model=OpenAIChat(
            id=agent_settings.gpt_4o_mini,
            api_key=agent_settings.openai_api_key,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
        instructions=[
            "Produce one cohesive, runnable Python code snippet using Agno that satisfies the clarified requirements."
        ],
    )


# ---------------- Team Factory ----------- #
async def get_coding_team(library: str, collection: str) -> Team:
    team = Team(
        name="CodingTeam",
        team_id="coding-team",
        mode="route",
        members=[
            build_clarifier(),
            build_requirements(),
            await build_rag(collection),
            build_synthesizer(),
        ],
        instructions=[
            "You are a team of coding experts: confirm defaults → retrieve Agno examples → synthesize final code."
        ],
        model=OpenAIChat(
            id=agent_settings.gpt_4o_mini,
            api_key=agent_settings.openai_api_key,
            max_tokens=agent_settings.default_max_completion_tokens,
            temperature=agent_settings.default_temperature,
        ),
    )
    logger.info("Initialized CodingTeam with members: %s", [m.name for m in team.members])
    return team


# ---------------- Endpoint --------------- #
@health_check_router.post("/code")
async def generate_code(req: CodeRequest):
    try:
        if "agno" in req.question.lower():
            # Only RAG + Synthesizer
            rag = await build_rag("website")
            synth = build_synthesizer()
            team = Team(
                name="AgnoCodeTeam",
                team_id="agno-code-team",
                mode="route",
                members=[rag, synth],
                model=OpenAIChat(
                    id=agent_settings.gpt_4o_mini,
                    api_key=agent_settings.openai_api_key,
                    max_tokens=agent_settings.default_max_completion_tokens,
                    temperature=agent_settings.default_temperature,
                ),
            )
        else:
            # Full pipeline: clarify + requirements + rag + synth
            team = await get_coding_team("agno", "website")

        # Run the selected team
        result = await run_in_threadpool(team.run, req.question)
        return {
            "status": "success",
            "code": result.content,
        }
    except Exception as e:
        logger.error(f"Error in generate_code: {e}")
        return {
            "status": "error",
            "message": str(e),
        }
