from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIChat
# ——— FastAPI setup ———
app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

# ——— Cached query function ———

# ——— API endpoint ———
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    try:
        # run the caching query in a threadpool
        agent = Agent(
            model=OpenAIChat(
                id="gpt-4o-mini",
                cache_system_prompt=True,
            ),
        )
        response = agent.run(prompt)
        return {
            "status": "success",
            "prompt": prompt,
            "results": response.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


