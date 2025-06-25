from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Agent settings that can be set using environment variables.

    Reference: https://pydantic-docs.helpmanual.io/usage/settings/
    """

    gpt_4: str = "gpt-4.1"
    gpt_o3: str = "o4-mini"
    gpt_4o_mini: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    default_max_completion_tokens: int = 16000
    default_temperature: float = 0.4
    openai_api_key: str

    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"  # You can override with .env if needed
        extra = "ignore"  
        
# Create an AgentSettings object
agent_settings = AgentSettings()