# src/utils/ai_assistant/load_ai_assistant_config.py
import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import openai

load_dotenv()

class LoadAIAssistantConfig:
    def __init__(self) -> None:
        with open(here("configs/ai_assistant.yml")) as cfg:
            app_config = yaml.safe_load(cfg)
        self.gpt_engine = app_config["gpt_config"]["gpt_engine"]
        self.gpt_system_role = app_config["gpt_config"]["gpt_system_role"]
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai
