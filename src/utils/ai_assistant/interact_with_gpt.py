# src/utils/ai_assistant/interact_with_gpt.py
import openai
import os
from typing import Dict

def interact_with_gpt_assistant(prompt: str, llm_engine: str, temperature: float, llm_system_role: str) -> Dict:
    """
    Interact with GPT-based chatbot assistant.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=llm_engine,
        messages=[
            {"role": "system", "content": llm_system_role},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return response

if __name__ == "__main__":
    prompt = "Hello, how are you?"
    llm_engine = "gpt-3.5-turbo"
    temperature = 0.7
    llm_system_role = "You are a helpful assistant."
    response = interact_with_gpt_assistant(prompt, llm_engine, temperature, llm_system_role)
    print(response)
