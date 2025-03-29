from typing import List, Tuple
from utils.raggpt.load_rag_config import LoadRAGConfig
from utils.ai_assistant.interact_with_gpt import interact_with_gpt_assistant
from utils.raggpt.perform_rag import PerformRAG
from utils.ai_assistant.load_ai_assistant_config import LoadAIAssistantConfig
from utils.service_calls import ServiceCall
from utils.webragquery.call_webragquery import WebRAGQuery
from utils.webragquery.load_wrq_config import LoadWRQConfig
from utils.web_servers.load_web_service_config import LoadWebServicesConfig
import os
import gradio as gr

WEB_SERVICES_CFG = LoadWebServicesConfig()
AICFG = LoadAIAssistantConfig()
RAGCFG = LoadRAGConfig()
WRQCFG = LoadWRQConfig()
URL = "https://github.com/Shreyojit/Complete-A-Z-Chatbot-Rag-StableDiff-OpenAI.git"
hyperlink = f"[RAG-GPT user guideline]({URL})"

class ChatBot:
    @staticmethod
    def respond(
            chatbot: List,
            user_input: dict,
            chatbot_functionality: str,
            gpt_temperature: float,
            llava_max_output_token: int,
            input_audio_block: Tuple,
            rag_top_k_retrieval: int,
            rag_search_type: str,
        ) -> Tuple:
        """
        Main respond method for the chatbot.
        """
        # --- Fix: robustly handle "files" list so we don't crash if an entry is a string. ---
        user_image_url = None
        for file_item in user_input.get("files", []):
            # Check whether file_item is a dict or a plain string:
            if isinstance(file_item, dict) and "path" in file_item:
                file_path = file_item["path"]
            else:
                file_path = file_item  # If it's a string, just use that.

            chatbot.append(((file_path,), None))
            user_image_url = file_path

        # If the user has submitted an audio, convert it to text and assign it to user "message"
        if input_audio_block:
            message = ServiceCall.speech_to_text(input_audio_block)
        elif user_input.get("text") is not None:
            message = user_input["text"]
        else:
            message = ""

        print("================")
        print(message)
        print("================")

        chat_history = f"Chat history:\n {str(chatbot[-RAGCFG.number_of_q_a_pairs:])}\n\n"

        # 1) GPT AI assistant
        if chatbot_functionality == "GPT AI assistant":
            question = "# User new question:\n" + message
            prompt = f"{chat_history}{question}"
            response = interact_with_gpt_assistant(
                prompt,
                llm_engine=AICFG.gpt_engine,
                temperature=gpt_temperature,
                llm_system_role=AICFG.gpt_system_role
            )
            response_content = response["choices"][0]["message"]["content"]
            chatbot.append((message, response_content))
            print(chatbot)
            return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), ""

        # 2) RAG-GPT: RAG with processed documents
        elif chatbot_functionality == "RAG-GPT: RAG with processed documents":
            if os.path.exists(RAGCFG.persist_directory):
                rag_instance = PerformRAG(
                    persist_directory=RAGCFG.persist_directory,
                    embedding_model=RAGCFG.embedding_model,
                    search_type=rag_search_type,
                    message=message,
                    k=rag_top_k_retrieval,
                    server_url=WEB_SERVICES_CFG.rag_reference_service_port,
                    chat_history=chat_history,
                    llm_engine=RAGCFG.llm_engine,
                    llm_system_role=RAGCFG.llm_system_role,
                    temperature=RAGCFG.temperature,
                    fetch_k=RAGCFG.fetch_k,
                    lambda_param=RAGCFG.lambda_param
                )
                response, retrieved_content = rag_instance.perform_rag()
                chatbot.append((message, response))
                return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), retrieved_content
            else:
                chatbot.append(
                    (message, f"VectorDB does not exist. Please first execute the 'upload_data_manually.py' module. "
                              f"For further information please visit {hyperlink}."))
                return "", chatbot, None

        # 3) RAG-GPT: RAG with upload documents
        elif chatbot_functionality == "RAG-GPT: RAG with upload documents":
            if os.path.exists(RAGCFG.custom_persist_directory):
                rag_instance = PerformRAG(
                    persist_directory=RAGCFG.custom_persist_directory,
                    embedding_model=RAGCFG.embedding_model,
                    search_type=rag_search_type,
                    message=message,
                    k=rag_top_k_retrieval,
                    server_url=WEB_SERVICES_CFG.rag_reference_service_port,
                    chat_history=chat_history,
                    llm_engine=RAGCFG.llm_engine,
                    llm_system_role=RAGCFG.llm_system_role,
                    temperature=RAGCFG.temperature,
                    fetch_k=RAGCFG.fetch_k,
                    lambda_param=RAGCFG.lambda_param
                )
                response, retrieved_content = rag_instance.perform_rag()
                chatbot.append((message, response))
                return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), retrieved_content
            else:
                chatbot.append(
                    (message, "No file was uploaded. Please first upload your files using the 'upload' button."))
                return "", chatbot, None

        # 4) WebRAGQuery: GPT + Duckduckgo search engine + Web RAG pipeline prep + Web Summarizer
        elif chatbot_functionality == "WebRAGQuery: GPT + Duckduckgo search engine + Web RAG pipeline prep + Web Summarizer":
            webragquery_instance = WebRAGQuery(
                chat_history=chat_history, message=message, wrgconfig=WRQCFG)
            response = webragquery_instance.call()
            chatbot.append((message, response))
            return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), ""

        # 5) WebRAGQuery: RAG with the requested website (GPT model)
        elif chatbot_functionality == "WebRAGQuery: RAG with the requested website (GPT model)":
            print("Here")
            response = ServiceCall.ask_rag_with_website_llm(
                wrq_config=WRQCFG,
                message=message,
                chat_history=chat_history,
                k=rag_top_k_retrieval,
                rag_search_type=rag_search_type
            )
            chatbot.append((message, response))
            return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), ""

        # 6) LLAVA AI assistant (Understands images)
        elif chatbot_functionality == "LLAVA AI assistant (Understands images)":
            response = ServiceCall.ask_llava(message, llava_max_output_token, user_image_url)
            response = ServiceCall.remove_inst(response)
            chatbot.append((message, response))
            return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), ""

        # 7) Generate image (stable-diffusion-xl-base-1.0)
        elif chatbot_functionality == "Generate image (stable-diffusion-xl-base-1.0)":
            image_dir = ServiceCall.ask_stable_diffusion(message)
            chatbot.append((message, None))
            chatbot.append((None, (image_dir,)))
            return chatbot, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"]), ""
