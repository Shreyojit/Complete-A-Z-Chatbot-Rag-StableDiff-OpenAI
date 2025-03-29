# src/utils/raggpt/load_rag_config.py
import yaml
from langchain_community.embeddings import OpenAIEmbeddings
from pyprojroot import here
from utils.app_utils import Apputils

class LoadRAGConfig:
    def __init__(self) -> None:
        with open(here("configs/rag_gpt.yml")) as cfg:
            rag_config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.llm_engine = rag_config["llm_config"]["engine"]
        self.llm_system_role = rag_config["llm_config"]["llm_system_role"]
        self.persist_directory = str(rag_config["directories"]["persist_directory"])
        self.custom_persist_directory = str(rag_config["directories"]["custom_persist_directory"])
        self.embedding_model = OpenAIEmbeddings()
        self.data_directory = rag_config["directories"]["data_directory"]
        self.k = rag_config["retrieval_config"]["k"]
        self.embedding_model_engine = rag_config["embedding_model_config"]["engine"]
        self.chunk_size = rag_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = rag_config["splitter_config"]["chunk_overlap"]
        self.max_final_token = rag_config["summarizer_config"]["max_final_token"]
        self.token_threshold = rag_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = rag_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap = rag_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role = rag_config["summarizer_config"]["final_summarizer_llm_system_role"]
        self.temperature = rag_config["llm_config"]["temperature"]
        self.fetch_k = rag_config["mmr_search_config"]["fetch_k"]
        self.lambda_param = rag_config["mmr_search_config"]["lambda_param"]
        self.number_of_q_a_pairs = rag_config["memory"]["number_of_q_a_pairs"]
        Apputils.create_directory(self.persist_directory)
        Apputils.remove_directory(self.custom_persist_directory)
