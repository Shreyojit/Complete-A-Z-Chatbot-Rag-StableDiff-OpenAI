# src/utils/webragquery/llm_rag.py
from typing import List, Dict
import openai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import re

class LLM_RAG:
    def __init__(self, persist_directory: str, user_query: str, k: int, rag_search_type: str, lambda_param: float, fetch_k: int, input_chat_history: str, llm_system_role: str, gpt_model: str, temperature: float):
        self.persist_directory = persist_directory
        self.user_query = user_query
        self.k = k
        self.rag_search_type = rag_search_type
        self.lambda_param = lambda_param
        self.fetch_k = fetch_k
        self.input_chat_history = input_chat_history
        self.llm_system_role = llm_system_role
        self.gpt_model = gpt_model
        self.temperature = temperature

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.strip()

    def search_vectordb(self) -> str:
        embedding_model = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=self.persist_directory,
                          embedding_function=embedding_model)
        print("============")
        print(self.k)
        print(self.user_query)
        print("============")
        if self.rag_search_type == "Similarity search":
            docs = vectordb.similarity_search(self.user_query, k=self.k)
        elif self.rag_search_type == "mmr":
            docs = vectordb.max_marginal_relevance_search(
                self.user_query, k=self.k, fetch_k=self.fetch_k, lambda_param=self.lambda_param)
        retrieved_docs_page_content = [str(x.page_content) + "\n\n" for x in docs]
        print(retrieved_docs_page_content)
        print("============")
        retrieved_docs = self.clean_text(retrieved_docs_page_content[0]) if retrieved_docs_page_content else ""
        print(retrieved_docs)
        documents = "# Retrieved content:\n" + str(retrieved_docs)
        print("============")
        print(documents)
        print("============")
        return documents

    def prepare_messages(self) -> List[Dict]:
        retrieved_docs_page_content = self.search_vectordb()
        query = f"{self.input_chat_history}\n\n# User's new query: {self.user_query}\n\n# vector search result on the url:\n\n{retrieved_docs_page_content}"
        messages = [
            {"role": "system", "content": self.llm_system_role},
            {"role": "user", "content": self.input_chat_history + query}
        ]
        return messages

    def ask(self) -> Dict:
        messages = self.prepare_messages()
        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=messages,
            temperature=self.temperature
        )
        return response
