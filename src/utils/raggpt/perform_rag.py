# src/utils/raggpt/perform_rag.py
import os
from langchain_community.vectorstores import Chroma
from typing import List, Tuple
import re
import ast
import html
import openai

class PerformRAG:
    def __init__(self,
                 persist_directory: str,
                 embedding_model,
                 search_type: str,
                 message: str,
                 k: int,
                 server_url: str,
                 chat_history: str,
                 llm_engine: str,
                 llm_system_role: str,
                 temperature: float,
                 fetch_k: int,
                 lambda_param: float) -> None:
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.search_type = search_type
        self.message = message
        self.k = k
        self.server_url = server_url
        self.chat_history = chat_history
        self.llm_engine = llm_engine
        self.llm_system_role = llm_system_role
        self.temperature = temperature
        self.fetch_k = fetch_k
        self.lambda_param = lambda_param

    def perform_rag(self) -> Tuple:
        vectordb = self._load_vectordb()
        retrieved_content = self._search_vectordb(vectordb=vectordb)
        response = self._ask_gpt(retrieved_content=retrieved_content)
        return response, retrieved_content

    def _load_vectordb(self):
        vectordb = Chroma(persist_directory=self.persist_directory,
                          embedding_function=self.embedding_model)
        return vectordb

    def _search_vectordb(self, vectordb) -> str:
        if self.search_type == "Similarity search":
            docs = vectordb.similarity_search(self.message, k=self.k)
        elif self.search_type == "mmr":
            docs = vectordb.max_marginal_relevance_search(
                self.message, k=self.k, fetch_k=self.fetch_k, lambda_param=self.lambda_param)
        print(docs)
        retrieved_content = self.clean_references(docs)
        return retrieved_content

    def _ask_gpt(self, retrieved_content):
        question = "# User new question:\n" + self.message
        prompt = f"{self.chat_history}{retrieved_content}{question}"
        response = openai.ChatCompletion.create(
            model=self.llm_engine,
            messages=[
                {"role": "system", "content": self.llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]["content"]

    def clean_references(self, documents: List, server_url: str = "http://localhost:8000") -> str:
        documents = [str(x) + "\n\n" for x in documents]
        markdown_documents = ""
        counter = 1
        for doc in documents:
            match = re.search(r"page_content=(.*?)( metadata=\{.*\})", doc)
            if match:
                content, metadata = match.groups()
                metadata = metadata.split('=', 1)[1]
                metadata_dict = ast.literal_eval(metadata)
                content = bytes(content, "utf-8").decode("unicode_escape")
                content = re.sub(r'\\n', '\n', content)
                content = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                content = html.unescape(content)
                content = content.encode('latin1').decode('utf-8', 'ignore')
                content = re.sub(r'â', '-', content)
                content = re.sub(r'â', '∈', content)
                content = re.sub(r'Ã', '×', content)
                content = re.sub(r'ï¬', 'fi', content)
                content = re.sub(r'â', '∈', content)
                content = re.sub(r'Â·', '·', content)
                content = re.sub(r'ï¬', 'fl', content)
                pdf_url = f"{server_url}/{os.path.basename(metadata_dict['source'])}"
                markdown_documents += f"# Retrieved content {counter}:\n" + content + "\n\n" + \
                    f"Source: {os.path.basename(metadata_dict['source'])}" + " | " +\
                    f"Page number: {str(metadata_dict['page'])}" + " | " +\
                    f"[View PDF]({pdf_url})" "\n\n"
                counter += 1
            else:
                markdown_documents += doc + "\n\n"
        return markdown_documents
