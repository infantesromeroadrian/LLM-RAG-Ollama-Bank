# 5.custom_retriever.py

from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field


class CustomRetriever(BaseRetriever):
    vectorstore: Any = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        summary_docs = self.vectorstore.similarity_search(query, filter={"source": "CSV_summary"}, k=1)
        other_docs = self.vectorstore.similarity_search(query, k=4)
        return summary_docs + [doc for doc in other_docs if doc not in summary_docs]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)