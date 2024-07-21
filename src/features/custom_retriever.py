# 5.custom_retriever.py

from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

class CustomRetriever(BaseRetriever):
    """
    Un recuperador personalizado que realiza búsquedas en un almacén de vectores,
    priorizando los documentos de resumen CSV y complementando con otros documentos relevantes.

    Atributos:
        vectorstore (Any): El almacén de vectores utilizado para las búsquedas de similitud.
    """

    vectorstore: Any = Field(default=None, description="Almacén de vectores para búsquedas de similitud")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Recupera documentos relevantes basados en la consulta proporcionada.

        Args:
            query (str): La consulta para la cual se buscan documentos relevantes.

        Returns:
            List[Document]: Una lista de documentos relevantes, con el resumen CSV (si se encuentra) al principio.
        """
        summary_docs = self.vectorstore.similarity_search(query, filter={"source": "CSV_summary"}, k=1)
        other_docs = self.vectorstore.similarity_search(query, k=4)
        return summary_docs + [doc for doc in other_docs if doc not in summary_docs]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Versión asíncrona de get_relevant_documents.

        Args:
            query (str): La consulta para la cual se buscan documentos relevantes.

        Returns:
            List[Document]: Una lista de documentos relevantes.
        """
        return self.get_relevant_documents(query)