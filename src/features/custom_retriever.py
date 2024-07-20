# 5.custom_retriever.py

"""
custom_retriever.py

Este módulo define un recuperador personalizado (CustomRetriever) que extiende la funcionalidad
del BaseRetriever de LangChain para realizar búsquedas personalizadas en un almacén de vectores.

Dependencias:
- langchain_core.retrievers
- langchain_core.documents
- pydantic

Clases:
- CustomRetriever
"""

from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field


class CustomRetriever(BaseRetriever):
    """
    Un recuperador personalizado que realiza búsquedas en un almacén de vectores,
    priorizando los documentos de resumen CSV y complementando con otros documentos relevantes.

    Atributos:
        vectorstore (Any): El almacén de vectores utilizado para las búsquedas de similitud.

    Configuración:
        arbitrary_types_allowed (bool): Permite tipos arbitrarios en la configuración de Pydantic.
    """

    vectorstore: Any = Field(default=None, description="Almacén de vectores para búsquedas de similitud")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Recupera documentos relevantes basados en la consulta proporcionada.

        Este método realiza dos búsquedas:
        1. Busca un documento de resumen CSV.
        2. Busca otros documentos relevantes.

        Luego combina los resultados, asegurándose de que no haya duplicados.

        Args:
            query (str): La consulta para la cual se buscan documentos relevantes.

        Returns:
            List[Document]: Una lista de documentos relevantes, con el resumen CSV (si se encuentra) al principio.
        """
        # Busca el documento de resumen CSV
        summary_docs = self.vectorstore.similarity_search(query, filter={"source": "CSV_summary"}, k=1)

        # Busca otros documentos relevantes
        other_docs = self.vectorstore.similarity_search(query, k=4)

        # Combina los resultados, eliminando duplicados
        return summary_docs + [doc for doc in other_docs if doc not in summary_docs]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Versión asíncrona de get_relevant_documents.

        Esta implementación simplemente llama a la versión síncrona, pero podría
        ser modificada en el futuro para realizar operaciones asíncronas si es necesario.

        Args:
            query (str): La consulta para la cual se buscan documentos relevantes.

        Returns:
            List[Document]: Una lista de documentos relevantes.
        """
        return self.get_relevant_documents(query)

# Ejemplo de uso:
# retriever = CustomRetriever(vectorstore=my_vector_store)
# relevant_docs = retriever.get_relevant_documents("¿Cuál es el saldo promedio de los clientes?")