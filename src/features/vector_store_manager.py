# 4. vector_store_manager.py

"""
vector_store_manager.py

Este módulo proporciona una clase para gestionar la creación y persistencia de un almacén
de vectores utilizando Chroma de LangChain.

Dependencias:
- os
- langchain_community.vectorstores

Clases:
- VectorStoreManager
"""

import os
from langchain_community.vectorstores import Chroma

class VectorStoreManager:
    """
    Gestor para la creación y manejo de un almacén de vectores Chroma.

    Esta clase facilita la creación de un almacén de vectores persistente
    utilizando documentos y un modelo de embeddings proporcionados.

    Atributos:
        embed_model: El modelo de embeddings a utilizar para vectorizar los documentos.
        base_dir (str): El directorio base donde se almacenará el almacén de vectores.
    """

    def __init__(self, embed_model, base_dir):
        """
        Inicializa el VectorStoreManager.

        Args:
            embed_model: El modelo de embeddings a utilizar.
            base_dir (str): El directorio base para almacenar los datos del almacén de vectores.
        """
        self.embed_model = embed_model
        self.base_dir = base_dir

    def create_vector_store(self, documents):
        """
        Crea y persiste un almacén de vectores Chroma a partir de los documentos proporcionados.

        Este método vectoriza los documentos utilizando el modelo de embeddings especificado
        y los almacena en un directorio persistente.

        Args:
            documents (List[Document]): Lista de documentos a vectorizar y almacenar.

        Returns:
            Chroma: Una instancia del almacén de vectores Chroma creado.

        Nota:
            El almacén de vectores se guarda en un subdirectorio 'bank_data_db' dentro del directorio base.
            La colección se nombra 'bank_regulations_and_data'.
        """
        persist_directory = os.path.join(self.base_dir, "bank_data_db")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embed_model,
            persist_directory=persist_directory,
            collection_name="bank_regulations_and_data"
        )

# Ejemplo de uso:
# embed_model = SomeEmbeddingModel()
# base_dir = "/path/to/base/directory"
# vector_store_manager = VectorStoreManager(embed_model, base_dir)
# documents = [Document(...), Document(...), ...]
# vector_store = vector_store_manager.create_vector_store(documents)