# 4. vector_store_manager.py

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

        Args:
            documents (List[Document]): Lista de documentos a vectorizar y almacenar.

        Returns:
            Chroma: Una instancia del almacén de vectores Chroma creado.
        """
        persist_directory = os.path.join(self.base_dir, "bank_data_db")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embed_model,
            persist_directory=persist_directory,
            collection_name="bank_regulations_and_data"
        )