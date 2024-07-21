# 1. document_loader.py

import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import pandas as pd

class DocumentLoader:
    """
    Clase estática para cargar documentos PDF y archivos CSV.

    Esta clase proporciona métodos estáticos para cargar múltiples documentos PDF
    de un directorio y para cargar un archivo CSV como un DataFrame de pandas.
    """

    @staticmethod
    def load_pdfs(directory: str) -> List[Document]:
        """
        Carga todos los archivos PDF de un directorio especificado.

        Args:
            directory (str): Ruta al directorio que contiene los archivos PDF.

        Returns:
            List[Document]: Una lista de objetos Document, cada uno representando
                            el contenido de un archivo PDF.
        """
        pdf_docs = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                loader = PyMuPDFLoader(os.path.join(directory, filename))
                pdf_docs.extend(loader.load())
        return pdf_docs

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Carga un archivo CSV y lo devuelve como un DataFrame de pandas.

        Args:
            file_path (str): Ruta al archivo CSV a cargar.

        Returns:
            pd.DataFrame: Un DataFrame de pandas que contiene los datos del CSV.
        """
        return pd.read_csv(file_path)