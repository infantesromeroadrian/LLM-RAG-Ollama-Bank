# 1. document_loader.py

"""
document_loader.py

Este módulo proporciona funcionalidades para cargar documentos PDF y archivos CSV.
Utiliza PyMuPDFLoader para procesar PDFs y pandas para manejar archivos CSV.

Dependencias:
- os
- typing
- langchain_community.document_loaders
- langchain_core.documents
- pandas

Clases:
- DocumentLoader
"""

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

        Este método recorre el directorio dado, carga cada archivo PDF encontrado
        utilizando PyMuPDFLoader, y devuelve una lista de objetos Document.

        Args:
            directory (str): Ruta al directorio que contiene los archivos PDF.

        Returns:
            List[Document]: Una lista de objetos Document, cada uno representando
                            el contenido de un archivo PDF.

        Nota:
            Los archivos que no terminan en '.pdf' son ignorados.
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

        Nota:
            Este método utiliza la función read_csv de pandas con sus configuraciones
            por defecto. Para casos de uso específicos, puede ser necesario ajustar
            los parámetros de lectura del CSV.
        """
        return pd.read_csv(file_path)

# Ejemplo de uso:
# pdf_docs = DocumentLoader.load_pdfs("/path/to/pdf/directory")
# csv_data = DocumentLoader.load_csv("/path/to/data.csv")