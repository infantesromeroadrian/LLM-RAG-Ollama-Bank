# 1. document_loader.py

import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import pandas as pd

class DocumentLoader:
    @staticmethod
    def load_pdfs(directory: str) -> List[Document]:
        pdf_docs = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                loader = PyMuPDFLoader(os.path.join(directory, filename))
                pdf_docs.extend(loader.load())
        return pdf_docs

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
