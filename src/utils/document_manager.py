# 3.document_manager.py

from src.utils.document_loader import DocumentLoader
from src.utils.document_processor import DataProcessor


class DocumentManager:
    def __init__(self, pdf_directory: str, csv_file: str):
        self.pdf_directory = pdf_directory
        self.csv_file = csv_file

    def load_and_process_documents(self):
        pdf_docs = DocumentLoader.load_pdfs(self.pdf_directory)
        df = DocumentLoader.load_csv(self.csv_file)

        split_docs = DataProcessor.split_documents(pdf_docs)
        csv_summary_doc = DataProcessor.create_csv_summary(df)
        csv_docs = DataProcessor.create_csv_docs(df)

        return [csv_summary_doc] + csv_docs + split_docs
