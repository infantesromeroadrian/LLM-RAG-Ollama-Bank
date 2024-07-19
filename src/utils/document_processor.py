# 2. data_processor.py

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd

class DataProcessor:
    @staticmethod
    def split_documents(docs: List[Document], chunk_size: int = 2000, chunk_overlap: int = 500) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(docs)

    @staticmethod
    def create_csv_summary(df: pd.DataFrame) -> Document:
        csv_summary = f"""
        RESUMEN IMPORTANTE DEL CSV 'BankCustomerChurnPrediction.csv':
        - Total de filas y clientes únicos: {len(df)}
        - Número exacto de clientes únicos: {df['customer_id'].nunique()}
        - Columnas: {', '.join(df.columns)}
        - Rango de edades: {df['age'].min()} - {df['age'].max()} años
        - Países representados: {', '.join(df['country'].unique())}
        - Saldo promedio: {df['balance'].mean():.2f}
        - Porcentaje de clientes con tarjeta de crédito: {(df['credit_card'].sum() / len(df) * 100):.2f}%
        - Tasa de abandono (churn): {(df['churn'].sum() / len(df) * 100):.2f}%

        Esta información es un resumen preciso basado en el análisis del archivo CSV completo.
        Para preguntas sobre estadísticas generales o totales, utiliza siempre esta información.
        """
        return Document(page_content=csv_summary, metadata={"source": "CSV_summary", "importance": 10})

    @staticmethod
    def create_csv_docs(df: pd.DataFrame, sample_size: int = 1000) -> List[Document]:
        csv_sample = df.sample(n=sample_size, random_state=42)
        return [Document(page_content=row.to_json(), metadata={"source": "CSV_record", "importance": 1})
                for _, row in csv_sample.iterrows()]
