# 2. data_processor.py

"""
data_processor.py

Este módulo proporciona funcionalidades para procesar documentos y datos CSV.
Incluye métodos para dividir documentos, crear resúmenes de CSV y generar documentos
a partir de registros CSV.

Dependencias:
- typing
- langchain.text_splitter
- langchain_core.documents
- pandas

Clases:
- DataProcessor
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd

class DataProcessor:
    """
    Clase estática para procesar documentos y datos CSV.

    Proporciona métodos para dividir documentos, crear resúmenes de CSV
    y generar documentos a partir de registros CSV.
    """

    @staticmethod
    def split_documents(docs: List[Document], chunk_size: int = 2000, chunk_overlap: int = 500) -> List[Document]:
        """
        Divide una lista de documentos en chunks más pequeños.

        Args:
            docs (List[Document]): Lista de documentos a dividir.
            chunk_size (int, optional): Tamaño de cada chunk. Por defecto 2000.
            chunk_overlap (int, optional): Superposición entre chunks. Por defecto 500.

        Returns:
            List[Document]: Lista de documentos divididos.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(docs)

    @staticmethod
    def create_csv_summary(df: pd.DataFrame) -> Document:
        """
        Crea un resumen del DataFrame CSV.

        Args:
            df (pd.DataFrame): DataFrame de pandas con los datos CSV.

        Returns:
            Document: Un objeto Document que contiene el resumen del CSV.

        Nota:
            El resumen incluye estadísticas clave como el número total de filas,
            clientes únicos, rango de edades, países representados, etc.
        """
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
        """
        Crea una lista de documentos a partir de una muestra del DataFrame CSV.

        Args:
            df (pd.DataFrame): DataFrame de pandas con los datos CSV.
            sample_size (int, optional): Tamaño de la muestra a tomar. Por defecto 1000.

        Returns:
            List[Document]: Lista de objetos Document, cada uno representando una fila del CSV.

        Nota:
            Utiliza un muestreo aleatorio con una semilla fija para reproducibilidad.
        """
        csv_sample = df.sample(n=sample_size, random_state=42)
        return [Document(page_content=row.to_json(), metadata={"source": "CSV_record", "importance": 1})
                for _, row in csv_sample.iterrows()]

# Ejemplo de uso:
# docs = [Document(...), Document(...)]
# split_docs = DataProcessor.split_documents(docs)
#
# df = pd.read_csv('BankCustomerChurnPrediction.csv')
# csv_summary = DataProcessor.create_csv_summary(df)
# csv_docs = DataProcessor.create_csv_docs(df)