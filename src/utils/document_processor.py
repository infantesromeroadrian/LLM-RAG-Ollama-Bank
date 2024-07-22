# 2. data_processor.py

# document_processor.py

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
import json
import numpy as np


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
        Crea un resumen detallado del DataFrame CSV.

        Args:
            df (pd.DataFrame): DataFrame de pandas con los datos CSV.

        Returns:
            Document: Un objeto Document que contiene el resumen detallado del CSV.
        """
        # Identificar columnas numéricas
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        def serialize_numpy(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        csv_summary = {
            "general_info": {
                "total_rows": int(len(df)),
                "unique_customers": int(df['customer_id'].nunique()),
                "columns": df.columns.tolist()
            },
            "numerical_stats": {col: {
                "min": serialize_numpy(df[col].min()),
                "max": serialize_numpy(df[col].max()),
                "mean": serialize_numpy(df[col].mean()),
                "median": serialize_numpy(df[col].median())
            } for col in numeric_columns},
            "categorical_stats": {col: df[col].value_counts().to_dict()
                                  for col in df.select_dtypes(include=['object']).columns},
            "correlations": df[numeric_columns].corr().to_dict()
        }

        summary_text = f"""
        RESUMEN DETALLADO DEL CSV 'BankCustomerChurnPrediction.csv':

        Información General:
        - Total de filas: {csv_summary['general_info']['total_rows']}
        - Clientes únicos: {csv_summary['general_info']['unique_customers']}
        - Columnas: {', '.join(csv_summary['general_info']['columns'])}

        Estadísticas Numéricas:
        {json.dumps(csv_summary['numerical_stats'], indent=2, default=serialize_numpy)}

        Estadísticas Categóricas:
        {json.dumps(csv_summary['categorical_stats'], indent=2, default=serialize_numpy)}

        Correlaciones:
        {json.dumps(csv_summary['correlations'], indent=2, default=serialize_numpy)}

        Este resumen proporciona una visión general detallada del dataset. 
        Para consultas específicas sobre clientes individuales, utiliza la función de búsqueda por customer_id.
        """

        return Document(page_content=summary_text, metadata={"source": "CSV_summary", "importance": 10,
                                                             "full_summary": json.dumps(csv_summary,
                                                                                        default=serialize_numpy)})

    @staticmethod
    def create_csv_docs(df: pd.DataFrame, sample_size: int = None) -> List[Document]:
        """
        Crea una lista de documentos a partir del DataFrame CSV completo o una muestra.

        Args:
            df (pd.DataFrame): DataFrame de pandas con los datos CSV.
            sample_size (int, optional): Tamaño de la muestra a tomar. Si es None, se usan todos los registros.

        Returns:
            List[Document]: Lista de objetos Document, cada uno representando una fila del CSV.
        """
        if sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df

        return [Document(page_content=row.to_json(default_handler=str),
                         metadata={"source": "CSV_record", "importance": 1, "customer_id": row['customer_id']})
                for _, row in df_sample.iterrows()]

    @staticmethod
    def get_customer_by_id(docs: List[Document], customer_id: int) -> Document:
        """
        Busca y devuelve la información de un cliente específico por su ID.

        Args:
            docs (List[Document]): Lista de documentos que representan registros de clientes.
            customer_id (int): ID del cliente a buscar.

        Returns:
            Document: Documento con la información del cliente, o None si no se encuentra.
        """
        for doc in docs:
            if doc.metadata.get("customer_id") == customer_id:
                return doc
        return None