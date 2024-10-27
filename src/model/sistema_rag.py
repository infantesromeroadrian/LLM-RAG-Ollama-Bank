import os
import time
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path
import logging

# Importación corregida de Ollama
from langchain_community.llms import Ollama
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from utils.decorators import time_decorator

class SistemaRAG:
    """Sistema RAG para análisis de datos bancarios."""

    def __init__(
            self,
            ruta_archivo: str = "../data/raw_data/BankCustomerChurnPrediction.csv",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            model_name: str = "llama3.2",
            persist_directory: Optional[str] = "./vector_db"
    ):
        """
        Inicializa el sistema RAG.

        Args:
            ruta_archivo: Ruta al archivo CSV
            chunk_size: Tamaño de chunk
            chunk_overlap: Solapamiento
            model_name: Modelo de Ollama
            persist_directory: Directorio de persistencia
        """
        self.ruta_archivo = self._validar_ruta_archivo(ruta_archivo)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.llm = None
        self.vector_db = None
        self.retriever = None

        self._verificar_y_preparar_modelo()
        self._cargar_y_procesar_documento()
        logging.info("Sistema RAG inicializado correctamente")



    def _validar_ruta_archivo(self, ruta: str) -> str:
            """
            Valida la ruta del archivo.

            Args:
                ruta: Ruta al archivo CSV

            Returns:
                str: Ruta absoluta validada

            Raises:
                ValueError: Si la ruta no es válida
            """
            try:
                ruta_absoluta = Path(ruta).resolve()
                if not ruta_absoluta.is_file():
                    raise ValueError(f"Archivo no encontrado: {ruta_absoluta}")
                if ruta_absoluta.suffix.lower() != '.csv':
                    raise ValueError("El archivo debe ser CSV")
                return str(ruta_absoluta)
            except Exception as e:
                raise ValueError(f"Error de validación: {e}")

    def _verificar_y_preparar_modelo(self) -> None:
        """
        Verifica y prepara el modelo Ollama.

        Raises:
            RuntimeError: Si hay problemas con el modelo
        """
        try:
            logging.info(f"Inicializando modelo {self.model_name}...")
            self.llm = Ollama(model=self.model_name)
            self.llm.invoke("test")
            logging.info(f"Modelo {self.model_name} inicializado correctamente")
        except Exception as e:
            logging.warning(f"Error al inicializar modelo: {e}. Intentando descargar...")
            try:
                subprocess.run(["ollama", "pull", self.model_name], check=True)
                self.llm = Ollama(model=self.model_name)
                logging.info(f"Modelo descargado e inicializado correctamente")
            except Exception as e:
                raise RuntimeError(f"No se pudo inicializar el modelo: {e}")

    @time_decorator
    def _cargar_y_procesar_documento(self) -> None:
        """
        Carga y procesa el documento CSV.

        Raises:
            Exception: Si hay errores en el procesamiento
        """
        try:
            logging.info("Iniciando carga y procesamiento del documento...")

            # Cargar el CSV
            loader = CSVLoader(
                file_path=self.ruta_archivo,
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"'
                }
            )
            documentos = loader.load()
            logging.info(f"Documento cargado: {len(documentos)} registros")

            # Dividir en chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(documentos)
            logging.info(f"Documento dividido en {len(chunks)} chunks")

            # Crear embeddings
            embeddings = FastEmbedEmbeddings()

            # Crear o cargar base vectorial
            if self.persist_directory:
                persist_path = Path(self.persist_directory)
                if persist_path.exists():
                    self.vector_db = Chroma(
                        persist_directory=str(persist_path),
                        embedding_function=embeddings
                    )
                    logging.info("Base de datos vectorial cargada desde disco")
                else:
                    persist_path.mkdir(parents=True, exist_ok=True)
                    self.vector_db = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=str(persist_path)
                    )
                    logging.info("Base de datos vectorial creada y persistida")
            else:
                self.vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                logging.info("Base de datos vectorial creada en memoria")

            # Configurar retriever
            self.retriever = self.vector_db.as_retriever(
                search_kwargs={"k": 5}
            )

        except Exception as e:
            logging.error(f"Error en el procesamiento del documento: {e}")
            raise

    def _crear_prompt_template(self) -> PromptTemplate:
        """
        Crea el template para las consultas.

        Returns:
            PromptTemplate configurado
        """
        template = """Analiza los datos bancarios proporcionados y responde la pregunta.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
- Basa tu respuesta solo en los datos proporcionados
- Menciona valores numéricos cuando estén disponibles
- Identifica patrones relevantes
- Si no hay suficiente información, indícalo claramente

Respuesta:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    @time_decorator
    def realizar_consulta(
            self,
            consulta: str,
            temperatura: float = 0.7,
            max_tokens: int = 500
    ) -> Dict[str, Any]:
        """Realiza una consulta al sistema."""
        if not isinstance(consulta, str) or not consulta.strip():
            raise ValueError("La consulta debe ser un texto no vacío")

        if not 0 <= temperatura <= 1:
            raise ValueError("La temperatura debe estar entre 0 y 1")

        try:
            logging.info(f"Realizando consulta: {consulta}")

            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self._crear_prompt_template()}
            )

            # Realizar consulta
            start_time = time.time()
            response = chain.invoke({"query": consulta})
            end_time = time.time()

            resultado = {
                'respuesta': response['result'],
                'documentos_fuente': [doc.page_content for doc in response['source_documents']],
                'metadatos': {
                    'tiempo_respuesta': end_time - start_time,
                    'num_documentos': len(response['source_documents']),
                    'modelo': self.model_name
                }
            }

            logging.info(f"Consulta completada en {end_time - start_time:.2f} segundos")
            return resultado

        except Exception as e:
            logging.error(f"Error al realizar la consulta: {e}")
            raise

    def reiniciar(self) -> None:
        """
        Reinicia el sistema y la base de datos vectorial.

        Raises:
            Exception: Si hay errores en el reinicio
        """
        try:
            if self.vector_db is not None:
                self.vector_db = None
                self.retriever = None
                if self.persist_directory:
                    persist_path = Path(self.persist_directory)
                    if persist_path.exists():
                        for item in persist_path.glob('*'):
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                item.rmdir()
                self._cargar_y_procesar_documento()
                logging.info("Sistema reiniciado correctamente")
        except Exception as e:
            logging.error(f"Error al reiniciar el sistema: {e}")
            raise