# 🏦 Sistema RAG Bancario con Ollama

## 🌟 Introducción

Este proyecto implementa un sistema de Generación Aumentada por Recuperación (RAG) para consultas bancarias, utilizando Ollama como modelo de lenguaje. El sistema está diseñado para proporcionar respuestas precisas a preguntas sobre datos bancarios, combinando la potencia de los modelos de lenguaje con la recuperación de información específica del dominio.

![Diagrama del Sistema RAG](https://github.com/infantesromeroadrian/RAG-Ollama-Bank/blob/93914d09dbb2758b191fdd7b6877ae7b747e97c6/assets/rag_system_diagram.png)

## 🚀 Características Principales

- 💻 Interfaz interactiva basada en Streamlit
- 📄 Procesamiento de documentos PDF y CSV
- 🤖 Generación de respuestas utilizando Ollama
- 🔍 Recuperación personalizada de documentos relevantes
- ⚙️ Sistema configurable con parámetros ajustables
- 📊 Evaluación de rendimiento integrada

## 📋 Requisitos

- 🐍 Python 3.11+
- 🖥️ Streamlit
- 🔗 LangChain
- 🦙 Ollama
- 📚 PyMuPDF
- 🐼 Pandas
- 📈 Plotly
- 🌳 Graphviz (para la generación de diagramas)

## 🛠️ Instalación

1. Clone el repositorio:

git clone https://github.com/infantesromeroadrian/RAG-Ollama-Bank.git
cd rag-ollama-bank

2. Instale las dependencias:

Asegúrese de tener Ollama instalado y configurado en su sistema.

## 🏃‍♂️ Uso

Para iniciar la aplicación Streamlit:

streamlit run app.py

## 📁 Estructura del Proyecto

- `app.py`: 🚪 Punto de entrada principal y interfaz de Streamlit
- `src/`
  - `models/`: 🧠 Contiene el sistema RAG y componentes relacionados
  - `utils/`: 🔧 Utilidades para carga de documentos, procesamiento y evaluación
  - `features/`: 📈 Funciones para generar gráficos y visualizaciones
- `data/`: 💾 Directorio para almacenar documentos PDF y CSV
- `assets/`: 🖼️ Contiene recursos como el diagrama del sistema

## 🔄 Explicación del Diagrama del Sistema

El diagrama muestra el flujo de datos y los componentes principales del sistema RAG:

1. 👤 **Usuario e Interfaz**: El flujo comienza con el usuario interactuando con la interfaz Streamlit.
2. 🔄 **Procesamiento de la Pregunta**: La pregunta pasa por el `StreamlitRAGSystem` al `RAGSystem` central.
3. 📥 **Carga y Procesamiento de Documentos**: Se utilizan `DocumentLoader` y `DataProcessor` para manejar PDFs y CSVs.
4. 🧮 **Gestión de Vectores**: `VectorStoreManager` y `FastEmbedEmbeddings` crean y gestionan embeddings en `Chroma VectorStore`.
5. 🔎 **Recuperación y Generación**: `CustomRetriever` obtiene documentos relevantes, y `Ollama LLM` genera la respuesta final.
6. ⚙️ **Configuración y Evaluación**: El sistema permite ajustes de configuración y incluye métricas de evaluación.

## 🎛️ Configuración

- Ajuste los parámetros del modelo y del sistema a través de la interfaz de Streamlit.
- Modifique los directorios de datos en `app.py` según sea necesario.

## 🧪 Evaluación y Pruebas

El sistema incluye funcionalidades para ejecutar pruebas y evaluar el rendimiento:

1. Use el botón "Ejecutar Pruebas" en la interfaz de Streamlit.
2. Los resultados se guardarán en un archivo CSV y se visualizarán en la interfaz.

## 🎥 Demostración

Para ver una demostración de cómo se usa este sistema RAG, consulte este video:
[RAG-UI Demo](https://github.com/infantesromeroadrian/RAG-Ollama-Bank/blob/6cf62a9250e49624845b0874724100e7ef93e1b4/assets/RAG-UI.MOV)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abra un issue para discutir cambios mayores antes de enviar un pull request.

## 📜 Licencia

[MIT](https://choosealicense.com/licenses/mit/) © [Adrián Infantes](https://github.com/infantesromeroadrian)

---

Para más información o consultas, por favor abra un issue en el repositorio de GitHub. 📩
