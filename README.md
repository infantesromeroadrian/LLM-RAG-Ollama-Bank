# 📚🤖 Chatbot con Recuperación y Generación de Respuestas (RAG)

¡Bienvenido al proyecto **Chatbot con RAG**! Este proyecto utiliza un modelo de lenguaje para responder a tus preguntas basándose en la información contenida en un archivo PDF que cargues. 🎉

## 🚀 Funcionalidades

- 📄 **Carga de archivos PDF**: Sube cualquier archivo PDF y nuestro sistema lo procesará para extraer la información.
- 🧠 **Modelo de Lenguaje**: Utiliza un potente modelo de lenguaje para generar respuestas basadas en el contenido del PDF.
- 🔍 **Recuperación de Información**: Encuentra y muestra las secciones relevantes del PDF que contienen las respuestas.
- 💬 **Interfaz de Chat**: Haz preguntas y obtén respuestas de manera interactiva a través de una interfaz web fácil de usar.

## 🛠️ Cómo Configurar y Ejecutar el Proyecto

### Prerrequisitos

Asegúrate de tener instaladas las siguientes herramientas:

- Python 3.7+
- Poetry (opcional, pero recomendado para manejar dependencias)

### Instalación

1. **Clona el repositorio**:
    ```sh
    git clone https://github.com/infantesromeroadrian/RAG-Ollama.git
    cd RAG-Ollama
    ```

2. **Instala las dependencias**:
    - Con Poetry:
        ```sh
        poetry install
        pipenv shell
        ```
    
### Ejecución

1. **Ejecuta la aplicación de Streamlit**:
    ```sh
    streamlit run app.py
    ```

2. **Abre tu navegador** y navega a `http://localhost:8501` para ver la aplicación en acción.

## 📂 Estructura del Proyecto

- `app.py`: Archivo principal que contiene la aplicación Streamlit.
- `language_model.py`: Módulo para cargar y utilizar el modelo de lenguaje RAG.
- `document_loader.py`: Módulo para cargar, procesar y almacenar los datos del PDF.
- `text_splitter.py`: Módulo para dividir el texto del PDF en fragmentos.
- `embedding_model.py`: Módulo para crear y almacenar los embeddings del PDF.
- `vector_store.py`: Módulo para crear y almacenar el vector de búsqueda.
- `qa_pipeline.py`: Módulo para crear y ejecutar la cadena de RAG.

## ✨ Cómo Usar la Aplicación

1. **Carga un archivo PDF**: En la barra lateral, selecciona un archivo PDF desde tu computadora.
2. **Espera a que se procese**: El archivo será cargado y procesado automáticamente.
3. **Haz una pregunta**: Escribe tu pregunta en el cuadro de texto y presiona el botón "Preguntar".
4. **Obtén la respuesta**: La respuesta generada se mostrará junto con los metadatos de los documentos fuente.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar el proyecto, siéntete libre de abrir un issue o enviar un pull request.

## 📧 Contacto

Si tienes alguna pregunta o necesitas ayuda, no dudes en contactarme a través de [infantesromeroadrian@gmail.com](mailto:infantesromeroadrian@gmail.com).

## 🌟 Agradecimientos

A todos los que han contribuido y apoyado este proyecto. ¡Gracias!

---

¡Esperamos que disfrutes usando el Chatbot con RAG! 🚀🤖📚