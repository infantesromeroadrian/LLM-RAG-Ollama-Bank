# 8. main.py

"""
main.py

Este módulo contiene la función principal para ejecutar el sistema RAG (Retrieval-Augmented Generation)
y realizar pruebas con preguntas predefinidas. Es el punto de entrada principal para la aplicación
cuando se ejecuta fuera del entorno de Streamlit.

Dependencias:
- os: Para manejar rutas de archivos y directorios.
- src.models.rag_system: Para el sistema RAG.
- src.utils.test_runner: Para ejecutar pruebas automatizadas.
"""

import os
from src.models.rag_system import RAGSystem
from src.utils.test_runner import run_tests

def main():
    """
    Función principal que inicializa y ejecuta el sistema RAG, y realiza pruebas con preguntas predefinidas.

    Esta función realiza las siguientes operaciones:
    1. Configura las rutas de los directorios y archivos necesarios.
    2. Inicializa y configura el sistema RAG.
    3. Define un conjunto de preguntas de prueba.
    4. Ejecuta las pruebas utilizando el sistema RAG.

    Note:
        Las rutas de los archivos están configuradas para un entorno específico y pueden
        necesitar ajustes dependiendo de la estructura del proyecto.
    """
    # Configuración de rutas
    base_dir = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/YouTube/LangChainRAGOllama/data"
    pdf_directory = os.path.join(base_dir, "GuideLines")
    csv_file = os.path.join(base_dir, "raw_data", "BankCustomerChurnPrediction.csv")

    # Inicialización y configuración del sistema RAG
    rag_system = RAGSystem(pdf_directory=pdf_directory, csv_file=csv_file, base_dir=base_dir)
    rag_system.run()

    # Definición de preguntas de prueba
    test_questions = [
        "¿Cuántos clientes únicos tenemos en el banco según nuestra data en el CSV?",
        "¿Cuál es el saldo promedio de los clientes?",
        "¿Cuántos países están representados en nuestros datos de clientes?",
        "¿Cuál es la tasa de abandono de clientes?",
        "¿Cuál es el rango de edades de nuestros clientes?",
        "¿Qué porcentaje de clientes tiene tarjeta de crédito?"
    ]

    # Ejecución de pruebas
    print("Iniciando pruebas del sistema RAG...")
    run_tests(rag_system, test_questions)
    print("Pruebas completadas.")

if __name__ == "__main__":
    main()