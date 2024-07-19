# 8. main.py

import os
from src.models.rag_system import RAGSystem

def run_tests(rag_system, questions):
    for i, question in enumerate(questions, 1):
        print(f"\nPregunta {i}: {question}")
        try:
            response = rag_system.ask_question(question)
            print("Respuesta del modelo:")
            print(response["result"])
            print("\nFuente principal:")
            print(f"Fuente: {response['source_documents'][0].metadata.get('source', 'No especificada')}")
            print(response['source_documents'][0].page_content[:200])  # Primeros 200 caracteres
        except Exception as e:
            print(f"Error al procesar la pregunta: {str(e)}")
        print("-" * 50)

def main():
    base_dir = "/Users/adrianinfantes/Desktop/AIR/CollegeStudies/MachineLearningPath/YouTube/LangChainRAGOllama/data"
    pdf_directory = os.path.join(base_dir, "GuideLines")
    csv_file = os.path.join(base_dir, "raw_data", "BankCustomerChurnPrediction.csv")

    rag_system = RAGSystem(pdf_directory=pdf_directory, csv_file=csv_file, base_dir=base_dir)
    rag_system.run()

    test_questions = [
        "¿Cuántos clientes únicos tenemos en el banco según nuestra data en el CSV?",
        "¿Cuál es el saldo promedio de los clientes?",
        "¿Cuántos países están representados en nuestros datos de clientes?",
        "¿Cuál es la tasa de abandono de clientes?",
        "¿Cuál es el rango de edades de nuestros clientes?",
        "¿Qué porcentaje de clientes tiene tarjeta de crédito?"
    ]

    run_tests(rag_system, test_questions)

if __name__ == "__main__":
    main()