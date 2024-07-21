# 8. main.py

from src.models.rag_system import RAGSystem
from src.utils.test_runner import run_tests, plot_results

def main():
    # Inicializar y ejecutar el sistema RAG
    rag_system = RAGSystem(
        pdf_directory="../data/GuideLines",
        csv_file="../data/raw_data/BankCustomerChurnPrediction.csv",
        base_dir="../data"
    )
    rag_system.run()

    # Definir preguntas de prueba y respuestas de referencia
    questions = [
        "¿Cuál es el saldo promedio de los clientes?",
        "¿Cuántos clientes tienen tarjeta de crédito?",
        "¿Cuáles son las regulaciones sobre el cumplimiento bancario?",
        "¿Qué recomendaciones se hacen en el documento PDF para la gestión de riesgos?"
    ]
    reference_answers = [
        "El saldo promedio de los clientes es 76,485.89 euros.",
        "Un total de 7050 clientes tienen tarjeta de crédito.",
        "Las regulaciones sobre el cumplimiento bancario incluyen medidas de KYC y AML.",
        "Las recomendaciones para la gestión de riesgos incluyen la implementación de controles internos efectivos."
    ]

    # Ejecutar pruebas
    results = run_tests(rag_system, questions, reference_answers, "resultados_test.csv")

    # Generar gráficos de resultados
    plot_results(results)

    # Ejemplo de uso interactivo
    while True:
        user_question = input("\nIngrese una pregunta (o 'salir' para terminar): ")
        if user_question.lower() == 'salir':
            break
        response = rag_system.ask_question(user_question)
        print(f"Respuesta: {response['result']}")

if __name__ == "__main__":
    main()