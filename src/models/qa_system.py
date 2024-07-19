# 6.qa_system.py

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


class QASystem:
    def __init__(self, llm, custom_retriever):
        self.llm = llm
        self.custom_retriever = custom_retriever
        self.qa_chain = None

    def setup_qa_chain(self):
        prompt = PromptTemplate(
            template="""Utiliza la siguiente información para responder a la pregunta del usuario.
            IMPORTANTE: Para preguntas sobre datos agregados o estadísticas del banco, SIEMPRE consulta primero el resumen del CSV.
            Este resumen contiene información precisa y confiable sobre el conjunto de datos completo.
            No te bases en ejemplos individuales para hacer generalizaciones sobre todo el conjunto de datos.

            Si la pregunta se refiere a datos numéricos o estadísticas del banco, asegúrate de usar la información del resumen del CSV.
            Si no encuentras la respuesta en el resumen, entonces puedes consultar los documentos CSV individuales.
            Si aún así no puedes responder, indica que no lo sabes.
            No inventes respuestas. Responde en el mismo idioma que la pregunta.

            Contexto: {context}
            Pregunta: {question}

            Proporciona solo la respuesta útil a continuación, nada más. Si la respuesta involucra un recuento o estadística, asegúrate de proporcionar el número exacto encontrado en el resumen del CSV.
            Respuesta útil:
            """,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.custom_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask_question(self, question: str):
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Run setup_qa_chain first.")
        return self.qa_chain.invoke({"query": question})
