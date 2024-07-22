# test_qa_system.py

import unittest
from unittest.mock import Mock, patch
from src.models.qa_system import QASystem


class TestQASystem(unittest.TestCase):

    def setUp(self):
        self.mock_llm = Mock()
        self.mock_retriever = Mock()
        self.qa_system = QASystem(self.mock_llm, self.mock_retriever)

    def test_init(self):
        self.assertEqual(self.qa_system.llm, self.mock_llm)
        self.assertEqual(self.qa_system.custom_retriever, self.mock_retriever)
        self.assertIsNone(self.qa_system.qa_chain)

    @patch('src.utils.qa_system.PromptTemplate')
    @patch('src.utils.qa_system.RetrievalQA')
    def test_setup_qa_chain(self, mock_retrieval_qa, mock_prompt_template):
        mock_chain = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_chain

        self.qa_system.setup_qa_chain()

        mock_prompt_template.assert_called_once()
        mock_retrieval_qa.from_chain_type.assert_called_once()
        self.assertEqual(self.qa_system.qa_chain, mock_chain)

    def test_ask_question_without_setup(self):
        with self.assertRaises(ValueError):
            self.qa_system.ask_question("Test question")

    @patch('src.utils.qa_system.RetrievalQA')
    def test_ask_question(self, mock_retrieval_qa):
        # Configurar el mock para RetrievalQA
        mock_chain = Mock()
        mock_chain.invoke.return_value = {"result": "Test answer", "source_documents": ["doc1", "doc2"]}
        mock_retrieval_qa.from_chain_type.return_value = mock_chain

        # Configurar la cadena QA
        self.qa_system.setup_qa_chain()

        # Realizar una pregunta
        result = self.qa_system.ask_question("Test question")

        # Verificar que se llamó al método invoke del mock_chain
        mock_chain.invoke.assert_called_once_with({"query": "Test question"})

        # Verificar el resultado
        self.assertEqual(result["result"], "Test answer")
        self.assertEqual(result["source_documents"], ["doc1", "doc2"])

    @patch('src.utils.qa_system.PromptTemplate')
    @patch('src.utils.qa_system.RetrievalQA')
    def test_prompt_template_content(self, mock_retrieval_qa, mock_prompt_template):
        self.qa_system.setup_qa_chain()

        # Verificar que PromptTemplate se llamó con el template correcto
        call_args = mock_prompt_template.call_args
        template = call_args[1]['template']

        self.assertIn("Utiliza la siguiente información del CSV y del PDF para responder a la pregunta del usuario.",
                      template)
        self.assertIn("IMPORTANTE: Para preguntas sobre datos legales o normativa del banco usa siempre el PDF.",
                      template)
        self.assertIn(
            "Si la pregunta se refiere a datos numéricos o estadísticas del banco, asegúrate de usar la información del resumen del CSV.",
            template)
        self.assertIn("Responde en el mismo idioma que la pregunta.", template)


if __name__ == '__main__':
    unittest.main()