import graphviz

def create_rag_system_diagram():
    # Crear un nuevo gráfico dirigido
    dot = graphviz.Digraph(comment='Sistema RAG Bancario')
    dot.attr(rankdir='TB', size='8,8')

    # Definir estilos
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', color='gray')

    # Agregar nodos
    dot.node('A', 'Usuario', shape='ellipse', fillcolor='lightgreen')
    dot.node('B', 'Interfaz Streamlit')
    dot.node('C', 'StreamlitRAGSystem')
    dot.node('D', 'RAGSystem', fillcolor='lightyellow')
    dot.node('E', 'DocumentLoader')
    dot.node('F', 'PyMuPDFLoader')
    dot.node('G', 'CSVLoader')
    dot.node('H', 'DataProcessor')
    dot.node('I', 'TextSplitter')
    dot.node('J', 'CSV Summary')
    dot.node('K', 'VectorStoreManager')
    dot.node('L', 'FastEmbedEmbeddings')
    dot.node('M', 'Chroma VectorStore', shape='cylinder', fillcolor='lightpink')
    dot.node('N', 'CustomRetriever')
    dot.node('O', 'Ollama LLM', shape='diamond', fillcolor='lightcoral')
    dot.node('P', 'Configuración del Sistema', shape='parallelogram', fillcolor='lightsalmon')
    dot.node('Q', 'Ejecución de Pruebas', shape='parallelogram', fillcolor='lightsalmon')
    dot.node('R', 'EvaluationMetrics')

    # Agregar conexiones
    dot.edge('A', 'B', 'Ingresa Pregunta')
    dot.edge('B', 'C', 'Envía Pregunta')
    dot.edge('C', 'D', 'Procesa Pregunta')
    dot.edge('D', 'E', 'Carga Documentos')
    dot.edge('E', 'F', 'PDF')
    dot.edge('E', 'G', 'CSV')
    dot.edge('F', 'H')
    dot.edge('G', 'H')
    dot.edge('H', 'I', 'Divide Documentos')
    dot.edge('H', 'J', 'Crea Resumen CSV')
    dot.edge('I', 'K')
    dot.edge('J', 'K')
    dot.edge('K', 'L', 'Crea Embeddings')
    dot.edge('L', 'M')
    dot.edge('D', 'N', 'Recupera Documentos Relevantes')
    dot.edge('N', 'M', 'Consulta')
    dot.edge('D', 'O', 'Genera Respuesta')
    dot.edge('O', 'D', 'Respuesta')
    dot.edge('D', 'C', 'Devuelve Respuesta')
    dot.edge('C', 'B', 'Muestra Respuesta')
    dot.edge('B', 'A', 'Presenta Respuesta')
    dot.edge('P', 'C', 'Actualiza Parámetros')
    dot.edge('Q', 'R', 'Evalúa Rendimiento')
    dot.edge('R', 'B', 'Resultados')

    # Agrupar componentes relacionados
    with dot.subgraph(name='cluster_0') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        c.edges(['FH', 'GH'])
        c.attr(label='Procesamiento de Documentos')

    with dot.subgraph(name='cluster_1') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        c.edges(['KL', 'LM'])
        c.attr(label='Gestión de Vectores')

    return dot

# Crear y guardar el diagrama
diagram = create_rag_system_diagram()
diagram.render('rag_system_diagram', format='png', cleanup=True)
print("Diagrama generado como 'rag_system_diagram.png'")