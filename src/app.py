import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import logging

from utils.logger_config import setup_logger
from features.cargador_datos_csv import CargadorDatosCSV
from features.gestor_clientes import GestorClientes
from model.sistema_rag import SistemaRAG

# Configurar logging
setup_logger("bank_app.log")


class BankApp:
    def __init__(self):
        """Inicializa la aplicación bancaria."""
        self.inicializar_sistema()

    def inicializar_sistema(self):
        """Inicializa los componentes del sistema."""
        try:
            # Configurar rutas
            ruta_csv = Path("data/raw_data/BankCustomerChurnPrediction.csv")
            vector_db = Path("vector_db")

            # Cargar datos
            self.cargador = CargadorDatosCSV(str(ruta_csv))
            self.df = self.cargador.cargar_datos()

            if self.df is not None:
                self.gestor = GestorClientes(self.df)
                self.rag = SistemaRAG(
                    ruta_archivo=str(ruta_csv),
                    persist_directory=str(vector_db)
                )
                return True
            return False

        except Exception as e:
            st.error(f"Error al inicializar el sistema: {str(e)}")
            logging.error(f"Error de inicialización: {str(e)}")
            return False

    def mostrar_estadisticas_generales(self):
        """Muestra estadísticas generales del banco."""
        st.subheader("📊 Estadísticas Generales")

        col1, col2, col3 = st.columns(3)

        with col1:
            total_clientes = len(self.df)
            st.metric("Total Clientes", f"{total_clientes:,}")

        with col2:
            churn_rate = (self.df['churn'].mean() * 100)
            st.metric("Tasa de Deserción", f"{churn_rate:.1f}%")

        with col3:
            balance_promedio = self.df['balance'].mean()
            st.metric("Balance Promedio", f"${balance_promedio:,.2f}")

        # Gráficos
        col1, col2 = st.columns(2)

        with col1:
            # Distribución de Credit Score
            fig_score = px.histogram(
                self.df,
                x='credit_score',
                title='Distribución de Credit Score',
                color='churn',
                barmode='group'
            )
            st.plotly_chart(fig_score, use_container_width=True)

        with col2:
            # Deserción por país
            churn_by_country = self.df.groupby('country')['churn'].mean().reset_index()
            fig_country = px.bar(
                churn_by_country,
                x='country',
                y='churn',
                title='Tasa de Deserción por País',
                labels={'churn': 'Tasa de Deserción'}
            )
            st.plotly_chart(fig_country, use_container_width=True)

    def analizar_cliente(self, customer_id):
        """Analiza un cliente específico."""
        try:
            stats = self.gestor.obtener_estadisticas_cliente(customer_id)
            if stats:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🔍 Información del Cliente")
                    for key, value in stats.items():
                        if key == 'risk_level':
                            color = {
                                'BAJO': 'green',
                                'MEDIO': 'orange',
                                'ALTO': 'red'
                            }.get(value, 'black')
                            st.markdown(f"**Nivel de Riesgo:** :{color}[{value}]")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

                with col2:
                    cliente_data = self.df[self.df['customer_id'] == customer_id]
                    if not cliente_data.empty:
                        # Comparar con promedios
                        st.subheader("📈 Comparativa con Promedios")
                        metrics = ['credit_score', 'balance', 'products_number']

                        for metric in metrics:
                            avg_value = self.df[metric].mean()
                            client_value = cliente_data[metric].iloc[0]
                            delta = ((client_value - avg_value) / avg_value) * 100

                            st.metric(
                                metric.replace('_', ' ').title(),
                                f"{client_value:,.2f}",
                                f"{delta:+.1f}% vs promedio"
                            )
            else:
                st.warning("No se encontró información para este cliente")

        except Exception as e:
            st.error(f"Error al analizar cliente: {str(e)}")
            logging.error(f"Error en análisis de cliente: {str(e)}")

    def realizar_consulta_rag(self, consulta):
        """Realiza una consulta al sistema RAG."""
        try:
            with st.spinner('Analizando datos...'):
                resultado = self.rag.realizar_consulta(consulta)

                st.subheader("🤖 Respuesta del Sistema")
                st.write(resultado['respuesta'])

                with st.expander("Ver detalles del análisis"):
                    st.write("**Metadatos:**")
                    st.write(f"- Documentos analizados: {resultado['metadatos']['num_documentos']}")
                    st.write(f"- Tiempo de respuesta: {resultado['metadatos']['tiempo_respuesta']:.2f} segundos")
                    st.write(f"- Modelo utilizado: {resultado['metadatos']['modelo']}")

                    st.write("\n**Documentos fuente utilizados:**")
                    for i, doc in enumerate(resultado['documentos_fuente'], 1):
                        st.text(f"Documento {i}:\n{doc}\n")

        except Exception as e:
            st.error(f"Error en la consulta: {str(e)}")
            logging.error(f"Error en consulta RAG: {str(e)}")


def main():
    st.set_page_config(
        page_title="Sistema Bancario Inteligente",
        page_icon="🏦",
        layout="wide"
    )

    st.title("🏦 Sistema Bancario Inteligente")

    # Inicializar aplicación
    if 'app' not in st.session_state:
        st.session_state.app = BankApp()

    # Menú lateral
    st.sidebar.title("Navegación")
    opciones = [
        "Dashboard General",
        "Análisis de Cliente",
        "Consultas Inteligentes"
    ]
    seleccion = st.sidebar.radio("Seleccione una opción:", opciones)

    if seleccion == "Dashboard General":
        st.session_state.app.mostrar_estadisticas_generales()

    elif seleccion == "Análisis de Cliente":
        st.subheader("🔍 Análisis de Cliente")
        customer_id = st.number_input(
            "Ingrese ID del cliente:",
            min_value=0,
            step=1
        )
        if st.button("Analizar Cliente"):
            st.session_state.app.analizar_cliente(customer_id)

    else:  # Consultas Inteligentes
        st.subheader("💡 Consultas Inteligentes")

        # Ejemplos de consultas
        st.write("**Ejemplos de consultas:**")
        ejemplos = [
            "¿Cuáles son los factores más comunes de deserción?",
            "¿Cómo influye el credit score en la deserción?",
            "¿Qué relación hay entre el balance y la retención?",
            "¿Los clientes activos tienen menor tasa de deserción?"
        ]
        for ejemplo in ejemplos:
            if st.button(ejemplo):
                st.session_state.app.realizar_consulta_rag(ejemplo)

        # Consulta personalizada
        st.write("\n**O realice su propia consulta:**")
        consulta = st.text_area("Escriba su consulta:")
        if st.button("Realizar Consulta"):
            if consulta:
                st.session_state.app.realizar_consulta_rag(consulta)
            else:
                st.warning("Por favor, ingrese una consulta")


if __name__ == "__main__":
    main()