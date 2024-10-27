# src/main.py
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Importaciones relativas desde el directorio src
from utils.logger_config import setup_logger
from features.cargador_datos_csv import CargadorDatosCSV
from features.gestor_clientes import GestorClientes
from model.sistema_rag import SistemaRAG


class SistemaBancario:
    """Sistema integrado para análisis bancario con RAG."""

    def __init__(
            self,
            ruta_csv: str = "../data/raw_data/BankCustomerChurnPrediction.csv",
            persist_directory: str = "./vector_db"
    ):
        """
        Inicializa el sistema bancario.

        Args:
            ruta_csv: Ruta al archivo de datos
            persist_directory: Directorio para la base vectorial
        """
        # Configurar logging
        setup_logger("banco_system.log")

        self.ruta_csv = Path(ruta_csv)
        self.persist_directory = Path(persist_directory)

        # Componentes del sistema
        self.cargador = None
        self.gestor = None
        self.rag = None
        self.dataframe = None

        # Inicializar sistema
        self._inicializar_sistema()

    def _inicializar_sistema(self) -> None:
        """Inicializa todos los componentes del sistema."""
        try:
            logging.info("Iniciando sistema bancario...")

            # Cargar datos
            self.cargador = CargadorDatosCSV(str(self.ruta_csv))
            self.dataframe = self.cargador.cargar_datos()
            if self.dataframe is None:
                raise ValueError("Error al cargar los datos")

            # Inicializar gestor
            self.gestor = GestorClientes(self.dataframe)

            # Inicializar RAG
            self.rag = SistemaRAG(
                ruta_archivo=str(self.ruta_csv),
                persist_directory=str(self.persist_directory)
            )

            logging.info("Sistema bancario inicializado correctamente")

        except Exception as e:
            logging.error(f"Error al inicializar el sistema: {e}")
            raise

    def analizar_cliente(self, customer_id: int) -> Optional[Dict]:
        """
        Analiza un cliente específico.

        Args:
            customer_id: ID del cliente

        Returns:
            Diccionario con estadísticas del cliente
        """
        try:
            return self.gestor.obtener_estadisticas_cliente(customer_id)
        except Exception as e:
            logging.error(f"Error al analizar cliente {customer_id}: {e}")
            return None

    def consultar_rag(self, consulta: str) -> Dict[str, Any]:
        """
        Realiza una consulta al sistema RAG.

        Args:
            consulta: Pregunta a realizar

        Returns:
            Respuesta del sistema RAG
        """
        return self.rag.realizar_consulta(consulta)

    def actualizar_cliente(self, customer_id: int, nuevos_datos: Dict) -> bool:
        """
        Actualiza información de un cliente.

        Args:
            customer_id: ID del cliente
            nuevos_datos: Datos a actualizar

        Returns:
            True si la actualización fue exitosa
        """
        return self.gestor.actualizar_cliente(customer_id, nuevos_datos)


def mostrar_menu():
    """Muestra el menú de opciones."""
    print("\n=== Sistema Bancario con RAG ===")
    print("1. Analizar cliente")
    print("2. Realizar consulta RAG")
    print("3. Actualizar cliente")
    print("4. Salir")
    return input("\nSeleccione una opción (1-4): ")


def main():
    """Función principal del sistema."""
    try:
        # Inicializar sistema
        sistema = SistemaBancario()
        print("Sistema inicializado correctamente")

        while True:
            opcion = mostrar_menu()

            if opcion == "1":
                # Analizar cliente
                customer_id = int(input("Ingrese ID del cliente: "))
                stats = sistema.analizar_cliente(customer_id)

                if stats:
                    print("\nEstadísticas del cliente:")
                    print("-" * 30)
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                else:
                    print("No se encontró información del cliente")

            elif opcion == "2":
                # Consulta RAG
                print("\nEjemplos de consultas:")
                print("- ¿Cuáles son los factores más comunes de deserción?")
                print("- ¿Cómo influye el credit score en la deserción?")
                print("- ¿Qué relación hay entre el balance y la retención?")

                consulta = input("\nIngrese su consulta: ")
                resultado = sistema.consultar_rag(consulta)

                print("\nRespuesta:")
                print("-" * 50)
                print(resultado['respuesta'])
                print("\nEstadísticas:")
                print(f"Documentos analizados: {resultado['metadatos']['num_documentos']}")
                print(f"Tiempo de respuesta: {resultado['metadatos']['tiempo_respuesta']:.2f} segundos")

            elif opcion == "3":
                # Actualizar cliente
                customer_id = int(input("Ingrese ID del cliente: "))
                print("\nIngrese los nuevos datos (deje en blanco para omitir):")

                nuevos_datos = {}
                campos = {
                    'credit_score': int,
                    'balance': float,
                    'products_number': int
                }

                for campo, tipo in campos.items():
                    valor = input(f"{campo}: ").strip()
                    if valor:
                        nuevos_datos[campo] = tipo(valor)

                if nuevos_datos:
                    if sistema.actualizar_cliente(customer_id, nuevos_datos):
                        print("Cliente actualizado correctamente")
                    else:
                        print("Error al actualizar el cliente")
                else:
                    print("No se proporcionaron datos para actualizar")

            elif opcion == "4":
                print("Gracias por usar el sistema")
                break

            else:
                print("Opción no válida")

            # Pausa para leer resultados
            input("\nPresione Enter para continuar...")

    except Exception as e:
        logging.error(f"Error en la ejecución: {e}")
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())