import pandas as pd
from pathlib import Path
import logging
from utils.decorators import time_decorator


class CargadorDatosCSV:
    def __init__(self, ruta_archivo: str):
        """
        Inicializa el cargador de datos CSV.

        Args:
            ruta_archivo: Ruta al archivo CSV
        """
        self.ruta_archivo = Path(ruta_archivo)
        self.df = None

    @time_decorator
    def cargar_datos(self) -> pd.DataFrame:
        """
        Carga los datos desde el archivo CSV.

        Returns:
            DataFrame con los datos cargados o None si hay error
        """
        try:
            logging.info(f"Cargando datos desde {self.ruta_archivo}...")
            self.df = pd.read_csv(self.ruta_archivo)
            logging.info("Datos cargados exitosamente")
            logging.info(f"Columnas del DataFrame: {self.df.columns.tolist()}")
            return self.df

        except FileNotFoundError:
            logging.error(f"Archivo no encontrado: {self.ruta_archivo}")
            return None
        except pd.errors.EmptyDataError:
            logging.error("El archivo CSV está vacío")
            return None
        except Exception as e:
            logging.error(f"Error inesperado: {e}")
            return None

    def obtener_dataframe(self) -> pd.DataFrame:
        """Retorna el DataFrame cargado."""
        if self.df is None:
            raise ValueError("No hay datos cargados. Ejecute cargar_datos() primero")
        return self.df.copy()