import pandas as pd
from typing import Dict, Optional
import logging
from utils.decorators import time_decorator, log_decorator


class GestorClientes:
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el gestor de clientes.

        Args:
            df: DataFrame con datos de clientes
        """
        self.df = df.copy()
        self._validar_columnas_requeridas()
        logging.info("Gestor de clientes inicializado correctamente")

    def _validar_columnas_requeridas(self) -> None:
        """Valida las columnas requeridas del DataFrame."""
        columnas_requeridas = {
            'customer_id', 'credit_score', 'balance',
            'products_number', 'active_member'
        }
        columnas_faltantes = columnas_requeridas - set(self.df.columns)
        if columnas_faltantes:
            raise ValueError(f"Faltan columnas requeridas: {columnas_faltantes}")

    @time_decorator
    @log_decorator
    def actualizar_cliente(self, customer_id: int, nuevos_datos: Dict) -> bool:
        """
        Actualiza información de un cliente.

        Args:
            customer_id: ID del cliente
            nuevos_datos: Datos a actualizar

        Returns:
            True si la actualización fue exitosa
        """
        if customer_id not in self.df['customer_id'].values:
            logging.warning(f"Cliente {customer_id} no encontrado")
            return False

        try:
            # Validar columnas
            columnas_invalidas = set(nuevos_datos.keys()) - set(self.df.columns)
            if columnas_invalidas:
                logging.error(f"Columnas inválidas: {columnas_invalidas}")
                return False

            # Validar tipos de datos
            for key, value in nuevos_datos.items():
                if not isinstance(value, type(self.df[key].iloc[0])):
                    logging.error(
                        f"Tipo inválido para {key}. "
                        f"Esperado: {type(self.df[key].iloc[0])}, "
                        f"Recibido: {type(value)}"
                    )
                    return False

            # Actualizar datos
            for key, value in nuevos_datos.items():
                self.df.loc[self.df['customer_id'] == customer_id, key] = value
                logging.info(f"Cliente {customer_id}: {key} actualizado a {value}")

            return True

        except Exception as e:
            logging.error(f"Error en actualización: {e}")
            return False

    @time_decorator
    @log_decorator
    def obtener_estadisticas_cliente(self, customer_id: int) -> Optional[Dict]:
        """
        Obtiene estadísticas de un cliente.

        Args:
            customer_id: ID del cliente

        Returns:
            Diccionario con estadísticas o None
        """
        try:
            cliente = self.df[self.df['customer_id'] == customer_id]
            if cliente.empty:
                logging.warning(f"Cliente {customer_id} no encontrado")
                return None

            stats = {
                'credit_score': int(cliente['credit_score'].iloc[0]),
                'balance': float(cliente['balance'].iloc[0]),
                'products_number': int(cliente['products_number'].iloc[0]),
                'is_active': bool(cliente['active_member'].iloc[0]),
                'risk_level': self._calcular_nivel_riesgo(
                    cliente['credit_score'].iloc[0],
                    cliente['balance'].iloc[0]
                )
            }
            return stats

        except Exception as e:
            logging.error(f"Error al obtener estadísticas: {e}")
            return None

    def _calcular_nivel_riesgo(self, credit_score: int, balance: float) -> str:
        """Calcula nivel de riesgo del cliente."""
        if credit_score >= 750:
            return 'BAJO'
        elif credit_score >= 600:
            return 'MEDIO' if balance > 0 else 'ALTO'
        else:
            return 'ALTO'

    def obtener_dataframe(self) -> pd.DataFrame:
        """Retorna copia del DataFrame."""
        return self.df.copy()