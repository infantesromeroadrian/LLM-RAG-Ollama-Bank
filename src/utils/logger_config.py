import logging
from pathlib import Path


def setup_logger(log_file: str = "system.log"):
    """
    Configura el logger del sistema.

    Args:
        log_file: Nombre del archivo de log
    """
    # Crear directorio logs si no existe
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configurar el logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / log_file)
        ]
    )