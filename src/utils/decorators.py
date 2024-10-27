import time
import logging
from functools import wraps

def time_decorator(func):
    """Decorador para medir tiempo de ejecución."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Tiempo de ejecución de '{func.__name__}': {end_time - start_time:.2f} segundos")
        return result
    return wrapper

def log_decorator(func):
    """Decorador para registrar información sobre las operaciones realizadas."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Ejecutando operación: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper