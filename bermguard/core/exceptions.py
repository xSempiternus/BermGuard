"""Jerarquía de excepciones de BermGuard.

Tener una jerarquía propia permite capturar con precisión sólo aquello de lo que
se puede recuperar. Un ``except Exception`` se tragaría también los errores de
programación (``AttributeError``, ``KeyError``) y los trataría como fallos
esperados, escondiendo bugs reales detrás de un reintento o una línea de log.

El contrato en todo el código es:

* Un ``BermGuardError`` es una condición esperada y recuperable. El procesamiento
  por lotes la registra y continúa con el siguiente video.
* Cualquier otra cosa es un bug y debe propagarse.
"""

from __future__ import annotations


class BermGuardError(Exception):
    """Clase base de todo error que esta aplicación lanza deliberadamente."""


class ConfigError(BermGuardError):
    """Un archivo de configuración falta, está mal formado o es inconsistente."""


class VideoReadError(BermGuardError):
    """No se pudo abrir o decodificar una fuente de video.

    Cubre códecs ilegibles, archivos truncados, fuentes de cero frames y archivos
    que no son video pero están en el directorio de entrada.
    """


class VideoWriteError(BermGuardError):
    """No se pudo crear o escribir un video de salida."""


class ModelLoadError(BermGuardError):
    """Faltan los pesos del modelo o no se pudieron cargar en el dispositivo."""


class CalibrationError(BermGuardError):
    """No se pudo recuperar la geometría de la escena.

    Se lanza cuando el plano del suelo, la escala píxel-metro o la homografía no
    pueden estimarse con la evidencia disponible.
    """


class InsufficientDataError(BermGuardError):
    """No hay observaciones válidas suficientes para un resultado significativo.

    Distinto de :class:`CalibrationError`: el método sí es aplicable, sólo que
    todavía no hay señal suficiente (por ejemplo, menos de dos vehículos a
    distintas profundidades al ajustar el modelo de escala).
    """
