"""Detección de cortes de toma.

Implementa la mitad estructural del ADR 0003. Un corte invalida las identidades
del tracker, la calibración del plano de suelo y todo filtro temporal, de modo que
el orquestador necesita saber, frame a frame, si empezó una toma nueva.

La señal elegida es el **salto de luminancia media**, y no el cambio estructural
que usan los detectores de escena convencionales. El motivo está medido: en este
material los cortes ocurren entre tomas de encuadre parecido, que alteran poco la
estructura y mucho la exposición. Un criterio estructural los pasa por alto,
mientras que la luminancia los separa con holgura —los cortes de ``video_01``
saltan más de 20 unidades, y las transiciones graduales de los otros clips se
mueven unas pocas por frame.

Opera en línea, sobre un frame a la vez y sin mirar hacia adelante, porque el
pipeline procesa video de forma secuencial y no puede permitirse una pasada previa
sobre cada archivo.
"""

from __future__ import annotations

import logging

from bermguard.core.types import ImageBGR
from bermguard.vision.lighting import mean_luma

logger = logging.getLogger(__name__)


class ShotDetector:
    """Señala el primer frame de cada toma.

    Example:
        >>> detector = ShotDetector()
        >>> for index, _ts, frame in reader.frames():
        ...     if detector.update(frame) and index > 0:
        ...         tracker.reset()
    """

    def __init__(self, luma_delta: float = 20.0, min_length: int = 6) -> None:
        """
        Args:
            luma_delta: Salto de luminancia media que se considera un corte.
            min_length: Frames mínimos entre dos cortes. Un destello, un frame
                corrupto o una explosión de polvo pueden producir un salto puntual;
                exigir una duración mínima evita que eso genere una toma espuria y
                reinicie el estado del pipeline sin necesidad.
        """
        self._luma_delta = luma_delta
        self._min_length = min_length
        self._luma_previa: float | None = None
        self._frames_en_toma = 0
        self._indice_toma = 0

    def reset(self) -> None:
        """Reinicia el detector. Se llama al empezar un video nuevo."""
        self._luma_previa = None
        self._frames_en_toma = 0
        self._indice_toma = 0

    @property
    def shot_index(self) -> int:
        """Índice de la toma actual dentro del video, empezando en cero."""
        return self._indice_toma

    def update(self, frame: ImageBGR) -> bool:
        """Procesa un frame y devuelve si inicia una toma nueva.

        Returns:
            ``True`` para el primer frame del video y para el primer frame de cada
            toma posterior. El llamador decide qué hacer con esa señal; en el caso
            del primer frame del video no hay estado que reiniciar.
        """
        luma = mean_luma(frame)

        if self._luma_previa is None:
            self._luma_previa = luma
            self._frames_en_toma = 1
            return True

        salto = abs(luma - self._luma_previa)
        self._luma_previa = luma
        self._frames_en_toma += 1

        if salto > self._luma_delta and self._frames_en_toma > self._min_length:
            self._indice_toma += 1
            self._frames_en_toma = 1
            logger.debug(
                "Corte de toma detectado: salto de luminancia %.1f, toma %d",
                salto,
                self._indice_toma,
            )
            return True

        return False
