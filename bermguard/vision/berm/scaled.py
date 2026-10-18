"""Corre un segmentador de pretil a menor resolución y devuelve el perfil a escala completa.

Con GPU, el pretil es la etapa más cara del pipeline: 36 ms por frame a 720p y 74 ms a
1080p, contra 16 y 29 ms del detector (reporte, sección 5). La GPU no lo acelera porque
corre en NumPy, y la mitad de ese costo es el camino óptimo, que recorre las columnas
una por una.

El pretil es una estructura de baja frecuencia: se busca sobre un gradiente suavizado
con sigma 6, así que la resolución completa no aporta información que el método use. A
la mitad hay cuatro veces menos celdas y la mitad de columnas que recorrer.

Es un envoltorio y no un parámetro del segmentador para que valga igual para los dos
métodos del benchmark, sin tocar ninguno.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import cv2
import numpy as np

from bermguard.core.interfaces import IBermSegmenter
from bermguard.core.types import BBox, BermPixels, Detection, FloatArray, ImageBGR


class ScaledBermSegmenter:
    """Satisface :class:`~bermguard.core.interfaces.IBermSegmenter` delegando en otro."""

    def __init__(self, inner: IBermSegmenter, scale: float = 0.5) -> None:
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"La escala debe estar en (0, 1], no {scale}")
        self._inner = inner
        self._scale = scale

    @property
    def name(self) -> str:
        return f"{self._inner.name}@{self._scale:g}"

    def reset(self) -> None:
        self._inner.reset()

    def segment(self, frame: ImageBGR, detections: Sequence[Detection] = ()) -> BermPixels | None:
        if self._scale == 1.0:
            return self._inner.segment(frame, detections)

        alto, ancho = frame.shape[:2]
        chico = cv2.resize(
            frame,
            (round(ancho * self._scale), round(alto * self._scale)),
            interpolation=cv2.INTER_AREA,
        )
        s = self._scale
        escaladas = [
            replace(d, bbox=BBox(d.bbox.x1 * s, d.bbox.y1 * s, d.bbox.x2 * s, d.bbox.y2 * s))
            for d in detections
        ]
        perfil = self._inner.segment(chico, escaladas)  # type: ignore[arg-type]
        if perfil is None:
            return None
        return replace(
            perfil,
            crest_y_px=_a_escala_completa(perfil.crest_y_px, ancho, 1.0 / s),
            base_y_px=_a_escala_completa(perfil.base_y_px, ancho, 1.0 / s),
        )


def _a_escala_completa(serie: FloatArray, ancho: int, factor: float) -> FloatArray:
    """Reescala filas y columnas. Una columna sin medición sigue sin medición."""
    x_chico = (np.arange(serie.size, dtype=np.float64) + 0.5) * factor - 0.5
    x = np.arange(ancho, dtype=np.float64)
    valores = np.interp(x, x_chico, np.nan_to_num(serie, nan=0.0)) * factor
    # Huecos: el vecino más cercano a escala reducida decide si hay medición.
    vecino = np.clip(np.round((x + 0.5) / factor - 0.5).astype(int), 0, serie.size - 1)
    valores[~np.isfinite(serie[vecino])] = np.nan
    return valores.astype(np.float32)
