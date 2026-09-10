"""Estimación de la altura métrica del pretil.

Convierte el perfil en píxeles que produce el segmentador a metros sobre la
rasante, con banda de incertidumbre.

## La geometría

Para un objeto apoyado en el plano de suelo, con la base en la fila ``y_base`` y la
cima en ``y_crest``, bajo el modelo pinhole del módulo :mod:`geometry`:

    y_base − y_h = f · h / Z          (la base está en el suelo)
    y_base − y_crest = f · H / Z      (extensión vertical del objeto)

Dividiendo una por la otra, la focal y la profundidad se cancelan:

    H = (y_base − y_crest) · h / (y_base − y_h)

Es un resultado cómodo: la altura **no depende de la distancia focal**, que es el
parámetro más incierto de la cadena —se asume de un campo de visión declarado—.
Sólo necesita el horizonte y la altura de montaje de la cámara, y esta última se
deriva del ancho nominal de un CAEX detectado.

Que la focal se cancele no significa que la medición sea precisa: el error se
traslada íntegro al ancho nominal supuesto y a la propia caja del vehículo.

## La banda de incertidumbre

Cada CAEX visible produce una estimación independiente de la altura de cámara. Con
varios, su dispersión es una medida directa —no modelada— de la incertidumbre del
ancla, y se propaga linealmente a la altura porque la relación es lineal en ``h``.

Con un solo vehículo no hay dispersión que medir, y se declara una incertidumbre
mínima del 20 % en lugar de reportar una banda de cero. Una banda nula afirmaría
una precisión que el método no tiene.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from bermguard.analytics.geometry import GroundPlaneModel, fit_ground_plane
from bermguard.core.exceptions import InsufficientDataError
from bermguard.core.types import BermPixels, BermProfile, Detection, FloatArray

logger = logging.getLogger(__name__)

INCERTIDUMBRE_MINIMA = 0.20
"""Incertidumbre relativa declarada cuando hay una sola ancla métrica.

No es una medición sino un piso honesto: con un único vehículo la dispersión
observable es cero, y reportar una banda de cero afirmaría una precisión que el
método no posee. El 20 % corresponde al orden del error esperado sin calibración.
"""


class GroundPlaneHeightEstimator:
    """Estima la altura del pretil proyectando sobre el plano de suelo.

    Satisface :class:`~bermguard.core.interfaces.IHeightEstimator`.

    Recibe el horizonte y el ancho del frame por :meth:`observe_frame` porque el
    contrato de ``estimate`` no los incluye. Es estado, pero explícito y de un solo
    frame: el orquestador lo actualiza antes de cada evaluación.
    """

    def __init__(self, horizontal_fov_deg: float = 60.0) -> None:
        self._fov = horizontal_fov_deg
        self._horizonte: float | None = None
        self._ancho: int = 0
        self._modelo: GroundPlaneModel | None = None

    def reset(self) -> None:
        """Descarta la calibración acumulada. Se llama en cada corte (ADR 0003)."""
        self._modelo = None
        self._horizonte = None

    def observe_frame(self, horizon_y_px: float, frame_width: int) -> None:
        self._horizonte = horizon_y_px
        self._ancho = frame_width

    @property
    def model(self) -> GroundPlaneModel | None:
        return self._modelo

    def estimate(
        self,
        pixels: BermPixels,
        detections: Sequence[Detection],
        frame_index: int,
    ) -> BermProfile | None:
        if self._horizonte is None or self._ancho <= 0:
            return None

        try:
            self._modelo = fit_ground_plane(
                list(detections), self._ancho, self._horizonte, self._fov
            )
        except InsufficientDataError:
            # Sin ancla no hay escala. Se devuelve None y el orquestador lo registra
            # como frame sin medicion metrica, en vez de arrastrar la calibracion de
            # otro frame y presentar metros que no se midieron aqui.
            return None

        modelo = self._modelo
        base = pixels.base_y_px.astype(np.float64)
        cresta = pixels.crest_y_px.astype(np.float64)

        # La relacion es lineal en la altura de camara, asi que el cociente de
        # extensiones verticales se escala directamente por ella.
        bajo_horizonte = base - modelo.horizon_y_px
        extension = base - cresta
        valido = np.isfinite(extension) & (extension > 0) & (bajo_horizonte > 1.0)

        cociente = np.where(valido, extension / np.where(valido, bajo_horizonte, 1.0), np.nan)
        altura = cociente * modelo.camera_height_m

        relativa = self._incertidumbre_relativa(modelo)
        return BermProfile(
            pixels=pixels,
            height_m=altura.astype(np.float32),
            height_m_lo=(altura * (1.0 - relativa)).astype(np.float32),
            height_m_hi=(altura * (1.0 + relativa)).astype(np.float32),
            height_ratio=(cociente * 100.0).astype(np.float32),
            scale_confidence=self._confianza(modelo, relativa),
        )

    # --- Auxiliares -----------------------------------------------------------

    @staticmethod
    def _incertidumbre_relativa(modelo: GroundPlaneModel) -> float:
        """Incertidumbre relativa de la altura, derivada de la dispersión del ancla."""
        if modelo.anchors < 2 or modelo.camera_height_spread_m <= 0:
            return INCERTIDUMBRE_MINIMA
        relativa = modelo.camera_height_spread_m / max(modelo.camera_height_m, 1e-6)
        return float(max(relativa, INCERTIDUMBRE_MINIMA))

    @staticmethod
    def _confianza(modelo: GroundPlaneModel, incertidumbre: float) -> float:
        """Confianza en el modelo de escala, no en la segmentación."""
        # Mas anclas y menos dispersion suben la confianza. Con una sola ancla el
        # techo es 0.5: la escala depende de un unico objeto y su error de caja se
        # propaga sin promediarse con nada.
        por_anclas = min(modelo.anchors / 3.0, 1.0)
        por_dispersion = float(np.clip(1.0 - incertidumbre, 0.0, 1.0))
        return float(np.clip(por_anclas * por_dispersion, 0.0, 1.0))


def median_height(profile: BermProfile) -> float | None:
    """Altura mediana del pretil en el frame, ignorando columnas sin medición."""
    validos: FloatArray = profile.height_m[np.isfinite(profile.height_m)]
    return float(np.median(validos)) if validos.size else None
