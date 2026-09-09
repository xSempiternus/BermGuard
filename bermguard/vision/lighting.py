"""Clasificación de la condición lumínica de un frame.

Implementa la mitad lumínica del ADR 0003. La condición se determina por frame y
no por archivo porque el material transiciona de noche a día pleno dentro de un
mismo clip de diez segundos, de modo que una etiqueta a nivel de archivo no
describiría nada.

Cumple dos funciones a la vez, y esa coincidencia es deliberada:

* Estratifica las métricas del benchmark, que de otro modo quedarían dominadas
  por el caso nocturno —entre el 53 % y el 57 % de los frames del material de
  muestra— mientras aparentan describir el sistema completo.
* Gobierna el acondicionamiento adaptativo del frame previo a la inferencia.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from bermguard.core.types import ImageBGR, LightingCondition


@dataclass(frozen=True, slots=True)
class LightingThresholds:
    """Cortes de luminancia media que separan las tres condiciones.

    Los valores por defecto se calibraron sobre el material de muestra, cuya
    luminancia media recorre el rango 19–163. El conjunto de evaluación es ciego,
    así que su generalización es un supuesto declarado y no un hecho verificado:
    por eso los umbrales son configurables y no constantes del módulo.
    """

    night_max: float = 70.0
    day_min: float = 110.0

    def __post_init__(self) -> None:
        if not 0 < self.night_max < self.day_min < 255:
            raise ValueError(
                f"Umbrales inconsistentes: night_max={self.night_max}, day_min={self.day_min}"
            )


DEFAULT_THRESHOLDS = LightingThresholds()


def mean_luma(frame: ImageBGR) -> float:
    """Luminancia media del frame, en el rango ``[0, 255]``.

    Se calcula sobre la conversión a escala de grises y no sobre el promedio de
    los canales BGR: la conversión pondera los canales según la sensibilidad del
    ojo humano, que es la referencia correcta cuando lo que se quiere describir es
    «qué tan oscura se ve la escena».
    """
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def classify_luma(
    luma: float, thresholds: LightingThresholds = DEFAULT_THRESHOLDS
) -> LightingCondition:
    """Clasifica una luminancia media ya calculada."""
    if luma < thresholds.night_max:
        return LightingCondition.NIGHT
    if luma > thresholds.day_min:
        return LightingCondition.DAY
    return LightingCondition.DUSK


def classify_frame(
    frame: ImageBGR, thresholds: LightingThresholds = DEFAULT_THRESHOLDS
) -> LightingCondition:
    """Clasifica un frame directamente."""
    return classify_luma(mean_luma(frame), thresholds)
