"""Segmentación del pretil por máximo de gradiente por columna — Método 2.

Es el enfoque convencional para localizar un borde estructural horizontal: derivar,
tomar el máximo de respuesta en cada columna, y suavizar el perfil resultante.

**Existe como línea base medible, no como aspirante.** Se implementó primero, se
midió, y su fracaso es lo que motivó la formulación por camino óptimo del Método 1
(ADR 0006). Conservarlo en el entregable permite cuantificar la diferencia entre
ambos en lugar de afirmarla: sin una línea base, «el camino óptimo es mejor» es una
opinión.

## La diferencia conceptual con el Método 1

Ambos parten de la misma respuesta de gradiente. Lo que cambia es **cuándo se impone
el prior de continuidad**:

* Aquí, **después**: cada columna elige su máximo de forma independiente y luego un
  filtro intenta reparar el perfil. Cuando el filtro actúa, la información de
  continuidad ya se perdió — el máximo local de cada columna se eligió sin saber
  nada de sus vecinas.
* En el Método 1, **durante**: la continuidad es una restricción de la búsqueda, de
  modo que ninguna columna puede elegir un valor incompatible con las contiguas.

Es la misma distinción que separa un ajuste robusto de un filtrado posterior, y el
reporte de benchmark la cuantifica mediante el jitter temporal del perfil.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from bermguard.core.types import BermPixels, Detection, FloatArray, ImageBGR

logger = logging.getLogger(__name__)


class ArgmaxBermSegmenter:
    """Línea base: máximo de gradiente por columna con filtrado posterior.

    Satisface :class:`~bermguard.core.interfaces.IBermSegmenter`.

    Comparte con el Método 1 el acondicionamiento, la exclusión de maquinaria y la
    banda de búsqueda, de modo que la comparación aísla **la única diferencia que
    importa**: cómo se decide la cresta. Un benchmark que además cambiara el
    preprocesado no diría nada sobre la formulación.
    """

    def __init__(
        self,
        clahe_clip: float = 2.5,
        blur_sigma: float = 6.0,
        noise_multiple: float = 3.0,
        median_window: int = 31,
        smooth_window: int = 31,
        outlier_tolerance: float = 3.0,
        sky_variance_threshold: float = 6.0,
        band_margin_frac: float = 0.04,
        box_dilation_px: int = 12,
    ) -> None:
        """
        Args:
            outlier_tolerance: Múltiplo de la desviación mediana a partir del cual un
                punto se descarta como enganche a otra estructura. Es el parámetro
                que intenta reparar lo que el ``argmax`` rompió.

        Note:
            El resto de los parámetros son idénticos a los del Método 1 y con el
            mismo significado, deliberadamente: la comparación debe aislar la
            formulación de la búsqueda.
        """
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        self._blur_sigma = blur_sigma
        self._noise_multiple = noise_multiple
        self._median_window = median_window
        self._smooth_window = smooth_window
        self._tolerancia = outlier_tolerance
        self._sky_threshold = sky_variance_threshold
        self._band_margin = band_margin_frac
        self._box_dilation = box_dilation_px

    @property
    def name(self) -> str:
        return "baseline:per-column-argmax"

    def reset(self) -> None:
        """Sin estado temporal que reiniciar.

        La ausencia de suavizado entre frames es parte de lo que se mide: el Método 1
        lo tiene, y el jitter reportado en el benchmark refleja esa diferencia junto
        con la de formulación.
        """

    def segment(self, frame: ImageBGR, detections: Sequence[Detection] = ()) -> BermPixels | None:
        alto, ancho = frame.shape[:2]

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        realzado = self._clahe.apply(gris)

        k = int(self._blur_sigma * 6) | 1
        suave = cv2.GaussianBlur(realzado, (k, k), self._blur_sigma)
        respuesta = np.maximum(cv2.Scharr(suave, cv2.CV_32F, 0, 1), 0.0)

        d = self._box_dilation
        for deteccion in detections:
            x1, y1, x2, y2 = deteccion.bbox.as_tuple()
            respuesta[
                max(0, int(y1) - d) : min(alto, int(y2) + d),
                max(0, int(x1) - d) : min(ancho, int(x2) + d),
            ] = 0.0

        varianza = realzado.std(axis=1)
        con_textura = np.flatnonzero(varianza > self._sky_threshold)
        y_min = int(con_textura[0]) if con_textura.size else 0
        if detections:
            rasante = max(det.bbox.ground_point[1] for det in detections)
            y_max = int(min(alto, rasante + alto * self._band_margin))
        else:
            y_max = int(alto * 0.80)
        if y_max - y_min < 16:
            return None

        banda = respuesta[y_min:y_max, :]
        pico = float(banda.max())
        if pico <= 0:
            return None

        # LA diferencia con el Metodo 1: cada columna decide sola.
        cresta_local = banda.argmax(axis=0)
        fuerza = banda[cresta_local, np.arange(ancho)]

        corte = max(self._noise_multiple * float(np.median(banda)), pico * 0.05)
        valida = fuerza > corte
        if valida.sum() < ancho * 0.1:
            return None

        cresta = (cresta_local + y_min).astype(np.float32)
        cresta[~valida] = np.nan
        cresta = self._reparar(cresta)

        base = self._estimar_base(respuesta, cresta, y_max)
        cobertura = float(np.isfinite(cresta).sum() / ancho)
        return BermPixels(
            crest_y_px=cresta,
            base_y_px=base,
            coverage=cobertura,
            confidence=cobertura * (0.5 if not detections else 1.0),
        )

    # --- Auxiliares -----------------------------------------------------------

    def _reparar(self, cresta: FloatArray) -> FloatArray:
        """Intenta recuperar continuidad a posteriori: mediana móvil y Savitzky-Golay.

        Este método es el objeto del experimento. El filtrado sólo puede atenuar los
        saltos que el ``argmax`` produjo; no puede recuperar la elección correcta,
        porque esa información se descartó al decidir cada columna por separado.
        """
        indices = np.flatnonzero(np.isfinite(cresta))
        if indices.size < 7:
            return cresta

        valores = cresta[indices]
        ventana = min(self._median_window | 1, (valores.size // 2) * 2 - 1)
        if ventana >= 5:
            mediana = median_filter(valores, size=ventana, mode="nearest")
            desvio = np.abs(valores - mediana)
            tolerancia = self._tolerancia * max(float(np.median(desvio)), 1.0)
            valores = np.where(desvio <= tolerancia, valores, np.nan)

        finitos = np.isfinite(valores)
        if finitos.sum() >= 7:
            largo = min(self._smooth_window | 1, (int(finitos.sum()) // 2) * 2 - 1)
            if largo >= 5:
                valores[finitos] = savgol_filter(valores[finitos], largo, 2, mode="nearest")

        salida = np.full_like(cresta, np.nan)
        salida[indices] = valores
        return salida

    @staticmethod
    def _estimar_base(respuesta: np.ndarray, cresta: FloatArray, y_max: int) -> FloatArray:
        """Idéntico al del Método 1, para que la comparación aísle la cresta."""
        alto, ancho = respuesta.shape
        fin = min(alto, y_max + int(alto * 0.10))
        valida = np.isfinite(cresta)
        if not valida.any():
            return np.full_like(cresta, np.nan)

        filas = np.arange(alto, dtype=np.float32)[:, None]
        cresta_segura = np.where(valida, cresta, np.inf)
        fondo = np.percentile(respuesta[:fin, :], 25, axis=0)
        indices = np.where(valida, np.nan_to_num(cresta, nan=0.0), 0).astype(int)
        pico = respuesta[np.clip(indices, 0, alto - 1), np.arange(ancho)]

        umbral = fondo + 0.5 * (pico - fondo)
        cae = (respuesta < umbral[None, :]) & (filas > cresta_segura[None, :]) & (filas < fin)
        primero = cae.argmax(axis=0).astype(np.float32)
        hay_cruce = cae.any(axis=0) & valida & (pico > fondo)
        return np.where(hay_cruce, primero, np.nan).astype(np.float32)
