"""Renderizado del On-Screen Display.

Dibuja sobre el frame lo que el pipeline concluyó: detecciones coloreadas por
nivel de riesgo, la cresta del pretil, y un encabezado con el estado de análisis.

Dos criterios gobiernan el diseño visual:

**Legibilidad sobre fondo desconocido.** La escena es ocre de día y casi negra de
noche. Un texto sin fondo propio resulta ilegible en la mitad del material, así que
toda etiqueta se dibuja sobre un rectángulo opaco.

**Escala relativa a la resolución.** El material mezcla 1080p y 720p. Grosores y
tamaños de fuente fijos en píxeles se ven finos en el primero y gruesos en el
segundo, de modo que se derivan de la altura del frame.
"""

from __future__ import annotations

from typing import Final

import cv2

from bermguard.core.types import (
    Detection,
    FrameResult,
    ImageBGR,
    LightingCondition,
    RiskLevel,
)

COLOR_POR_RIESGO: Final[dict[RiskLevel, tuple[int, int, int]]] = {
    RiskLevel.SAFE: (80, 200, 80),
    RiskLevel.CAUTION: (40, 200, 245),
    RiskLevel.CRITICAL: (60, 60, 240),
}
"""Colores en BGR, que es el orden nativo de OpenCV.

Verde, ámbar y rojo, según el enunciado. Los tonos están desaturados respecto al
puro para que no se confundan con los focos de faena, que en las secuencias
nocturnas producen halos de rojo y ámbar saturados.
"""

COLOR_SIN_EVALUAR: Final[tuple[int, int, int]] = (200, 200, 200)
"""Gris para detecciones sin nivel de riesgo asignado.

Una detección sin evaluar no es una detección segura. Pintarla de verde afirmaría
algo que el sistema no calculó.
"""

COLOR_CRESTA: Final[tuple[int, int, int]] = (255, 170, 0)
COLOR_BASE: Final[tuple[int, int, int]] = (255, 90, 200)
COLOR_HUD_FONDO: Final[tuple[int, int, int]] = (25, 25, 25)
COLOR_HUD_TEXTO: Final[tuple[int, int, int]] = (245, 245, 245)

FUENTE: Final[int] = cv2.FONT_HERSHEY_SIMPLEX


class OsdRenderer:
    """Compone la capa de anotación sobre cada frame."""

    def __init__(self, draw_hud: bool = True) -> None:
        self._draw_hud = draw_hud

    def render(self, frame: ImageBGR, result: FrameResult) -> ImageBGR:
        """Devuelve una copia anotada de ``frame``.

        No modifica la entrada: el frame original sigue disponible para las etapas
        de análisis, que deben operar sobre píxeles limpios y no sobre los dibujos.
        """
        lienzo = frame.copy()
        alto = lienzo.shape[0]
        grosor = max(2, round(alto / 360))
        escala = alto / 900.0

        if result.berm_pixels is not None:
            self._dibujar_pretil(lienzo, result, grosor)

        for deteccion in result.detections:
            riesgo = result.risk_by_track.get(deteccion.track_id)
            self._dibujar_deteccion(lienzo, deteccion, riesgo, grosor, escala)

        if self._draw_hud:
            self._dibujar_hud(lienzo, result, escala)

        return lienzo

    def _dibujar_deteccion(
        self,
        lienzo: ImageBGR,
        deteccion: Detection,
        riesgo: RiskLevel | None,
        grosor: int,
        escala: float,
    ) -> None:
        color = COLOR_POR_RIESGO[riesgo] if riesgo else COLOR_SIN_EVALUAR
        x1, y1, x2, y2 = (round(v) for v in deteccion.bbox.as_tuple())
        cv2.rectangle(lienzo, (x1, y1), (x2, y2), color, grosor)

        # El punto de contacto con el suelo se dibuja porque es el que alimenta la
        # proyeccion a vista cenital: verlo permite auditar visualmente si la caja
        # esta bien apoyada, que es de donde sale toda la metrica de proximidad.
        gx, gy = deteccion.bbox.ground_point
        cv2.drawMarker(lienzo, (round(gx), round(gy)), color, cv2.MARKER_CROSS, grosor * 6, grosor)

        etiqueta = deteccion.vehicle_class.value.upper()
        if deteccion.track_id >= 0:
            etiqueta += f" #{deteccion.track_id}"
        etiqueta += f" {deteccion.confidence:.2f}"
        if riesgo is not None:
            etiqueta += f" | {riesgo.value}"

        self._texto_con_fondo(lienzo, etiqueta, (x1, y1), color, escala, grosor)

    def _dibujar_pretil(self, lienzo: ImageBGR, result: FrameResult, grosor: int) -> None:
        pixeles = result.berm_pixels
        if pixeles is None:
            return
        for serie, color in ((pixeles.crest_y_px, COLOR_CRESTA), (pixeles.base_y_px, COLOR_BASE)):
            anterior: tuple[int, int] | None = None
            for x, y in enumerate(serie):
                # NaN marca columnas sin medicion. Se interrumpe la linea en vez de
                # interpolar: un hueco declarado informa, una interpolacion miente.
                if y != y:
                    anterior = None
                    continue
                actual = (x, round(float(y)))
                if anterior is not None:
                    cv2.line(lienzo, anterior, actual, color, grosor)
                anterior = actual

    def _dibujar_hud(self, lienzo: ImageBGR, result: FrameResult, escala: float) -> None:
        latencia = sum(result.stage_latency_ms.values())
        lineas = [
            f"frame {result.index}  t={result.timestamp_s:5.2f}s",
            f"luz: {_ETIQUETA_LUZ[result.lighting]}",
            f"equipos: {len(result.detections)}",
            f"{latencia:.0f} ms/frame",
        ]
        if result.berm_pixels is not None:
            lineas.append(f"pretil: cobertura {result.berm_pixels.coverage:.0%}")

        alto_linea = round(26 * escala)
        margen = round(12 * escala)
        ancho = max(round(300 * escala), 180)
        cv2.rectangle(
            lienzo,
            (margen, margen),
            (margen + ancho, margen + alto_linea * len(lineas) + margen),
            COLOR_HUD_FONDO,
            cv2.FILLED,
        )
        for i, linea in enumerate(lineas):
            cv2.putText(
                lienzo,
                linea,
                (margen * 2, margen + alto_linea * (i + 1)),
                FUENTE,
                0.55 * escala,
                COLOR_HUD_TEXTO,
                max(1, round(escala)),
                cv2.LINE_AA,
            )

    @staticmethod
    def _texto_con_fondo(
        lienzo: ImageBGR,
        texto: str,
        origen: tuple[int, int],
        color: tuple[int, int, int],
        escala: float,
        grosor: int,
    ) -> None:
        tam_fuente = 0.5 * escala
        (ancho, alto), linea_base = cv2.getTextSize(texto, FUENTE, tam_fuente, 1)
        x, y = origen
        # Si la caja toca el borde superior, la etiqueta se dibuja por dentro para
        # que no quede recortada fuera del frame.
        y_texto = y - linea_base if y - alto - linea_base > 0 else y + alto + linea_base * 2
        cv2.rectangle(
            lienzo,
            (x, y_texto - alto - linea_base),
            (x + ancho + grosor * 2, y_texto + linea_base),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            lienzo,
            texto,
            (x + grosor, y_texto),
            FUENTE,
            tam_fuente,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


_ETIQUETA_LUZ: Final[dict[LightingCondition, str]] = {
    LightingCondition.DAY: "dia",
    LightingCondition.DUSK: "crepusculo",
    LightingCondition.NIGHT: "noche",
}
