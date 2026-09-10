"""Asociación de detecciones entre frames e identidad persistente.

Implementa las dos ideas centrales de ByteTrack (Zhang et al., ECCV 2022) sin su
maquinaria completa: **asociación en dos rondas** y **ciclo de vida con
confirmación**. No hay filtro de Kalman ni re-identificación por apariencia; el
modelo de movimiento es velocidad constante estimada del centro de la caja.

La simplificación es deliberada y tiene un motivo medido. Un Kalman aporta sobre
todo cuando el ruido de medición es alto frente al movimiento entre frames, y acá
las máquinas se desplazan poco por frame a 24–30 fps. La re-identificación por
apariencia aporta cuando hay muchos objetos parecidos que se cruzan; acá hay dos o
tres equipos por escena. Añadir ambas cosas habría sumado complejidad sin resolver
un problema observado.

Las tres funciones que sí resuelve:

**Confirmación antes de reportar.** Un track debe acumular varias detecciones
consecutivas antes de considerarse un equipo real. Esto ataca directamente el
problema medido en ``docs/resultados_deteccion.md``: el detector tiene recall 1.000
con precisión 0.297, de modo que cerca del 70 % de sus detecciones son falsos
positivos. Un falso positivo aislado no persiste entre frames —aparece sobre una
nube de polvo y desaparece—, así que exigir persistencia lo filtra sin tocar el
umbral de confianza, que es lo único que la clase minoritaria no puede permitirse.

**Supervivencia a oclusiones cortas.** Un track no visto se arrastra unos frames
según su velocidad estimada, en lugar de desaparecer. Sin esto la caja parpadea
cada vez que el detector pierde la máquina por un frame, y el enunciado penaliza
explícitamente el parpadeo.

**Recuperación de detecciones débiles.** La segunda ronda de asociación empareja
detecciones de baja confianza contra los tracks que quedaron sin pareja. Con polvo
y de noche un CAEX real baja a confianza 0.3; un tracker que sólo mirase
detecciones fuertes lo perdería, mientras que acá hay un track existente que lo
explica.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from bermguard.core.types import BBox, Detection, VehicleClass

logger = logging.getLogger(__name__)


def iou(a: BBox, b: BBox) -> float:
    """Intersección sobre unión de dos cajas, en ``[0, 1]``."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    interseccion = (x2 - x1) * (y2 - y1)
    union = a.width * a.height + b.width * b.height - interseccion
    return float(interseccion / union) if union > 0 else 0.0


@dataclass(slots=True)
class _Track:
    """Estado interno de un track. Mutable, a diferencia de los tipos del dominio."""

    track_id: int
    bbox: BBox
    vehicle_class: VehicleClass
    confidence: float

    velocidad: tuple[float, float] = (0.0, 0.0)
    """Desplazamiento del centro por frame, en píxeles."""

    aciertos: int = 1
    """Detecciones asociadas en total. Gobierna la confirmación."""

    sin_ver: int = 0
    """Frames consecutivos sin detección asociada."""

    confirmado: bool = False

    def predecir(self) -> BBox:
        """Caja esperada en el frame siguiente, por extrapolación lineal.

        Se traslada la caja sin cambiarle el tamaño. Extrapolar también la escala
        amplifica el ruido: un frame con la caja algo más grande haría crecer la
        predicción sin freno durante una oclusión.
        """
        dx, dy = self.velocidad
        return BBox(self.bbox.x1 + dx, self.bbox.y1 + dy, self.bbox.x2 + dx, self.bbox.y2 + dy)

    def asociar(self, deteccion: Detection, suavizado: float) -> None:
        """Incorpora una detección al track y corrige la velocidad estimada.

        Cuando este método se ejecuta, ``self.bbox`` ya es la **predicción** de este
        frame y no la última observación, porque el tracker predice antes de asociar.
        El desplazamiento real desde la observación anterior es por tanto
        ``velocidad + residual``, donde el residual es la diferencia entre lo
        predicho y lo observado.

        De ahí que la velocidad se corrija **sumando** el residual ponderado en lugar
        de promediar posiciones: es la misma estructura que la actualización de
        velocidad de un filtro de Kalman con ganancia fija. Calcularla como
        diferencia contra ``self.bbox`` haría que, con movimiento lineal, el residual
        fuese cero y la velocidad decayera a cero — y el arrastre durante una
        oclusión congelaría la caja en lugar de extrapolar.
        """
        cx_predicho, cy_predicho = self.bbox.center
        cx_observado, cy_observado = deteccion.bbox.center

        dx, dy = self.velocidad
        self.velocidad = (
            dx + suavizado * (cx_observado - cx_predicho),
            dy + suavizado * (cy_observado - cy_predicho),
        )

        self.bbox = deteccion.bbox
        self.confidence = deteccion.confidence
        self.vehicle_class = deteccion.vehicle_class
        self.aciertos += 1
        self.sin_ver = 0

    def arrastrar(self) -> None:
        """Avanza el track un frame sin detección."""
        self.bbox = self.predecir()
        self.sin_ver += 1


class IouTracker:
    """Tracking por asociación de IoU con asignación óptima.

    Satisface :class:`~bermguard.core.interfaces.ITracker`.

    Opera sobre :class:`~bermguard.core.types.Detection` y no sabe nada de modelos
    ni de imágenes, de modo que se puede testear con cajas construidas a mano.
    """

    def __init__(
        self,
        high_confidence: float = 0.45,
        min_hits: int = 3,
        max_age: int = 8,
        iou_high: float = 0.30,
        iou_low: float = 0.15,
        velocity_smoothing: float = 0.5,
    ) -> None:
        """
        Args:
            high_confidence: Frontera entre detecciones fuertes y débiles. Las
                fuertes pueden crear tracks nuevos; las débiles sólo continuar los
                existentes.
            min_hits: Detecciones necesarias para confirmar un track y empezar a
                reportarlo. Es el filtro de falsos positivos: sube la precisión sin
                tocar el umbral de confianza del detector.
            max_age: Frames que un track sobrevive sin ser visto antes de
                descartarse. A 24 fps, 8 frames son un tercio de segundo — suficiente
                para una nube de polvo, corto para no arrastrar un equipo que ya salió
                de escena.
            iou_high: Solape mínimo para aceptar un emparejamiento en la primera
                ronda.
            iou_low: Solape mínimo en la segunda ronda. Más permisivo, porque una
                detección débil suele traer la caja peor ajustada.
            velocity_smoothing: Peso del desplazamiento observado al actualizar la
                velocidad.
        """
        self._high_confidence = high_confidence
        self._min_hits = min_hits
        self._max_age = max_age
        self._iou_high = iou_high
        self._iou_low = iou_low
        self._suavizado = velocity_smoothing

        self._tracks: list[_Track] = []
        self._siguiente_id = 0

    def reset(self) -> None:
        """Descarta todos los tracks. Se llama en cada corte de toma (ADR 0003).

        Los identificadores **no** se reinician: un ``track_id`` sigue siendo único
        dentro del video, de modo que los artefactos no confunden dos equipos
        distintos de tomas distintas bajo el mismo número.
        """
        self._tracks.clear()

    def update(self, detections: Sequence[Detection], frame_index: int) -> Sequence[Detection]:
        """Asigna identidad y devuelve sólo los tracks confirmados."""
        for track in self._tracks:
            track.bbox = track.predecir()

        fuertes = [d for d in detections if d.confidence >= self._high_confidence]
        debiles = [d for d in detections if d.confidence < self._high_confidence]

        # Ronda 1: detecciones fuertes contra todos los tracks.
        sin_pareja = list(range(len(self._tracks)))
        emparejados, dets_libres, sin_pareja = self._emparejar(fuertes, sin_pareja, self._iou_high)
        for indice_track, deteccion in emparejados:
            self._tracks[indice_track].asociar(deteccion, self._suavizado)

        # Ronda 2: la idea distintiva de ByteTrack. Las detecciones debiles no crean
        # tracks, pero si pueden continuar uno existente que quedo huerfano.
        emparejados, _, sin_pareja = self._emparejar(debiles, sin_pareja, self._iou_low)
        for indice_track, deteccion in emparejados:
            self._tracks[indice_track].asociar(deteccion, self._suavizado)

        for indice_track in sin_pareja:
            self._tracks[indice_track].arrastrar()

        for deteccion in dets_libres:
            self._tracks.append(
                _Track(
                    track_id=self._siguiente_id,
                    bbox=deteccion.bbox,
                    vehicle_class=deteccion.vehicle_class,
                    confidence=deteccion.confidence,
                )
            )
            self._siguiente_id += 1

        antes = len(self._tracks)
        self._tracks = [t for t in self._tracks if t.sin_ver <= self._max_age]
        if antes != len(self._tracks):
            logger.debug(
                "Frame %d: %d track(s) descartado(s) por antiguedad",
                frame_index,
                antes - len(self._tracks),
            )

        salida: list[Detection] = []
        for track in self._tracks:
            if track.aciertos >= self._min_hits:
                track.confirmado = True
            if not track.confirmado:
                continue
            salida.append(
                Detection(
                    track_id=track.track_id,
                    vehicle_class=track.vehicle_class,
                    bbox=track.bbox,
                    confidence=track.confidence,
                )
            )
        return salida

    # --- Auxiliares -----------------------------------------------------------

    def _emparejar(
        self,
        detecciones: Sequence[Detection],
        indices_track: list[int],
        umbral_iou: float,
    ) -> tuple[list[tuple[int, Detection]], list[Detection], list[int]]:
        """Empareja detecciones con tracks maximizando el IoU total.

        Se usa el algoritmo húngaro (``linear_sum_assignment``) y no un emparejamiento
        greedy. La diferencia importa cuando dos máquinas se cruzan: greedy asigna la
        primera pareja que encuentra y puede forzar a la segunda a un emparejamiento
        malo, mientras la asignación óptima minimiza el costo total y suele conservar
        las identidades correctas.

        Returns:
            Las parejas aceptadas, las detecciones sin pareja, y los índices de track
            sin pareja.
        """
        if not detecciones or not indices_track:
            return [], list(detecciones), list(indices_track)

        costo = np.ones((len(indices_track), len(detecciones)), dtype=np.float32)
        for fila, indice_track in enumerate(indices_track):
            caja_track = self._tracks[indice_track].bbox
            for columna, deteccion in enumerate(detecciones):
                costo[fila, columna] = 1.0 - iou(caja_track, deteccion.bbox)

        filas, columnas = linear_sum_assignment(costo)

        emparejados: list[tuple[int, Detection]] = []
        tracks_usados: set[int] = set()
        dets_usadas: set[int] = set()
        for fila, columna in zip(filas, columnas, strict=True):
            # El algoritmo devuelve una asignacion completa: hay que rechazar a mano
            # las parejas cuyo solape es demasiado bajo para ser creibles.
            if 1.0 - costo[fila, columna] < umbral_iou:
                continue
            emparejados.append((indices_track[fila], detecciones[columna]))
            tracks_usados.add(indices_track[fila])
            dets_usadas.add(columna)

        libres_det = [d for i, d in enumerate(detecciones) if i not in dets_usadas]
        libres_track = [i for i in indices_track if i not in tracks_usados]
        return emparejados, libres_det, libres_track


__all__ = ["IouTracker", "iou"]
