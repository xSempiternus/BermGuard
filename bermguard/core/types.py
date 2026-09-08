"""Tipos inmutables del dominio.

Todo lo que cruza una frontera entre módulos es uno de estos tipos. No tienen
comportamiento más allá de geometría derivada, son inmutables, y no saben nada
de OpenCV, PyTorch ni matplotlib. Eso es lo que hace que ``analytics`` sea
testeable sin GPU, sin modelo y sin archivo de video.

Las dataclasses inmutables que contienen arrays de NumPy declaran ``eq=False``:
el ``==`` de NumPy devuelve un array, no un booleano, así que el ``__eq__``
generado automáticamente reventaría al usarse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

# --- Alias de arrays ---------------------------------------------------------
# Documentar el dtype a nivel de tipos: un float64 llegando donde se espera un
# uint8 es un bug silencioso clásico en código de visión.

ImageBGR: TypeAlias = npt.NDArray[np.uint8]
"""Frame en el orden de canales nativo de OpenCV (BGR), forma ``(H, W, 3)``."""

Mask: TypeAlias = npt.NDArray[np.bool_]
"""Máscara binaria, forma ``(H, W)``."""

FloatArray: TypeAlias = npt.NDArray[np.float32]
"""Señal por columna o por frame. ``NaN`` marca «acá no hay medición válida»."""


# --- Enumeraciones -----------------------------------------------------------


class VehicleClass(str, Enum):
    """Categorías de maquinaria relevantes para la seguridad en el botadero.

    Hereda de ``str`` para que los valores serialicen directo a JSON sin
    necesidad de un encoder propio.
    """

    CAEX = "caex"
    """Camión de extracción (Komatsu 930E, Caterpillar 793 y similares, >300 t)."""

    BULLDOZER = "bulldozer"
    """Tractor de oruga que empuja material hacia el borde."""

    UNKNOWN = "unknown"
    """Detectado como maquinaria pesada, pero sin subclasificación confiable."""


class RiskLevel(str, Enum):
    """Nivel de riesgo por proximidad, define el color de la detección en el OSD."""

    SAFE = "safe"
    CAUTION = "caution"
    CRITICAL = "critical"


class LightingCondition(str, Enum):
    """Régimen de iluminación de un frame.

    Se asigna por frame y no por archivo: el material de muestra transiciona
    entre noche y pleno día dentro de un mismo clip de diez segundos, así que una
    etiqueta a nivel de archivo no significaría nada. Las métricas del benchmark
    se estratifican por este valor.
    """

    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"


# --- Geometría ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BBox:
    """Caja delimitadora alineada a los ejes, en píxeles, ``(x1, y1, x2, y2)``.

    El origen es la esquina superior izquierda y la ``y`` crece hacia abajo,
    siguiendo la convención de OpenCV.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def ground_point(self) -> tuple[float, float]:
        """Centro del borde inferior: donde la máquina toca el suelo.

        Es el único punto de una detección que puede proyectarse por una
        homografía del plano de suelo. Una homografía relaciona planos, así que
        pasarle un punto elevado (el techo del camión, el centro de la caja)
        entrega una posición en el mundo simplemente incorrecta. Toda distancia
        métrica de este pipeline empieza acá.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True, eq=False)
class Detection:
    """Una máquina observada en un frame."""

    track_id: int
    """Identidad asignada por el tracker. Vale ``-1`` antes de trackear."""

    vehicle_class: VehicleClass
    bbox: BBox
    confidence: float

    mask: Mask | None = None
    """Máscara de instancia, cuando el detector la produce. ``None`` si sólo da cajas."""

    world_xy_m: tuple[float, float] | None = None
    """Posición en el plano cenital, en metros. ``None`` hasta que la calibración funcione."""


@dataclass(frozen=True, slots=True, eq=False)
class BermPixels:
    """El pretil localizado en el espacio imagen, antes de toda interpretación métrica.

    Esto es todo aquello de lo que un segmentador es responsable: *dónde* está el
    pretil, en píxeles. Convertirlo a metros es metrología y le corresponde a un
    estimador de altura. Separar ambas cosas permite cambiar de segmentador sin
    tocar la cadena de medición, y testear la cadena de medición sobre arrays
    sintéticos, sin imagen, sin modelo y sin GPU.

    Ambos arrays tienen largo ``W`` y se indexan por columna de la imagen, de modo
    que ``crest_y_px[x]`` y ``base_y_px[x]`` describen la misma franja vertical.
    ``NaN`` marca las columnas sin pretil: un hueco explícito, nunca un cero que
    pasaría en silencio como medición válida.
    """

    crest_y_px: FloatArray
    """Fila de la cresta del pretil, por columna."""

    base_y_px: FloatArray
    """Fila donde el pretil se encuentra con la rasante transitable, por columna."""

    coverage: float
    """Fracción de columnas con medición válida, en ``[0, 1]``."""

    confidence: float
    """Confianza propia del segmentador en este perfil, en ``[0, 1]``."""


@dataclass(frozen=True, slots=True, eq=False)
class BermProfile:
    """Un pretil medido: geometría en píxeles más su interpretación métrica.

    Los valores métricos se reportan como intervalo, no como punto. Sin
    calibración de cámara la escala se ancla en las dimensiones nominales de una
    máquina detectada, y ese supuesto arrastra incertidumbre real; publicar un
    número seco exageraría lo que el método puede saber.
    """

    pixels: BermPixels

    height_m: FloatArray
    """Altura por columna, en metros. Todo ``NaN`` si no se pudo anclar la escala."""

    height_m_lo: FloatArray
    height_m_hi: FloatArray
    """Límites del intervalo de credibilidad al 90 %, propagado desde el ruido de
    la caja, la dispersión de la dimensión nominal y el ajuste del plano de suelo."""

    height_ratio: FloatArray
    """Altura como múltiplo del radio de rueda de referencia, por columna.

    Es la magnitud relevante para el cumplimiento normativo, y además la mejor
    condicionada: es un cociente entre dos longitudes medidas en la misma imagen,
    así que el sesgo global de escala —el que viene de asumir una dimensión
    nominal— se cancela en buena medida. La normativa define la altura mínima del
    pretil exactamente en estos términos.
    """

    scale_confidence: float
    """Confianza en el modelo píxel-metro en sí, en ``[0, 1]``.

    Distinta de ``pixels.confidence``: el pretil puede estar perfectamente
    segmentado mientras el ancla de escala es débil, por ejemplo cuando no hay
    ningún vehículo visible.
    """


@dataclass(frozen=True, slots=True)
class Shot:
    """Un tramo contiguo de frames que comparte una misma toma de cámara.

    El material de muestra está editado: ``video_01`` corta a otra posición de
    cámara dos veces en diez segundos. Los tracks, la calibración del plano de
    suelo y la serie de altura del pretil son válidos sólo dentro de una toma, así
    que el pipeline los reinicia en cada corte en vez de producir una curva
    continua que mezcla escenas sin relación.
    """

    start_frame: int
    end_frame: int
    """Exclusivo, de modo que ``end_frame - start_frame`` es el largo en frames."""

    lighting: LightingCondition

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


# --- Resultados por frame y por video ----------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class FrameResult:
    """Todo lo que el pipeline concluyó sobre un frame."""

    index: int
    timestamp_s: float
    detections: Sequence[Detection]
    berm: BermProfile | None
    risk_by_track: Mapping[int, RiskLevel]
    lighting: LightingCondition

    stage_latency_ms: Mapping[str, float] = field(default_factory=dict)
    """Tiempo de reloj por etapa. Desglosado para que el benchmark pueda nombrar
    el cuello de botella en vez de reportar un FPS único y opaco."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Datos del contenedor de un video, leídos antes de decodificar.

    Nada de esto puede suponerse constante dentro del directorio de entrada: sólo
    el set de muestra ya mezcla 1920x1080 a 30 fps con 1280x720 a 24 fps.
    """

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    """Según lo declara el contenedor. Se trata como pista: es habitual que esté
    mal o ausente, así que la decodificación se detiene en la primera lectura
    fallida y no al llegar a este conteo."""

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)
