"""Contratos estructurales entre las etapas del pipeline.

Se declaran como :class:`typing.Protocol` y no como clases base abstractas: una
implementación cumple el contrato por tener la *forma* correcta, sin heredar de
nada. Eso mantiene a las clases concretas libres de acoplamiento al framework y
hace triviales los dobles de prueba — una clase chica que devuelve detecciones
fijas satisface :class:`IDetector` sin importar nada de este módulo.

El orquestador depende únicamente de estos nombres. Agregar un tercer segmentador
de pretil significa escribir una clase con la forma de :class:`IBermSegmenter` y
registrarla en la factory; no cambia ni una línea del orquestador.

Varios protocolos declaran ``reset()``. El material de entrada está editado, y un
corte invalida las identidades del tracker, la calibración del plano de suelo y
cualquier filtro temporal. El orquestador llama a ``reset()`` en cada corte, así
que arrastrar estado a través de un corte es una violación del contrato, no un
detalle de implementación.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from bermguard.core.types import (
    BermPixels,
    BermProfile,
    Detection,
    ImageBGR,
    RiskLevel,
)


class IPreprocessor(Protocol):
    """Acondiciona un frame antes de la inferencia."""

    def apply(self, frame: ImageBGR) -> ImageBGR:
        """Devuelve una copia acondicionada de ``frame``, con igual forma y dtype."""
        ...


class IDetector(Protocol):
    """Localiza maquinaria pesada en un frame."""

    @property
    def name(self) -> str:
        """Identificador legible, se registra en los artefactos para trazabilidad."""
        ...

    def detect(self, frame: ImageBGR) -> Sequence[Detection]:
        """Devuelve las detecciones de ``frame``, con ``track_id`` en ``-1``.

        Asignar identidad es tarea del tracker. Un detector que inventara
        identidades haría imposible evaluar ambas etapas por separado.
        """
        ...


class ITracker(Protocol):
    """Asocia detecciones entre frames y les da identidad persistente."""

    def update(
        self, detections: Sequence[Detection], frame_index: int
    ) -> Sequence[Detection]:
        """Devuelve ``detections`` con el ``track_id`` asignado.

        Puede devolver cajas que no venían en ``detections``, cuando arrastra un
        track a través de una oclusión corta, y puede descartar detecciones que
        considere espurias.
        """
        ...

    def reset(self) -> None:
        """Descarta todo el estado de tracks. Se llama en cada corte de escena."""
        ...


class IBermSegmenter(Protocol):
    """Localiza el pretil de seguridad en el espacio imagen.

    Deliberadamente no dice nada sobre metros. Ver :class:`IHeightEstimator`.
    """

    @property
    def name(self) -> str:
        ...

    def segment(self, frame: ImageBGR) -> BermPixels | None:
        """Devuelve cresta y base del pretil por columna, o ``None`` si no hay.

        ``None`` significa «no hay pretil en este frame», que es un resultado
        legítimo y se contabiliza como dropout en el benchmark. No es un error.
        """
        ...

    def reset(self) -> None:
        """Descarta el estado de suavizado temporal. Se llama en cada corte."""
        ...


class IHeightEstimator(Protocol):
    """Convierte geometría en píxeles a altura métrica con banda de incertidumbre.

    Es la única etapa que razona sobre escala, proyección y unidades. Opera sobre
    arrays y dataclasses puras, así que se puede testear sin video, sin modelo y
    sin GPU.
    """

    def estimate(
        self,
        pixels: BermPixels,
        detections: Sequence[Detection],
        frame_index: int,
    ) -> BermProfile | None:
        """Devuelve ``pixels`` enriquecido con altura métrica, o ``None``.

        ``detections`` aporta el ancla métrica: máquinas de dimensiones nominales
        conocidas observadas a profundidades conocidas de la imagen. ``None``
        significa que no se pudo anclar escala en este frame — se reporta como
        tal, en vez de adivinarla.
        """
        ...

    def reset(self) -> None:
        """Descarta las estimaciones acumuladas de plano de suelo y escala."""
        ...


class IProximityAnalyzer(Protocol):
    """Asigna un nivel de riesgo a cada máquina trackeada."""

    def evaluate(
        self, detections: Sequence[Detection], frame_index: int
    ) -> Mapping[int, RiskLevel]:
        """Devuelve el nivel de riesgo por ``track_id``.

        Se espera que las implementaciones apliquen histéresis: un nivel que
        cambia en cada frame porque la distancia oscila alrededor del umbral
        produce fatiga de alarma, y el enunciado penaliza explícitamente el
        parpadeo visual.
        """
        ...

    def reset(self) -> None:
        """Descarta los contadores de histéresis y la calibración."""
        ...
