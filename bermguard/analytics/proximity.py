"""Evaluación de riesgo por proximidad entre equipos.

Asigna a cada máquina trackeada un nivel de riesgo según su distancia al equipo más
cercano, y **aplica histéresis** para que ese nivel no oscile.

## Por qué la histéresis no es un adorno

Si el umbral crítico está en 10 m y la distancia medida oscila entre 9.8 y 10.2 —lo
que ocurre por el ruido de la caja, no porque las máquinas se muevan—, el color del
BBOX parpadearía entre ámbar y rojo varias veces por segundo.

En un sistema de seguridad real eso produce **fatiga de alarma**: el operador aprende
que las alertas no significan nada y deja de mirarlas, con lo que el sistema empeora
la seguridad en lugar de mejorarla. El enunciado penaliza el parpadeo explícitamente.

La histéresis es **asimétrica a propósito**: escalar exige pocos frames de
confirmación y desescalar exige muchos más. Subir rápido es seguro —ante la duda, se
avisa—; bajar despacio también, porque mantener una alerta de más es barato y
retirarla antes de tiempo no lo es. Un diseño simétrico trataría los dos errores como
equivalentes, y en seguridad industrial no lo son.

## Alcance declarado

El nivel de riesgo se deriva de distancias con un error absoluto del orden de ±30 %
(ver `analytics/geometry.py`). Los umbrales por defecto son conservadores frente a
esa incertidumbre y frente a la distancia de frenado de un CAEX cargado, y son
configurables en `configs/`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bermguard.analytics.geometry import (
    GroundPlaneModel,
    fit_ground_plane,
    pairwise_distances,
)
from bermguard.core.exceptions import InsufficientDataError
from bermguard.core.types import Detection, RiskLevel

logger = logging.getLogger(__name__)

_ORDEN = {RiskLevel.SAFE: 0, RiskLevel.CAUTION: 1, RiskLevel.CRITICAL: 2}


@dataclass(slots=True)
class _EstadoTrack:
    """Nivel reportado de un track y su avance hacia un cambio de nivel."""

    nivel: RiskLevel = RiskLevel.SAFE
    candidato: RiskLevel = RiskLevel.SAFE
    frames_en_candidato: int = 0


class ProximityAnalyzer:
    """Calcula el nivel de riesgo por equipo, con histéresis.

    Satisface :class:`~bermguard.core.interfaces.IProximityAnalyzer`.

    Recibe la geometría del frame como números —la fila del horizonte y el ancho—
    y nunca la imagen. Eso mantiene el módulo testeable sin video y sin OpenCV, y
    evita el estado oculto de una llamada previa obligatoria.
    """

    def __init__(
        self,
        caution_m: float = 20.0,
        critical_m: float = 10.0,
        frames_to_escalate: int = 3,
        frames_to_deescalate: int = 10,
        horizontal_fov_deg: float = 60.0,
    ) -> None:
        """
        Args:
            caution_m: Por debajo de esta distancia el equipo pasa a precaución.
            critical_m: Por debajo de esta distancia, a alerta crítica.
            frames_to_escalate: Frames consecutivos que confirman una subida.
            frames_to_deescalate: Frames consecutivos que confirman una bajada.
            horizontal_fov_deg: Campo de visión horizontal asumido. Es el supuesto
                más débil de la cadena métrica y por eso vive en la configuración.
        """
        self._caution_m = caution_m
        self._critical_m = critical_m
        self._frames_subir = frames_to_escalate
        self._frames_bajar = frames_to_deescalate
        self._fov = horizontal_fov_deg

        self._estados: dict[int, _EstadoTrack] = {}
        self._modelo: GroundPlaneModel | None = None
        self._ultimas_distancias: dict[tuple[int, int], float] = {}

    def reset(self) -> None:
        """Descarta histéresis y calibración. Se llama en cada corte (ADR 0003).

        La calibración se descarta porque un corte cambia la cámara: conservar el
        plano de suelo de la toma anterior produciría distancias sin sentido con
        aspecto perfectamente creíble.
        """
        self._estados.clear()
        self._modelo = None
        self._ultimas_distancias.clear()

    @property
    def model(self) -> GroundPlaneModel | None:
        """Último modelo de plano ajustado, o ``None`` si aún no se pudo ajustar."""
        return self._modelo

    @property
    def last_distances(self) -> Mapping[tuple[int, int], float]:
        """Distancias del último frame evaluado, para los artefactos y gráficos."""
        return self._ultimas_distancias

    def evaluate(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        horizon_y_px: float | None = None,
        frame_width: int = 0,
    ) -> Mapping[int, RiskLevel]:
        activos = [d for d in detections if d.track_id >= 0]
        self._estados = {
            t: e for t, e in self._estados.items() if any(d.track_id == t for d in activos)
        }

        if horizon_y_px is None or frame_width <= 0 or len(activos) < 2:
            # Con un solo equipo no hay proximidad que evaluar. Se devuelve SAFE
            # explicito y no un mapa vacio: el OSD debe poder distinguir "evaluado y
            # seguro" de "no evaluado", que es la diferencia entre una caja verde y
            # una gris.
            self._ultimas_distancias = {}
            return {d.track_id: self._reportar(d.track_id, RiskLevel.SAFE) for d in activos}

        try:
            self._modelo = fit_ground_plane(activos, frame_width, horizon_y_px, self._fov)
        except InsufficientDataError:
            # Sin ancla metrica no se inventa una escala. Los niveles previos se
            # mantienen congelados: es preferible repetir la ultima evaluacion
            # conocida a emitir un veredicto sin base.
            if frame_index % 60 == 0:
                logger.debug("Frame %d sin ancla metrica; se mantiene el nivel", frame_index)
            return {
                d.track_id: self._estados.get(d.track_id, _EstadoTrack()).nivel for d in activos
            }

        self._ultimas_distancias = pairwise_distances(activos, self._modelo)

        # Cada equipo se juzga por su vecino mas cercano: lo que importa es la
        # distancia minima, no la media a todos los demas.
        mas_cercano: dict[int, float] = {}
        for (a, b), distancia in self._ultimas_distancias.items():
            mas_cercano[a] = min(mas_cercano.get(a, distancia), distancia)
            mas_cercano[b] = min(mas_cercano.get(b, distancia), distancia)

        salida: dict[int, RiskLevel] = {}
        for deteccion in activos:
            distancia = mas_cercano.get(deteccion.track_id)
            crudo = RiskLevel.SAFE if distancia is None else self._nivel_por_distancia(distancia)
            salida[deteccion.track_id] = self._reportar(deteccion.track_id, crudo)
        return salida

    # --- Auxiliares -----------------------------------------------------------

    def _nivel_por_distancia(self, distancia_m: float) -> RiskLevel:
        if distancia_m < self._critical_m:
            return RiskLevel.CRITICAL
        if distancia_m < self._caution_m:
            return RiskLevel.CAUTION
        return RiskLevel.SAFE

    def _reportar(self, track_id: int, crudo: RiskLevel) -> RiskLevel:
        """Aplica la histéresis y devuelve el nivel que debe mostrarse."""
        estado = self._estados.setdefault(track_id, _EstadoTrack())

        if crudo is estado.nivel:
            # El nivel crudo coincide con el reportado: no hay transicion pendiente.
            estado.candidato = crudo
            estado.frames_en_candidato = 0
            return estado.nivel

        if crudo is not estado.candidato:
            estado.candidato = crudo
            estado.frames_en_candidato = 1
        else:
            estado.frames_en_candidato += 1

        subiendo = _ORDEN[crudo] > _ORDEN[estado.nivel]
        requeridos = self._frames_subir if subiendo else self._frames_bajar
        if estado.frames_en_candidato >= requeridos:
            estado.nivel = crudo
            estado.frames_en_candidato = 0

        return estado.nivel
