"""Coordinación del pipeline: el único módulo que conoce el orden de las etapas.

Recibe las implementaciones ya construidas —no las construye— y depende sólo de
los `Protocol` de ``core.interfaces``. Esa inversión de dependencias es lo que
permite que agregar un método nuevo se resuelva en la factory sin tocar una línea
de acá, y lo que hace que la lógica del bucle sea testeable con dobles de prueba,
sin GPU, sin modelo y sin video.

Las etapas opcionales se declaran como tales. Durante el desarrollo el pipeline
corre extremo a extremo con sólo un detector, y las etapas ausentes producen
ausencia de dato en lugar de un valor inventado.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from bermguard.core.config import PipelineConfig
from bermguard.core.device import describe_runtime
from bermguard.core.interfaces import (
    IBermSegmenter,
    IDetector,
    IHeightEstimator,
    IPreprocessor,
    IProximityAnalyzer,
    ITracker,
)
from bermguard.core.types import (
    Detection,
    FrameResult,
    LightingCondition,
    RiskLevel,
)
from bermguard.io.artifacts import (
    ResolutionInfo,
    TimingInfo,
    VideoMetadata,
    code_version,
    write_csv,
    write_json,
)
from bermguard.io.video_reader import VideoReader
from bermguard.io.video_writer import VideoWriter
from bermguard.pipeline.osd import OsdRenderer
from bermguard.vision.lighting import LightingThresholds, classify_luma, mean_luma
from bermguard.vision.shots import ShotDetector

logger = logging.getLogger(__name__)

_ORDEN_RIESGO: dict[RiskLevel, int] = {
    RiskLevel.SAFE: 0,
    RiskLevel.CAUTION: 1,
    RiskLevel.CRITICAL: 2,
}


class Orchestrator:
    """Ejecuta el pipeline completo sobre un video."""

    def __init__(
        self,
        config: PipelineConfig,
        detector: IDetector,
        device: str,
        preprocessor: IPreprocessor | None = None,
        tracker: ITracker | None = None,
        berm_segmenter: IBermSegmenter | None = None,
        height_estimator: IHeightEstimator | None = None,
        proximity: IProximityAnalyzer | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._device = device
        self._preprocessor = preprocessor
        self._tracker = tracker
        self._berm_segmenter = berm_segmenter
        self._height_estimator = height_estimator
        self._proximity = proximity

        self._osd = OsdRenderer(draw_hud=config.output.draw_hud)
        self._umbrales_luz = LightingThresholds(
            night_max=config.lighting.night_max, day_min=config.lighting.day_min
        )

    def process_video(
        self, video: Path, output_dir: Path, max_frames: int | None = None
    ) -> VideoMetadata:
        """Procesa un video y escribe sus artefactos en ``output_dir``.

        Args:
            video: Archivo de entrada.
            output_dir: Subdirectorio propio de este video y método.
            max_frames: Tope opcional de frames, para iteración rápida.

        Returns:
            Los metadatos del procesamiento, ya escritos en disco.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        detector_de_tomas = ShotDetector(
            luma_delta=self._config.shots.luma_delta,
            min_length=self._config.shots.min_length,
        )

        latencias_por_etapa: Counter[str] = Counter()
        latencias_frame: list[float] = []
        conteo_luz: Counter[LightingCondition] = Counter()
        riesgo_previo: dict[int, RiskLevel] = {}
        alertas = 0
        coberturas: list[float] = []
        frames_sin_pretil = 0
        filas_perfil: list[dict[str, object]] = []
        filas_eventos: list[dict[str, object]] = []
        avisos: list[str] = []

        inicio = time.perf_counter()

        with VideoReader(video) as reader:
            info = reader.info
            salida = output_dir / f"{video.stem}_osd.mp4"
            with VideoWriter(
                salida, info.fps, info.resolution, self._config.output.fourcc
            ) as writer:
                for index, timestamp, frame in reader.frames(max_frames=max_frames):
                    t_frame = time.perf_counter()
                    etapas: dict[str, float] = {}

                    with _cronometro(etapas, "shot"):
                        toma_nueva = detector_de_tomas.update(frame)
                        if toma_nueva and index > 0:
                            self._reiniciar_estado()
                            riesgo_previo.clear()

                    with _cronometro(etapas, "lighting"):
                        luz = classify_luma(mean_luma(frame), self._umbrales_luz)
                    conteo_luz[luz] += 1

                    with _cronometro(etapas, "preprocess"):
                        analizado = (
                            self._preprocessor.apply(frame)
                            if self._preprocessor is not None
                            else frame
                        )

                    with _cronometro(etapas, "detect"):
                        detecciones = self._detector.detect(analizado)

                    with _cronometro(etapas, "track"):
                        if self._tracker is not None:
                            detecciones = self._tracker.update(detecciones, index)

                    with _cronometro(etapas, "berm"):
                        pixeles = (
                            self._berm_segmenter.segment(analizado)
                            if self._berm_segmenter is not None
                            else None
                        )

                    with _cronometro(etapas, "height"):
                        perfil = (
                            self._height_estimator.estimate(pixeles, detecciones, index)
                            if self._height_estimator is not None and pixeles is not None
                            else None
                        )

                    with _cronometro(etapas, "proximity"):
                        riesgos = (
                            dict(self._proximity.evaluate(detecciones, index))
                            if self._proximity is not None
                            else {}
                        )

                    alertas += self._contar_escaladas(riesgos, riesgo_previo)

                    if self._berm_segmenter is not None:
                        if pixeles is None:
                            frames_sin_pretil += 1
                        else:
                            coberturas.append(pixeles.coverage)

                    resultado = FrameResult(
                        index=index,
                        timestamp_s=timestamp,
                        detections=detecciones,
                        berm=perfil,
                        risk_by_track=riesgos,
                        lighting=luz,
                        stage_latency_ms=etapas,
                    )

                    with _cronometro(etapas, "render"):
                        anotado = self._osd.render(frame, resultado)
                    writer.write(anotado)

                    filas_perfil.append(self._fila_perfil(resultado, detector_de_tomas.shot_index))
                    filas_eventos.extend(
                        self._filas_eventos(resultado, detector_de_tomas.shot_index)
                    )

                    for etapa, ms in etapas.items():
                        latencias_por_etapa[etapa] += ms
                    latencias_frame.append((time.perf_counter() - t_frame) * 1000.0)

                frames = writer.frames_written

        transcurrido = time.perf_counter() - inicio
        if frames == 0:
            raise RuntimeError(f"No se escribio ningun frame para {video}")

        if self._tracker is None:
            avisos.append(
                "Sin tracker: las detecciones no tienen identidad persistente y el "
                "conteo de alertas no es representativo."
            )
        if self._berm_segmenter is None:
            avisos.append("Sin segmentador de pretil: no hay medicion de altura.")

        metadatos = VideoMetadata(
            video=video.name,
            method=self._config.method,
            method_name=self._config.name,
            frames_processed=frames,
            resolution=ResolutionInfo(
                source=info.resolution,
                analyzed=(self._config.detector.image_size,) * 2,
                output=info.resolution,
            ),
            timing=TimingInfo(
                average_fps=frames / transcurrido,
                ms_per_frame=sum(latencias_frame) / frames,
                ms_per_frame_p95=_percentil(latencias_frame, 95),
                stage_ms={k: v / frames for k, v in sorted(latencias_por_etapa.items())},
                total_seconds=transcurrido,
            ),
            proximity_alerts=alertas,
            shots_detected=detector_de_tomas.shot_index + 1,
            lighting_distribution={
                condicion.value: conteo_luz[condicion] / frames for condicion in LightingCondition
            },
            berm_coverage_mean=(sum(coberturas) / len(coberturas) if coberturas else None),
            berm_dropout_rate=(
                frames_sin_pretil / frames if self._berm_segmenter is not None else None
            ),
            runtime=describe_runtime(self._device),
            code_version=code_version(),
            warnings=avisos,
        )

        write_json(metadatos, output_dir / "metadata.json")
        write_csv(filas_perfil, output_dir / "berm_profile.csv")
        write_csv(filas_eventos, output_dir / "proximity_events.csv")

        logger.info(
            "%s: %d frames, %.1f fps, %d tomas, %d alertas",
            video.name,
            frames,
            metadatos.timing.average_fps,
            metadatos.shots_detected,
            alertas,
        )
        return metadatos

    # --- Auxiliares -----------------------------------------------------------

    def _reiniciar_estado(self) -> None:
        """Descarta el estado temporal en cada corte de toma (ADR 0003)."""
        for componente in (
            self._tracker,
            self._berm_segmenter,
            self._height_estimator,
            self._proximity,
        ):
            if componente is not None:
                componente.reset()

    @staticmethod
    def _contar_escaladas(riesgos: dict[int, RiskLevel], previos: dict[int, RiskLevel]) -> int:
        """Cuenta transiciones hacia un nivel de riesgo mayor.

        Se cuentan eventos y no frames: un equipo que permanece diez segundos en
        zona crítica es una alerta, no doscientas. Contar frames haría la cifra
        proporcional a la tasa de muestreo e incomparable entre videos.
        """
        escaladas = 0
        for track_id, nivel in riesgos.items():
            anterior = previos.get(track_id, RiskLevel.SAFE)
            if _ORDEN_RIESGO[nivel] > _ORDEN_RIESGO[anterior]:
                escaladas += 1
            previos[track_id] = nivel
        return escaladas

    @staticmethod
    def _fila_perfil(resultado: FrameResult, toma: int) -> dict[str, object]:
        perfil = resultado.berm
        fila: dict[str, object] = {
            "frame": resultado.index,
            "timestamp_s": round(resultado.timestamp_s, 4),
            "shot": toma,
            "lighting": resultado.lighting.value,
            "detections": len(resultado.detections),
        }
        if perfil is None:
            fila |= {"coverage": "", "height_m_median": "", "height_ratio_median": ""}
        else:
            fila |= {
                "coverage": round(perfil.pixels.coverage, 4),
                "height_m_median": _mediana_sin_nan(perfil.height_m),
                "height_ratio_median": _mediana_sin_nan(perfil.height_ratio),
            }
        return fila

    @staticmethod
    def _filas_eventos(resultado: FrameResult, toma: int) -> list[dict[str, object]]:
        return [
            {
                "frame": resultado.index,
                "timestamp_s": round(resultado.timestamp_s, 4),
                "shot": toma,
                "track_id": deteccion.track_id,
                "vehicle_class": deteccion.vehicle_class.value,
                "risk": resultado.risk_by_track[deteccion.track_id].value,
                "ground_x_px": round(deteccion.bbox.ground_point[0], 1),
                "ground_y_px": round(deteccion.bbox.ground_point[1], 1),
            }
            for deteccion in resultado.detections
            if deteccion.track_id in resultado.risk_by_track
        ]


@contextmanager
def _cronometro(destino: dict[str, float], etapa: str) -> Iterator[None]:
    """Acumula en ``destino`` la duración del bloque, en milisegundos."""
    inicio = time.perf_counter()
    try:
        yield
    finally:
        destino[etapa] = destino.get(etapa, 0.0) + (time.perf_counter() - inicio) * 1000.0


def _percentil(valores: Sequence[float], percentil: float) -> float:
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    indice = min(len(ordenados) - 1, int(len(ordenados) * percentil / 100.0))
    return ordenados[indice]


def _mediana_sin_nan(serie: object) -> float | str:
    import numpy as np

    arreglo = np.asarray(serie, dtype=float)
    validos = arreglo[~np.isnan(arreglo)]
    return round(float(np.median(validos)), 4) if validos.size else ""


__all__ = ["Orchestrator", "Detection"]
