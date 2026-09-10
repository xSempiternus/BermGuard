"""Escritura de los artefactos no visuales: metadatos y series de datos.

El enunciado exige cuatro campos en ``metadata.json``: FPS promedio, tiempo de
procesamiento por frame, conteo de alertas de proximidad y resolución analizada.
Se emiten esos cuatro y algunos más, con un criterio: **un número de rendimiento
sin el contexto que lo produjo no se puede comparar contra nada**, así que cada
archivo registra también el dispositivo, las versiones de las librerías y el commit
del código que lo generó.

Los esquemas se declaran con pydantic en vez de armar diccionarios a mano, para que
un cambio en el pipeline que rompa el contrato de salida falle al escribir y no
semanas después, cuando alguien intente leerlo.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResolutionInfo(_Base):
    """Las tres resoluciones que intervienen, que no tienen por qué coincidir."""

    source: tuple[int, int]
    """La del archivo de entrada."""

    analyzed: tuple[int, int]
    """A la que se ejecutó la inferencia. Es el campo que pide el enunciado."""

    output: tuple[int, int]
    """La del video anotado que se escribe."""


class TimingInfo(_Base):
    average_fps: float
    """Frames procesados por segundo de reloj, extremo a extremo."""

    ms_per_frame: float
    ms_per_frame_p95: float
    """El percentil 95 acompaña a la media porque un pipeline de video se percibe
    por sus peores frames, no por su promedio."""

    stage_ms: dict[str, float] = Field(default_factory=dict)
    """Desglose por etapa, para poder nombrar el cuello de botella."""

    total_seconds: float


class VideoMetadata(_Base):
    """Metadatos de un video procesado por un método."""

    video: str
    method: int
    method_name: str

    frames_processed: int
    resolution: ResolutionInfo
    timing: TimingInfo

    proximity_alerts: int
    """Conteo de alertas de proximidad, según pide el enunciado.

    Se cuentan **escaladas de nivel**, no frames en riesgo: un equipo que
    permanece diez segundos en zona crítica es un evento, no doscientas alertas.
    Contar frames inflaría la cifra en proporción a la tasa de muestreo y la haría
    incomparable entre videos de distinto fps.
    """

    shots_detected: int
    lighting_distribution: dict[str, float]
    berm_coverage_mean: float | None = None
    berm_dropout_rate: float | None = None
    """Fracción de frames sin pretil detectado. Mide robustez en condiciones
    adversas mejor que el promedio del IoU, que ignora los frames perdidos."""

    berm_crest_jitter_px: float | None = None
    """Desviación estándar del cambio de la cresta entre frames consecutivos, en
    píxeles, medida dentro de cada toma.

    Es la métrica que el enunciado pide sin nombrarla: cuantifica el «parpadeo» del
    perfil. Y es la única medida de calidad de la segmentación que no requiere
    ground truth, a diferencia de la cobertura, que resultó estar dominada por el
    umbral de validación y no por la escena (ADR 0006).

    Un pretil físico cambia lentísimo, de modo que todo jitter por encima del ruido
    de cuantización es error de medición, no señal."""

    runtime: dict[str, str | bool] = Field(default_factory=dict)
    code_version: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RunMetadata(_Base):
    """Metadatos globales de una ejecución sobre un directorio completo."""

    command: str
    input_dir: str
    output_dir: str
    methods: list[int]
    videos_found: int
    videos_processed: int
    videos_failed: list[dict[str, str]] = Field(default_factory=list)
    """Cada fallo con su motivo. Un video corrupto no puede tumbar el lote, pero
    tampoco puede desaparecer sin dejar rastro."""

    total_seconds: float
    runtime: dict[str, str | bool] = Field(default_factory=dict)
    code_version: str | None = None


def code_version() -> str | None:
    """Devuelve el commit actual, o ``None`` fuera de un repositorio.

    Dentro del contenedor no hay historial de git, así que la ausencia es un caso
    normal y no un error.
    """
    try:
        salida = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return salida.stdout.strip() or None


def write_json(model: BaseModel, path: Path) -> None:
    """Serializa un modelo a JSON con formato legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
    logger.debug("Escrito %s", path)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Escribe una serie de datos como CSV.

    Cada gráfico se acompaña de su CSV a propósito: permite a quien evalúa
    reproducir la figura desde los datos en vez de confiar en la imagen. Es la
    forma más directa de sostener que no se inventó nada.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.debug("Escritas %d filas en %s", len(rows), path)


def read_json(path: Path) -> dict[str, object]:
    """Lee un JSON de artefactos, para el comparativo del benchmark."""
    return json.loads(path.read_text(encoding="utf-8"))
