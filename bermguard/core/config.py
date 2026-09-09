"""Configuración del pipeline, cargada desde YAML y validada.

Ningún umbral vive en el código. Todo lo que un evaluador podría querer ajustar
—confianza del detector, umbrales de proximidad, cortes de luminancia— está en
``configs/`` y llega hasta acá validado por pydantic.

La validación importa más de lo que parece en un entregable que se ejecuta sin
supervisión: un YAML con una clave mal escrita fallaría en silencio con un valor
por defecto inesperado, y el error aparecería veinte minutos después como un
resultado raro en vez de como un mensaje claro al arrancar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bermguard.core.exceptions import ConfigError


class _Base(BaseModel):
    """Base común: prohíbe claves desconocidas.

    Sin ``extra="forbid"``, un ``confidance: 0.5`` mal escrito se ignoraría y el
    pipeline correría con el valor por defecto sin avisar.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class DetectorConfig(_Base):
    weights: Path = Path("weights/yolo11s.pt")
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    image_size: int = Field(640, ge=32, le=4096)
    half_precision: bool = True


class LightingConfig(_Base):
    night_max: float = Field(70.0, ge=0.0, le=255.0)
    day_min: float = Field(110.0, ge=0.0, le=255.0)


class ShotConfig(_Base):
    luma_delta: float = Field(20.0, gt=0.0, le=255.0)
    """Salto de luminancia media entre frames consecutivos que marca un corte.

    Calibrado sobre el material de muestra: las transiciones graduales se mueven
    unas pocas unidades por frame, mientras que los cortes duros superan 20.
    """

    min_length: int = Field(6, ge=1)
    """Frames mínimos de una toma. Evita que un destello o un frame corrupto
    generen una toma espuria y disparen un reinicio de estado innecesario."""


class ProximityConfig(_Base):
    caution_m: float = Field(20.0, gt=0.0)
    critical_m: float = Field(10.0, gt=0.0)
    frames_to_escalate: int = Field(3, ge=1)
    frames_to_deescalate: int = Field(10, ge=1)
    """Histéresis asimétrica: escalar rápido es seguro, desescalar lento también.
    Sin esto el color parpadea cuando la distancia oscila en torno al umbral."""


class OutputConfig(_Base):
    fourcc: str = Field("mp4v", min_length=4, max_length=4)
    draw_hud: bool = True


class PipelineConfig(_Base):
    """Configuración completa de un método."""

    method: int = Field(..., ge=1)
    name: str
    description: str = ""

    detector: DetectorConfig = DetectorConfig()
    lighting: LightingConfig = LightingConfig()
    shots: ShotConfig = ShotConfig()
    proximity: ProximityConfig = ProximityConfig()
    output: OutputConfig = OutputConfig()

    def model_post_init(self, _context: Any) -> None:
        if self.lighting.night_max >= self.lighting.day_min:
            raise ValueError(
                f"lighting.night_max ({self.lighting.night_max}) debe ser menor que "
                f"lighting.day_min ({self.lighting.day_min})"
            )
        if self.proximity.critical_m >= self.proximity.caution_m:
            raise ValueError(
                f"proximity.critical_m ({self.proximity.critical_m}) debe ser menor "
                f"que proximity.caution_m ({self.proximity.caution_m})"
            )


def load_config(path: Path) -> PipelineConfig:
    """Carga y valida un archivo de configuración.

    Raises:
        ConfigError: Si el archivo no existe, no es YAML válido, o su contenido
            no satisface el esquema.
    """
    if not path.exists():
        raise ConfigError(f"No existe el archivo de configuracion: {path}")

    try:
        crudo = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalido en {path}: {exc}") from exc

    if not isinstance(crudo, dict):
        raise ConfigError(f"{path} debe contener un mapeo en su raiz")

    try:
        return PipelineConfig(**crudo)
    except ValidationError as exc:
        raise ConfigError(f"Configuracion invalida en {path}:\n{exc}") from exc


def config_for_method(method: int, configs_dir: Path = Path("configs")) -> PipelineConfig:
    """Resuelve la configuración de un método por convención de nombre."""
    ruta = configs_dir / f"method_{method}.yaml"
    config = load_config(ruta)
    if config.method != method:
        raise ConfigError(
            f"{ruta} declara method={config.method} pero se solicito el metodo {method}"
        )
    return config


DeviceOption = Literal["auto", "cuda", "cpu"]
