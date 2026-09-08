"""Selección del dispositivo de cómputo.

Implementa la mitad en tiempo de ejecución del ADR 0001: CUDA es el stack
objetivo, pero el comando de referencia del enunciado no pasa ``--gpus``, así que
un contenedor puede perfectamente arrancar sin acceso a GPU. El pipeline degrada
a CPU en vez de fallar, y lo deja dicho en voz alta.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DeviceSpec = Literal["auto", "cuda", "cpu"]


def resolve_device(requested: DeviceSpec = "auto") -> str:
    """Resuelve un dispositivo pedido contra lo que la máquina realmente ofrece.

    Args:
        requested: ``"cuda"`` y ``"cpu"`` se respetan tal cual; ``"auto"`` elige
            CUDA cuando está disponible.

    Returns:
        ``"cuda"`` o ``"cpu"``.

    Note:
        Pedir ``"cuda"`` en una máquina que no la tiene degrada a CPU con una
        advertencia, en vez de lanzar excepción. Un lote que produce artefactos
        lentamente vale más que uno que no produce ninguno, y la advertencia deja
        constancia de que la degradación ocurrió.
    """
    available = torch.cuda.is_available()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda" and not available:
        logger.warning(
            "Se pidió CUDA pero no hay dispositivo visible. Se continúa en CPU. "
            "Si este contenedor arrancó sin '--gpus all', esa es la razón."
        )
        return "cpu"

    if not available:
        logger.warning(
            "No hay dispositivo CUDA visible; ejecutando en CPU. El rendimiento "
            "será sustancialmente menor que las cifras reportadas para GPU."
        )
        return "cpu"

    logger.info("Usando dispositivo CUDA: %s", torch.cuda.get_device_name(0))
    return "cuda"


def describe_runtime(device: str) -> dict[str, str | bool]:
    """Devuelve los datos del entorno que vale la pena registrar en los metadatos.

    Fijar esto en cada artefacto es lo que hace reproducible una cifra de
    benchmark: un FPS sin el dispositivo y las versiones de librerías que lo
    produjeron no se puede comparar contra nada.
    """
    info: dict[str, str | bool] = {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or "unknown"
    return info
