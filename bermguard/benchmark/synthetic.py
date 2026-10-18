"""Escenas sintéticas con la cresta del pretil conocida.

El material no trae el perfil real del pretil, así que sobre él sólo se puede
medir estabilidad. Una escena sintética invierte el problema: la cresta se dibuja
en una fila conocida y el error de cada método se puede medir en píxeles.

La escena reproduce los tres hechos medidos en ``docs/analisis_material.md`` que
hacen difícil el problema, no una imagen realista:

1. **Contraste bajo en la cresta**: un escalón suave de brillo, de pocos niveles.
2. **Huellas de neumático en el primer plano**: tramos horizontales cortos, con
   más contraste que la cresta, a filas aleatorias bajo el pretil.
3. **Ruido de textura** sobre toda la imagen.

Con el ruido y las huellas como parámetros, el benchmark recorre la dificultad y
muestra dónde se rompe cada método. No reemplaza al ground truth del material
real: mide la formulación en condiciones controladas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bermguard.core.types import FloatArray, ImageBGR


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """Parámetros de una escena sintética."""

    height: int = 360
    width: int = 640
    crest_row: float = 170.0
    """Fila media de la cresta."""
    crest_amplitude: float = 12.0
    """Amplitud de la ondulación de la cresta, en píxeles."""
    crest_contrast: float = 18.0
    """Salto de brillo en la cresta, en niveles de gris (de 255)."""
    noise_sigma: float = 8.0
    """Desviación estándar del ruido de textura."""
    tracks: int = 6
    """Cantidad de huellas de neumático en el primer plano."""
    track_contrast: float = 30.0
    """Salto de brillo de cada huella. Por encima del de la cresta, como en el material."""


@dataclass(frozen=True, slots=True)
class Scene:
    """Una escena y su verdad de terreno."""

    image: ImageBGR
    crest_y_px: FloatArray
    """Fila verdadera de la cresta por columna."""


def make_scene(spec: SceneSpec, seed: int = 0, phase: float = 0.0) -> Scene:
    """Genera una escena con la cresta en una posición conocida.

    Args:
        spec: Parámetros de la escena.
        seed: Semilla del ruido y de las huellas. La misma semilla da la misma escena.
        phase: Fase de la ondulación de la cresta. Variarla entre frames simula una
            secuencia en la que el pretil se mueve poco, para medir jitter.
    """
    rng = np.random.default_rng(seed)
    alto, ancho = spec.height, spec.width
    x = np.arange(ancho, dtype=np.float32)
    cresta = (
        spec.crest_row
        + spec.crest_amplitude * np.sin(2 * np.pi * x / ancho * 1.5 + phase)
        + 0.4 * spec.crest_amplitude * np.sin(2 * np.pi * x / ancho * 4.0 + 1.3)
    ).astype(np.float32)

    filas = np.arange(alto, dtype=np.float32)[:, None]

    # Fondo: terreno que se aclara levemente hacia el primer plano, y un escalón
    # en la cresta. El escalón es positivo hacia abajo, que es el signo que buscan
    # los segmentadores.
    imagen = 70.0 + 0.08 * filas + np.zeros((alto, ancho), dtype=np.float32)
    imagen = imagen + spec.crest_contrast * (filas > cresta[None, :])

    # Huellas: tramos cortos y horizontales bajo la cresta, más contrastados que ella.
    for _ in range(spec.tracks):
        fila = int(rng.uniform(cresta.max() + 25, alto * 0.78))
        largo = int(rng.uniform(ancho * 0.10, ancho * 0.30))
        inicio = int(rng.uniform(0, ancho - largo))
        grosor = int(rng.integers(3, 7))
        imagen[fila : fila + grosor, inicio : inicio + largo] += spec.track_contrast

    imagen = imagen + rng.normal(0.0, spec.noise_sigma, size=imagen.shape)
    gris = np.clip(imagen, 0, 255).astype(np.uint8)
    return Scene(image=np.repeat(gris[:, :, None], 3, axis=2), crest_y_px=cresta)


def crest_error_px(estimated: FloatArray, truth: FloatArray) -> dict[str, float]:
    """Error de una cresta estimada contra la verdadera, sólo en columnas medidas.

    Returns:
        ``mae`` y ``p95`` del error absoluto en píxeles, ``within_5`` (fracción de
        columnas a 5 px o menos) y ``coverage`` (fracción de columnas medidas).
    """
    medidas = np.isfinite(estimated)
    if not medidas.any():
        return {"mae": float("nan"), "p95": float("nan"), "within_5": 0.0, "coverage": 0.0}
    error = np.abs(estimated[medidas] - truth[medidas])
    return {
        "mae": float(error.mean()),
        "p95": float(np.percentile(error, 95)),
        # Las columnas sin medición cuentan como fallo: un método que se abstiene en
        # todo lo difícil no debe salir con mejor exactitud que uno que lo intenta.
        "within_5": float((error <= 5.0).sum() / truth.size),
        "coverage": float(medidas.mean()),
    }
