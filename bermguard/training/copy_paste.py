"""Copy-paste augmentation de bulldozers con polígonos del propio dominio.

La clase ``bulldozer`` tiene 33 instancias de entrenamiento contra 270 CAEX, y esa
diferencia explica las dos fallas del detector (``docs/resultados.md``, sección
3.1). Los datasets públicos no sirven por brecha de dominio (ADR 0004), pero las
anotaciones del proyecto son polígonos: cada bulldozer anotado se puede recortar
con su silueta exacta y pegar en otros frames del mismo material.

Tres cuidados hacen que el pegado sea creíble para el detector:

1. **Escala por perspectiva.** Un objeto más abajo en la imagen está más cerca y
   se ve más grande. El recorte se escala con la fila donde se apoya, tomando como
   referencia el alto de los CAEX del frame destino a esa altura.
2. **Brillo del destino.** Un bulldozer diurno pegado en un frame nocturno es un
   objeto que el detector nunca verá en operación. La luminancia del recorte se
   ajusta a la de la zona donde se pega.
3. **Bordes suaves.** La máscara se difumina unos píxeles; un borde recortado a
   cuchillo es un atajo que la red puede aprender en lugar de la máquina.

Todo es determinista dada la semilla, para que el dataset aumentado se pueda
regenerar sin versionarlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from bermguard.core.types import ImageBGR

CLASE_BULLDOZER = 1


@dataclass(frozen=True, slots=True)
class Polygon:
    """Polígono anotado en píxeles absolutos."""

    cls: int
    points: npt.NDArray[np.float64]
    """Forma ``(N, 2)``, columnas ``x, y``."""

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = self.points[:, 0], self.points[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


@dataclass(frozen=True, slots=True)
class Cutout:
    """Recorte de un objeto con su máscara suave y su polígono local."""

    pixels: ImageBGR
    alpha: npt.NDArray[np.float32]
    """Máscara en ``[0, 1]``, misma forma que el recorte sin canales."""
    polygon: npt.NDArray[np.float64]
    """Polígono relativo a la esquina del recorte."""
    source_bottom_y: float
    source_height: float


def parse_polygons(text: str, width: int, height: int) -> list[Polygon]:
    """Lee etiquetas YOLO normalizadas, de caja o de polígono.

    El dataset mezcla ambos formatos: las pre-anotaciones aceptadas sin cambios
    quedaron como caja (``cls cx cy w h``) y las corregidas a mano como polígono. Una
    caja se convierte en un polígono de cuatro vértices; ignorarla dejaría la máquina
    fuera de la lista y el pegado podría taparla.
    """
    poligonos = []
    for linea in text.splitlines():
        valores = linea.split()
        if len(valores) == 5:
            cx, cy, w, h = (float(v) for v in valores[1:])
            coords = np.array(
                [
                    [cx - w / 2, cy - h / 2],
                    [cx + w / 2, cy - h / 2],
                    [cx + w / 2, cy + h / 2],
                    [cx - w / 2, cy + h / 2],
                ]
            )
        elif len(valores) >= 7:
            coords = np.asarray(valores[1:], dtype=np.float64).reshape(-1, 2)
        else:
            continue
        coords = coords.copy()
        coords[:, 0] *= width
        coords[:, 1] *= height
        poligonos.append(Polygon(int(valores[0]), coords))
    return poligonos


def format_polygons(polygons: list[Polygon], width: int, height: int) -> str:
    """Escribe polígonos en el formato YOLO de segmentación."""
    lineas = []
    for p in polygons:
        normalizado = p.points / np.array([width, height])
        normalizado = np.clip(normalizado, 0.0, 1.0)
        lineas.append(f"{p.cls} " + " ".join(f"{v:.6f}" for v in normalizado.ravel()))
    return "\n".join(lineas) + ("\n" if lineas else "")


def extract_cutout(image: ImageBGR, polygon: Polygon, feather_px: int = 3) -> Cutout | None:
    """Recorta el objeto con su silueta. ``None`` si es demasiado chico para servir."""
    x1, y1, x2, y2 = (int(round(v)) for v in polygon.bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    local = polygon.points - np.array([x1, y1])
    mascara = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mascara, [np.round(local).astype(np.int32)], 255)
    k = 2 * feather_px + 1
    alpha = cv2.GaussianBlur(mascara, (k, k), 0).astype(np.float32) / 255.0
    return Cutout(
        pixels=image[y1:y2, x1:x2].copy(),
        alpha=alpha,
        polygon=local,
        source_bottom_y=float(y2),
        source_height=float(y2 - y1),
    )


def _luminancia(region: ImageBGR, peso: npt.NDArray[np.float32] | None = None) -> float:
    gris = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if peso is None:
        return float(gris.mean())
    return float((gris * peso).sum() / max(float(peso.sum()), 1e-6))


def paste(
    image: ImageBGR,
    polygons: list[Polygon],
    cutout: Cutout,
    rng: np.random.Generator,
    max_overlap: float = 0.1,
    attempts: int = 20,
) -> tuple[ImageBGR, list[Polygon]] | None:
    """Pega ``cutout`` en ``image`` en una posición plausible y libre.

    La fila de apoyo se elige entre los contactos de los equipos del frame, con
    algo de variación, porque ahí está el plano transitable. La escala sigue al
    alto de los CAEX del frame destino; sin CAEX se conserva la del origen.

    Returns:
        La imagen nueva y la lista de polígonos con el pegado agregado, o ``None``
        si no hubo lugar libre en ``attempts`` intentos.
    """
    alto, ancho = image.shape[:2]
    caex = [p for p in polygons if p.cls != CLASE_BULLDOZER]
    contactos = [p.bbox[3] for p in polygons] or [alto * 0.7]

    for _ in range(attempts):
        apoyo = float(rng.choice(contactos)) + rng.uniform(-0.04, 0.04) * alto
        apoyo = float(np.clip(apoyo, alto * 0.35, alto - 2))

        if caex:
            # Alto de referencia: el CAEX más cercano en fila, escalado por la
            # perspectiva lineal entre su contacto y la fila de apoyo elegida.
            ref = min(caex, key=lambda p: abs(p.bbox[3] - apoyo))
            alto_ref = ref.bbox[3] - ref.bbox[1]
            alto_obj = alto_ref * apoyo / max(ref.bbox[3], 1.0) * rng.uniform(0.55, 0.85)
            escala = alto_obj / cutout.source_height
        else:
            escala = rng.uniform(0.8, 1.2)
        escala = float(np.clip(escala, 0.3, 2.5))

        w = int(round(cutout.pixels.shape[1] * escala))
        h = int(round(cutout.pixels.shape[0] * escala))
        if w < 12 or h < 12 or w >= ancho or h >= alto:
            continue
        x = int(rng.uniform(0, ancho - w))
        y = int(round(apoyo)) - h
        if y < 0:
            continue

        nueva = (x, y, x + w, y + h)
        if any(_solape(nueva, p.bbox) > max_overlap for p in polygons):
            continue

        # OpenCV tipa la salida de resize como array genérico; conserva el dtype.
        pixeles = cast(ImageBGR, cv2.resize(cutout.pixels, (w, h), interpolation=cv2.INTER_AREA))
        alpha = cast(
            npt.NDArray[np.float32],
            cv2.resize(cutout.alpha, (w, h), interpolation=cv2.INTER_LINEAR),
        )

        destino = image[y : y + h, x : x + w]
        ganancia = (_luminancia(destino) + 1.0) / (_luminancia(pixeles, alpha) + 1.0)
        ganancia = float(np.clip(ganancia, 0.3, 3.0))
        ajustado = np.clip(pixeles.astype(np.float32) * ganancia, 0, 255)

        salida = image.copy()
        a = alpha[:, :, None]
        salida[y : y + h, x : x + w] = (a * ajustado + (1 - a) * destino).astype(np.uint8)

        puntos = cutout.polygon * escala + np.array([x, y])
        return salida, [*polygons, Polygon(CLASE_BULLDOZER, puntos)]
    return None


def _solape(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersección sobre el área de la caja más chica (contención, como el tracker)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    menor = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / menor if menor > 0 else 0.0
