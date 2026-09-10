"""Proyección del plano de suelo a vista cenital y escala métrica.

Convierte posiciones en píxeles a metros sobre el plano transitable, que es lo que
permite medir distancias entre equipos. Es la única etapa que razona sobre unidades.

## Por qué el problema es duro

Una imagen no contiene escala. Una cámara pinhole proyecta el mundo perdiendo la
profundidad, de modo que un objeto del doble de tamaño al doble de distancia produce
exactamente la misma imagen. Esa ambigüedad es irresoluble sin información adicional,
y no hay calibración de cámara, ni parámetros intrínsecos, ni metadatos de montaje.

Y medir distancias en píxeles es incorrecto, no aproximado: dos equipos separados
cincuenta píxeles cerca del horizonte están a cientos de metros entre sí, y los
mismos cincuenta píxeles en primer plano son unos pocos metros.

## El modelo

Se asume que la maquinaria circula sobre un plano y que la cámara es pinhole. Bajo
esos supuestos, para una fila de imagen ``y`` por debajo del horizonte ``y_h``:

    Z(y) = f · h / (y − y_h)          profundidad en metros
    X(x, y) = (x − c_x) · Z(y) / f    posición lateral en metros

Quedan tres incógnitas: el horizonte ``y_h``, la distancia focal ``f`` y la altura de
montaje ``h``. Se resuelven así:

* **``y_h`` se estima** de la frontera entre cielo y terreno, la misma señal que usa
  la segmentación del pretil.
* **``f`` se asume** a partir de un campo de visión declarado. Es el supuesto más
  débil del módulo y está en `configs/`, no en el código.
* **``h`` se deriva** del ancho nominal de un CAEX detectado: conocida su anchura real
  y su anchura en píxeles a una fila dada, la ecuación de arriba despeja ``h``.

## Lo que esto puede y no puede

El error es del orden de **±30 %** en distancias absolutas, dominado por el supuesto
de campo de visión y por la dispersión de la dimensión nominal. No es una medición
metrológica y no se presenta como tal.

En términos **relativos** el sistema es mucho más confiable: detectar que dos equipos
se están acercando, o que uno se acercó más que en la ronda anterior, no depende de
acertar la escala absoluta. Ese es además el caso de uso operativo — a un supervisor
le importa que la distancia esté disminuyendo, no si son 11 o 14 metros.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from bermguard.core.exceptions import InsufficientDataError
from bermguard.core.types import Detection, VehicleClass

logger = logging.getLogger(__name__)

CAEX_ANCHO_NOMINAL_M = 9.8
"""Anchura de un camión de extracción, en metros.

Corresponde al orden de magnitud de un Caterpillar 793 o un Komatsu 930E, que es la
silueta compatible con la maquinaria del material. Se usa la **anchura** y no la
altura porque la tolva bascula durante la descarga: la altura de la caja envolvente
cambia con la operación, mientras el ancho de vía no.

La dispersión entre modelos ronda el 10 %, y es una de las dos fuentes dominantes de
error del módulo.
"""


@dataclass(frozen=True, slots=True)
class GroundPlaneModel:
    """Modelo de plano de suelo ajustado, suficiente para proyectar a metros."""

    horizon_y_px: float
    focal_px: float
    camera_height_m: float
    principal_x_px: float

    anchors: int
    """Número de detecciones que contribuyeron al ajuste. Con una sola, la escala
    depende de un único objeto y su error de caja se propaga sin promediar."""

    camera_height_spread_m: float = 0.0
    """Dispersión entre las alturas de cámara que implica cada ancla, medida como
    semirrango intercuartílico.

    Es una medida **observada** de la incertidumbre del ancla, no un modelo de
    error: si tres CAEX implican alturas de cámara distintas, esa discrepancia
    acota lo que la escala puede saber. Vale cero con una sola ancla, y entonces la
    incertidumbre hay que declararla en lugar de medirla."""

    def project(self, x_px: float, y_px: float) -> tuple[float, float] | None:
        """Proyecta un punto del suelo a ``(lateral_m, profundidad_m)``.

        Returns:
            ``None`` si el punto está en o por encima del horizonte, donde la
            proyección diverge. Devolver una distancia enorme sería peor que
            devolver nada: entraría en los cálculos como un número plausible.
        """
        bajo_horizonte = y_px - self.horizon_y_px
        if bajo_horizonte <= 1.0:
            return None
        profundidad = self.focal_px * self.camera_height_m / bajo_horizonte
        lateral = (x_px - self.principal_x_px) * profundidad / self.focal_px
        return lateral, profundidad


def estimate_horizon(frame_gray: np.ndarray, variance_threshold: float = 6.0) -> float:
    """Fila del horizonte, estimada por la primera fila con textura desde arriba.

    El cielo tiene varianza por fila muy baja. Es la misma señal que usa la
    segmentación del pretil, y compartirla evita que ambos módulos discrepen sobre
    dónde empieza el terreno.
    """
    varianza = frame_gray.std(axis=1)
    con_textura = np.flatnonzero(varianza > variance_threshold)
    return float(con_textura[0]) if con_textura.size else 0.0


def fit_ground_plane(
    detections: list[Detection],
    frame_width: int,
    horizon_y_px: float,
    horizontal_fov_deg: float,
) -> GroundPlaneModel:
    """Ajusta el modelo de plano de suelo con las detecciones disponibles.

    Args:
        detections: Detecciones del frame. Sólo las de clase CAEX sirven de ancla,
            porque es la única cuya dimensión nominal se conoce.
        frame_width: Ancho del frame, para derivar la focal del campo de visión.
        horizon_y_px: Fila del horizonte.
        horizontal_fov_deg: Campo de visión horizontal asumido, en grados.

    Returns:
        El modelo ajustado.

    Raises:
        InsufficientDataError: Si no hay ninguna ancla utilizable.

    Note:
        Con varias anclas se toma la **mediana** de las alturas de cámara implicadas
        y no la media: una caja mal ajustada produce un valor muy desviado, y la
        mediana lo ignora en lugar de dejar que arrastre el ajuste. Es el mismo
        motivo por el que se usa RANSAC en problemas de este tipo, aplicado a un
        caso donde el modelo tiene un solo parámetro libre.
    """
    focal_px = frame_width / (2.0 * np.tan(np.radians(horizontal_fov_deg) / 2.0))
    principal_x = frame_width / 2.0

    alturas: list[float] = []
    for deteccion in detections:
        if deteccion.vehicle_class is not VehicleClass.CAEX:
            continue
        _gx, gy = deteccion.bbox.ground_point
        bajo_horizonte = gy - horizon_y_px
        if bajo_horizonte <= 1.0 or deteccion.bbox.width <= 1.0:
            continue

        # Del modelo: ancho_px = ancho_real · f / Z  y  Z = f · h / (y − y_h).
        # Sustituyendo y despejando h queda esta expresion, independiente de f.
        alturas.append(CAEX_ANCHO_NOMINAL_M * bajo_horizonte / deteccion.bbox.width)

    if not alturas:
        raise InsufficientDataError(
            "No hay ningun CAEX utilizable como ancla metrica en este frame"
        )

    arreglo = np.asarray(alturas, dtype=np.float64)
    dispersion = (
        float(np.percentile(arreglo, 75) - np.percentile(arreglo, 25)) / 2.0
        if arreglo.size >= 2
        else 0.0
    )
    return GroundPlaneModel(
        horizon_y_px=horizon_y_px,
        focal_px=focal_px,
        camera_height_m=float(np.median(arreglo)),
        principal_x_px=principal_x,
        anchors=int(arreglo.size),
        camera_height_spread_m=dispersion,
    )


def pairwise_distances(
    detections: list[Detection], model: GroundPlaneModel
) -> dict[tuple[int, int], float]:
    """Distancias métricas entre cada par de equipos, sobre el plano de suelo.

    Se proyecta el **punto de contacto con el suelo** de cada detección, nunca su
    centro: la proyección sólo es válida para puntos que están en el plano, y el
    centro de la caja está a media altura de la máquina. Usar el centro produciría
    posiciones desplazadas hacia el fondo, con error creciente para los equipos más
    grandes — exactamente los que más importan.

    Returns:
        Mapa de ``(track_id_menor, track_id_mayor)`` a distancia en metros. Los pares
        cuya proyección falla quedan fuera en lugar de aparecer con una distancia
        inventada.
    """
    posiciones: dict[int, tuple[float, float]] = {}
    for deteccion in detections:
        if deteccion.track_id < 0:
            continue
        gx, gy = deteccion.bbox.ground_point
        proyectado = model.project(gx, gy)
        if proyectado is not None:
            posiciones[deteccion.track_id] = proyectado

    distancias: dict[tuple[int, int], float] = {}
    ids = sorted(posiciones)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            xa, za = posiciones[a]
            xb, zb = posiciones[b]
            distancias[(a, b)] = float(np.hypot(xa - xb, za - zb))
    return distancias


def world_positions(
    detections: list[Detection], model: GroundPlaneModel
) -> dict[int, tuple[float, float]]:
    """Posición métrica de cada equipo, para el mapa de dispersión cenital."""
    salida: dict[int, tuple[float, float]] = {}
    for deteccion in detections:
        if deteccion.track_id < 0:
            continue
        gx, gy = deteccion.bbox.ground_point
        proyectado = model.project(gx, gy)
        if proyectado is not None:
            salida[deteccion.track_id] = proyectado
    return salida
