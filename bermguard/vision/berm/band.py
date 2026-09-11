"""Banda vertical donde se busca el pretil, común a todos los segmentadores.

Vive en un módulo propio y no dentro de cada segmentador por una razón concreta: la
comparación entre métodos del benchmark sólo es válida si **todos buscan en la misma
región**. Con la lógica duplicada, un cambio aplicado a uno y olvidado en el otro
alteraría la comparación sin que nada lo indicara.

## La restricción, y la corrección que la motivó

La banda se acota por arriba y por abajo con la **maquinaria detectada**, no sólo con
el frame.

La primera versión usaba la maquinaria únicamente como límite inferior —el pretil
está por encima de la rasante— y tomaba como límite superior la primera fila con
textura, es decir, prácticamente el horizonte. De noche eso falla de forma
sistemática: el borde horizontal más fuerte y continuo de la escena no es el pretil
sino la frontera entre el terreno oscuro y el valle iluminado del fondo, y el camino
óptimo se engancha a ella. En el frame 30 de ``video_02`` la cresta reportada corría
sobre la línea del valle, a 265 px por encima del punto de contacto del camión,
mientras el pretil real estaba a unos 55 px.

El límite superior se deriva de un hecho físico: **el pretil está al borde de la
plataforma por la que circula la maquinaria**, es decir, aproximadamente a la misma
profundidad que los equipos. En el espacio imagen no puede aparecer mucho más arriba
que el techo de un camión que opera junto a él. La banda se acota entonces a una
fracción de la altura de la caja por encima del punto de contacto.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bermguard.core.types import Detection


def search_band(
    realzado: np.ndarray,
    detections: Sequence[Detection],
    sky_variance_threshold: float = 6.0,
    band_margin_frac: float = 0.04,
    vehicle_height_multiple: float = 1.0,
) -> tuple[int, int]:
    """Filas ``(y_min, y_max)`` entre las que se busca el pretil.

    Args:
        realzado: Frame en escala de grises, ya acondicionado.
        detections: Maquinaria del frame. Define ambos límites cuando está presente.
        sky_variance_threshold: Desviación estándar por fila bajo la cual se
            considera cielo.
        band_margin_frac: Margen bajo la rasante, como fracción de la altura del
            frame, para tolerar el error de la propia referencia de suelo.
        vehicle_height_multiple: Cuántas alturas de caja por encima del punto de
            contacto puede estar el pretil. Con 1.0 se admite un pretil tan alto en
            la imagen como el techo del vehículo, que es generoso: un pretil real
            mide del orden de un tercio de la altura de un CAEX.

    Returns:
        Límites de la banda. Sin maquinaria se recurre sólo al cielo y a un recorte
        del primer plano, que es un criterio grosero; el segmentador lo refleja
        bajando su confianza.
    """
    alto = realzado.shape[0]

    # El cielo sigue excluyendose en todos los casos: es el borde horizontal mas
    # nitido de la escena, y sin maquinaria es el unico limite superior disponible.
    varianza = realzado.std(axis=1)
    con_textura = np.flatnonzero(varianza > sky_variance_threshold)
    y_cielo = int(con_textura[0]) if con_textura.size else 0

    if not detections:
        return y_cielo, max(y_cielo, int(alto * 0.80))

    # Limite inferior: el contacto mas bajo, el de la maquina mas cercana.
    rasante = max(d.bbox.ground_point[1] for d in detections)
    y_max = int(min(alto, rasante + alto * band_margin_frac))

    # Limite superior: la cota mas alta que admite alguna de las maquinas. Se toma el
    # minimo entre todas porque el pretil puede estar junto a cualquiera de ellas, y
    # descartarlo por estar cerca de un equipo lejano seria perder pretil real.
    techo = min(
        d.bbox.ground_point[1] - vehicle_height_multiple * d.bbox.height for d in detections
    )
    y_min = max(y_cielo, int(techo))

    return y_min, max(y_min, y_max)
