"""Etiquetas YOLO en un solo formato por archivo.

Las etiquetas propias mezclan, en un mismo frame, cajas (``c cx cy w h``: las
pre-anotaciones que no se corrigieron) y polígonos (las corregidas a mano).
Ultralytics 8.4 trata un archivo así como corrupto y **descarta la imagen entera**
("labels mix segment and detection rows"): con los datos de 2026-10 eso dejaba fuera
36 de los 40 frames de video_01.

Las etiquetas versionadas no se reescriben, porque la auditoría
(``data/auditoria_etiquetas.json``) identifica cada corrección por la línea original.
Se homogeneizan al copiar el dataset a la carpeta de entrenamiento.
"""

from __future__ import annotations


def caja_a_poligono(valores: list[str]) -> list[str]:
    """``[c, cx, cy, w, h]`` a ``[c, x1, y1, x2, y1, x2, y2, x1, y2]``, sin perder área."""
    cx, cy, w, h = (float(v) for v in valores[1:5])
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    esquinas = [x1, y1, x2, y1, x2, y2, x1, y2]
    return [valores[0], *(f"{v:.6f}" for v in esquinas)]


def homogeneizar(texto: str) -> str:
    """Si el archivo mezcla cajas y polígonos, pasa las cajas a polígonos de 4 vértices.

    Un archivo que ya está en un solo formato se devuelve tal cual.
    """
    filas = [linea.split() for linea in texto.splitlines() if linea.strip()]
    hay_cajas = any(len(f) == 5 for f in filas)
    hay_poligonos = any(len(f) > 5 for f in filas)
    if not (hay_cajas and hay_poligonos):
        return texto
    salida = [caja_a_poligono(f) if len(f) == 5 else f for f in filas]
    return "\n".join(" ".join(f) for f in salida) + "\n"
