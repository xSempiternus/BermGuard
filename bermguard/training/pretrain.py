"""Remapeo de datasets públicos a las clases del proyecto, para preentrenar.

El ADR 0004 descartó entrenar *sólo* con datos públicos por brecha de dominio, y
dejó abierto el punto medio: preentrenar con ellos y afinar con los frames
propios. Cada dataset público nombra sus clases a su manera (``dozer``,
``Bull_dozer``, ``tippertruck``, ``haul_truck``...), así que antes hay que
traducirlas a ``caex`` y ``bulldozer``.

Una clase pública sin traducción se descarta, no se reasigna: una excavadora no es
ni un CAEX ni un bulldozer, y forzarla a alguno de los dos enseñaría una confusión
que el detector propio no tiene.
"""

from __future__ import annotations

CLASES_PROYECTO = ("caex", "bulldozer")


def normalizar(nombre: str) -> str:
    """``Bull_dozer``, ``bull dozer`` y ``bull-dozer`` son la misma clase."""
    return "".join(c for c in nombre.lower() if c.isalnum())


def parse_class_map(items: list[str]) -> dict[str, int]:
    """Traduce ``["dozer=bulldozer", "haul_truck=caex"]`` a ``{"dozer": 1, "haultruck": 0}``."""
    mapa: dict[str, int] = {}
    for item in items:
        origen, sep, destino = item.partition("=")
        if not sep or destino not in CLASES_PROYECTO:
            raise ValueError(
                f"'{item}': se espera origen=destino, con destino en {CLASES_PROYECTO}"
            )
        mapa[normalizar(origen)] = CLASES_PROYECTO.index(destino)
    return mapa


def remap_labels(texto: str, nombres_origen: list[str], mapa: dict[str, int]) -> str:
    """Reescribe un archivo de etiquetas YOLO con los índices del proyecto.

    Conserva las coordenadas tal cual (sirven igual cajas que polígonos) y descarta
    las líneas de clases sin traducción.
    """
    salida = []
    for linea in texto.splitlines():
        partes = linea.split()
        if not partes:
            continue
        indice = int(float(partes[0]))
        if not 0 <= indice < len(nombres_origen):
            continue
        destino = mapa.get(normalizar(nombres_origen[indice]))
        if destino is not None:
            salida.append(" ".join([str(destino), *partes[1:]]))
    return "\n".join(salida) + ("\n" if salida else "")
