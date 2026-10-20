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


def contiene(texto: str, clase: str) -> bool:
    """Si un archivo ya remapeado tiene al menos una instancia de ``clase``."""
    indice = str(CLASES_PROYECTO.index(clase))
    return any(linea.split()[:1] == [indice] for linea in texto.splitlines())


def descartar_degeneradas(texto: str, minimo: float = 0.005) -> tuple[str, int]:
    """Quita instancias cuyo ancho o alto normalizado es menor que ``minimo``.

    Algunos exports de Roboflow traen polígonos de dos o tres puntos casi
    superpuestos (un clic de más al anotar). Ultralytics los convierte en una caja
    de menos de un píxel que no corresponde a ningún objeto. Devuelve el texto
    limpio y cuántas instancias quitó.
    """
    salida = []
    quitadas = 0
    for linea in texto.splitlines():
        partes = linea.split()
        if not partes:
            continue
        valores = [float(v) for v in partes[1:]]
        if len(valores) == 4:
            ancho, alto = valores[2], valores[3]
        else:
            xs, ys = valores[0::2], valores[1::2]
            ancho, alto = max(xs) - min(xs), max(ys) - min(ys)
        if ancho < minimo or alto < minimo:
            quitadas += 1
            continue
        salida.append(linea)
    return "\n".join(salida) + ("\n" if salida else ""), quitadas


def leer_exclusiones(texto: str) -> set[str]:
    """Nombres de archivo de una lista de exclusión, sin comentarios ni líneas vacías.

    Formato: un nombre por línea; lo que sigue a ``#`` es el motivo.
    """
    nombres = set()
    for linea in texto.splitlines():
        nombre = linea.split("#", 1)[0].strip()
        if nombre:
            nombres.add(nombre)
    return nombres


def parse_repeat(items: list[str]) -> dict[str, int]:
    """Traduce ``["mines=5"]`` a ``{"mines": 5}``: cuántas veces entra cada fuente en train."""
    repeticiones: dict[str, int] = {}
    for item in items:
        fuente, sep, veces = item.partition("=")
        if not sep or not fuente or not veces.isdigit() or int(veces) < 1:
            raise ValueError(f"'{item}': se espera FUENTE=N con N entero >= 1")
        repeticiones[fuente] = int(veces)
    return repeticiones


def nombres_repetidos(nombre: str, veces: int) -> list[str]:
    """``nombre`` y sus copias ``nombre_r1`` ... ``nombre_r{veces-1}``."""
    return [nombre] + [f"{nombre}_r{i}" for i in range(1, veces)]
