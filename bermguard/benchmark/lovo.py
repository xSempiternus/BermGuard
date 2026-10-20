"""Resumen de la validación cruzada dejando un video fuera (LOVO) y decisión de variante.

Entrada: un JSON por variante y fold, ``{variante}_{video}.json``, que es la salida de
``scripts/evaluate_detector.py`` más la clave ``particion`` que agrega
``scripts/cross_validate.py``. Salida: un diccionario serializable (``resumen.json``) y un
bloque markdown para ``docs/resultados.md``. Todo número del resumen sale de esos JSON;
este módulo no mide nada, sólo agrega y decide.

**Agregación.** Por variante, clase y métrica: media y desviación estándar muestral
(``ddof=1``) entre folds, y el valor de cada fold con su número de objetos. Un fold en el
que la métrica no está definida (precisión sin ninguna detección) queda fuera de esa media
y se informa cuántos folds entraron.

**Comparación contra ``base``.** Diferencia pareada por fold, ``variante - base``, en los
folds que tienen las dos. Se reporta media, mínimo, máximo y cuántos folds mejoran,
empeoran o empatan. Una diferencia cuyo valor absoluto no supera ``empate`` cuenta como
empate. El veredicto de cada métrica es:

* ``mejora`` si mejora en al menos ``folds_minimos`` folds (3 de 4 por defecto) **y** la
  media de la diferencia es al menos ``efecto_minimo``;
* ``empeora`` en el caso simétrico;
* ``no distinguible del ruido`` en cualquier otro caso.

Con 4 folds ningún resultado es estadísticamente significativo al 5 %: aun 4 de 4 a favor
da p = 0.125 en la prueba de signos bilateral (que también se reporta). El criterio es
práctico, no inferencial: pide consistencia entre videos y un tamaño mínimo.

**Umbral de confianza fuera de fold.** Para cada fold *k*, el umbral de la clase se elige
con los barridos de los **otros** folds de la misma variante, sumando sus ``tp``, ``fp`` y
objetos por umbral (si el JSON es anterior y no trae conteos, se promedian las
proporciones), y se evalúa en el barrido de *k*. El barrido de *k* nunca entra a la
elección del umbral de *k*. El umbral de producción se elige con los cuatro folds; su
rendimiento esperado es el fuera de fold, no el que se obtiene al aplicarlo a esos mismos
folds. Criterios (``CriterioUmbral``):

* ``precision_min`` (por defecto, con 0.5): el umbral de mayor recall cuya precisión
  agregada es al menos el mínimo. Para el bulldozer manda el recall (una máquina no
  detectada es una máquina sin vigilar, ``docs/resultados.md`` §1.3) y el piso de
  precisión acota las falsas alarmas a nivel de frame, que el tracker filtra después con
  ``min_hits``.
* ``recall_min``: el umbral de mayor precisión cuyo recall es al menos el mínimo (el
  ejemplo de la metodología para ``caex``: recall ≥ 0.95).
* ``max_f1``: el de mayor F1.

Los empates se rompen por la otra métrica y después por el umbral más alto (menos falsas
alarmas). Si ningún umbral cumple la restricción, se usa el de mayor F1 y se marca
``cumple: false``.

**Regla de decisión de la variante** (``ReglaDecision``). Una variante distinta de
``base`` es candidata si tiene los mismos folds que ``base``, todos los de ``base``
(cuatro), y:

1. el AP@0.5 del bulldozer tiene veredicto ``mejora`` contra ``base`` (mejora en al menos
   3 de 4 folds y en media al menos ``efecto_minimo``), y
2. el AP@0.5 del CAEX no cae en media más de ``caida_media_caex`` ni en ningún fold más de
   ``caida_fold_caex``.

Gana la candidata con mayor mejora media del bulldozer; si dos quedan a menos de
``empate``, gana la más simple en el orden ``copypaste`` (sólo datos) antes que
``pretrain`` (un entrenamiento extra). Si ninguna es candidata, gana ``base`` por
parsimonia. La decisión usa AP@0.5, que no depende del umbral; el umbral se elige después,
sobre la variante ganadora.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

CLASES = ("bulldozer", "caex")
"""El bulldozer primero: es la clase escasa y la que manda la decisión."""
METRICAS = ("ap50", "ap50_95", "precision", "recall", "f1")
VARIANTES = ("base", "copypaste", "pretrain")
"""También es el orden de simplicidad para desempatar."""
PATRON_ARCHIVO = re.compile(r"^(?P<variante>[a-z]+)_(?P<video>video_\d+)\.json$")

Reporte = Mapping[str, Any]
Folds = Mapping[str, Reporte]
"""video -> JSON de evaluate_detector para ese fold."""


@dataclass(frozen=True, slots=True)
class CriterioUmbral:
    tipo: Literal["precision_min", "recall_min", "max_f1"] = "precision_min"
    minimo: float = 0.5
    """Piso de precisión (``precision_min``) o de recall (``recall_min``)."""


@dataclass(frozen=True, slots=True)
class ReglaDecision:
    empate: float = 0.01
    """|Δ| por debajo de esto, en un fold, es empate."""
    efecto_minimo: float = 0.02
    """Media de Δ necesaria para declarar mejora o empeora."""
    folds_minimos: int = 3
    """Folds que tienen que mejorar (o empeorar) para salir del ruido."""
    caida_media_caex: float = 0.02
    caida_fold_caex: float = 0.05


# --- Utilidades --------------------------------------------------------------


def _num(valor: Any) -> float:
    """Float, con ``None`` y valores no numéricos como NaN."""
    if valor is None:
        return float("nan")
    try:
        return float(valor)
    except (TypeError, ValueError):
        return float("nan")


def _valido(x: float) -> bool:
    return not math.isnan(x)


def _media_desv(valores: Iterable[float]) -> dict[str, float | int]:
    v = [x for x in valores if _valido(x)]
    return {
        "media": float(np.mean(v)) if v else float("nan"),
        "desviacion": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
        "n_folds": len(v),
    }


def p_signo(mejoras: int, empeoras: int) -> float:
    """p bilateral exacto de la prueba de signos; los empates no cuentan."""
    n = mejoras + empeoras
    if n == 0:
        return 1.0
    k = min(mejoras, empeoras)
    cola = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * cola))


def limpiar_nan(obj: Any) -> Any:
    """Reemplaza NaN por ``None`` para escribir JSON estándar."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, Mapping):
        return {k: limpiar_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [limpiar_nan(v) for v in obj]
    return obj


# --- Carga -------------------------------------------------------------------


def cargar(directorio: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Lee ``{variante}_{video}.json``. Devuelve variante -> video -> reporte."""
    salida: dict[str, dict[str, dict[str, Any]]] = {}
    for ruta in sorted(directorio.glob("*.json")):
        m = PATRON_ARCHIVO.match(ruta.name)
        if not m:
            continue
        reporte = json.loads(ruta.read_text(encoding="utf-8"))
        salida.setdefault(m["variante"], {})[m["video"]] = reporte
    return salida


def metrica(reporte: Reporte, clase: str, nombre: str) -> float:
    return _num(reporte.get("por_clase", {}).get(clase, {}).get(nombre))


# --- 1. Agregación por variante ---------------------------------------------


def agregar_variante(folds: Folds) -> dict[str, Any]:
    """Media ± desviación entre folds y detalle por fold, por clase y métrica."""
    videos = sorted(folds)
    salida: dict[str, Any] = {
        "folds": videos,
        "imagenes": {v: folds[v].get("imagenes") for v in videos},
        "particion": {v: folds[v].get("particion") for v in videos},
        "umbral_operacion": {
            v: {
                c: _num(
                    folds[v].get("umbral_por_clase", {}).get(c, folds[v].get("umbral_operacion"))
                )
                for c in CLASES
            }
            for v in videos
        },
        "por_clase": {},
    }
    for clase in CLASES:
        bloque: dict[str, Any] = {
            "n_objetos": {v: metrica(folds[v], clase, "n_truth") for v in videos}
        }
        for nombre in METRICAS:
            por_fold = {v: metrica(folds[v], clase, nombre) for v in videos}
            bloque[nombre] = {**_media_desv(por_fold.values()), "por_fold": por_fold}
        salida["por_clase"][clase] = bloque
    return salida


# --- 2. Diferencias pareadas -------------------------------------------------


def diferencia_pareada(
    base: Folds, variante: Folds, clase: str, nombre: str, regla: ReglaDecision
) -> dict[str, Any]:
    """Δ = variante - base por fold, en los folds comunes con la métrica definida."""
    por_fold: dict[str, float] = {}
    for video in sorted(set(base) & set(variante)):
        d = metrica(variante[video], clase, nombre) - metrica(base[video], clase, nombre)
        if _valido(d):
            por_fold[video] = d
    difs = list(por_fold.values())
    mejoran = sum(x > regla.empate for x in difs)
    empeoran = sum(x < -regla.empate for x in difs)
    media = float(np.mean(difs)) if difs else float("nan")
    if mejoran >= regla.folds_minimos and media >= regla.efecto_minimo:
        veredicto = "mejora"
    elif empeoran >= regla.folds_minimos and media <= -regla.efecto_minimo:
        veredicto = "empeora"
    else:
        veredicto = "no distinguible del ruido"
    return {
        "por_fold": por_fold,
        "media": media,
        "minimo": float(min(difs)) if difs else float("nan"),
        "maximo": float(max(difs)) if difs else float("nan"),
        "n_folds": len(difs),
        "folds_mejora": mejoran,
        "folds_empeora": empeoran,
        "folds_empate": len(difs) - mejoran - empeoran,
        "p_signo": p_signo(mejoran, empeoran),
        "veredicto": veredicto,
    }


def comparar_contra_base(
    datos: Mapping[str, Folds], regla: ReglaDecision
) -> dict[str, dict[str, dict[str, Any]]]:
    """variante -> clase -> métrica -> diferencia pareada contra ``base``."""
    if "base" not in datos:
        return {}
    return {
        variante: {
            clase: {
                nombre: diferencia_pareada(datos["base"], folds, clase, nombre, regla)
                for nombre in METRICAS
            }
            for clase in CLASES
        }
        for variante, folds in datos.items()
        if variante != "base"
    }


# --- 3. Umbral fuera de fold -------------------------------------------------


def _filas(reporte: Reporte, clase: str) -> dict[float, dict[str, float]]:
    """Barrido de una clase indexado por umbral (redondeado para comparar grillas)."""
    return {
        round(_num(f["conf"]), 6): {k: _num(v) for k, v in f.items()}
        for f in reporte.get("barrido", {}).get(clase, [])
    }


def _tiene_conteos(fila: Mapping[str, Any]) -> bool:
    return all(k in fila and _valido(_num(fila[k])) for k in ("tp", "fp", "n_truth"))


def agregar_barridos(reportes: Sequence[Reporte], clase: str) -> list[dict[str, Any]]:
    """Barrido conjunto de varios folds, sobre los umbrales comunes a todos.

    Si todos los barridos traen ``tp``/``fp``/``n_truth``, se suman los conteos y se
    recalculan precisión, recall y F1 (cada objeto pesa lo mismo). Si no, se promedian
    las proporciones de los folds (precisión sin definir queda fuera de su media).
    """
    tablas = [_filas(r, clase) for r in reportes]
    if not tablas:
        return []
    comunes = sorted(set.intersection(*(set(t) for t in tablas)))
    salida = []
    for conf in comunes:
        filas = [t[conf] for t in tablas]
        if all(_tiene_conteos(f) for f in filas):
            tp = sum(f["tp"] for f in filas)
            fp = sum(f["fp"] for f in filas)
            n = sum(f["n_truth"] for f in filas)
            precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
            recall = tp / n if n > 0 else float("nan")
            agregacion = "conteos"
        else:
            ps = [f["precision"] for f in filas if _valido(f["precision"])]
            precision = float(np.mean(ps)) if ps else float("nan")
            recall = float(np.mean([f["recall"] for f in filas]))
            tp = fp = n = float("nan")
            agregacion = "promedio"
        f1 = (
            2 * precision * recall / (precision + recall)
            if _valido(precision) and _valido(recall) and precision + recall > 0
            else 0.0
        )
        salida.append(
            {
                "conf": conf,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "n_truth": n,
                "agregacion": agregacion,
            }
        )
    return salida


def elegir_umbral(filas: Sequence[Mapping[str, Any]], criterio: CriterioUmbral) -> dict[str, Any]:
    """Aplica el criterio a un barrido. Ver el docstring del módulo."""
    if not filas:
        return {"conf": float("nan"), "cumple": False, "en_borde_de_grilla": False}

    def val(f: Mapping[str, Any], k: str) -> float:
        x = _num(f.get(k))
        return x if _valido(x) else -1.0

    orden = {
        "precision_min": ("recall", "precision", "conf"),
        "recall_min": ("precision", "recall", "conf"),
        "max_f1": ("f1", "conf"),
    }
    restringida = {"precision_min": "precision", "recall_min": "recall"}.get(criterio.tipo)
    aptas = [f for f in filas if restringida is None or val(f, restringida) >= criterio.minimo]
    cumple = bool(aptas)
    claves = orden[criterio.tipo] if cumple else orden["max_f1"]
    elegida = max(aptas or filas, key=lambda f: tuple(val(f, k) for k in claves))
    # En el umbral más bajo de la grilla, el óptimo puede estar por debajo de ella.
    borde = val(elegida, "conf") <= min(val(f, "conf") for f in filas)
    return {**dict(elegida), "cumple": cumple, "en_borde_de_grilla": borde}


def umbral_fuera_de_fold(folds: Folds, clase: str, criterio: CriterioUmbral) -> dict[str, Any]:
    """Umbral elegido sin el fold *k* y medido en *k*, para cada *k*; y el de producción."""
    videos = sorted(folds)
    por_fold: dict[str, Any] = {}
    tp_total = fp_total = n_total = 0.0
    con_conteos = True
    for k in videos:
        otros = [v for v in videos if v != k]
        eleccion = elegir_umbral(agregar_barridos([folds[v] for v in otros], clase), criterio)
        conf = _num(eleccion.get("conf"))
        fila = _filas(folds[k], clase).get(round(conf, 6), {}) if _valido(conf) else {}
        por_fold[k] = {
            "umbral": conf,
            "elegido_con": otros,
            "cumple_en_otros": eleccion["cumple"],
            "en_borde_de_grilla": eleccion["en_borde_de_grilla"],
            "precision": _num(fila.get("precision")),
            "recall": _num(fila.get("recall")),
            "f1": _num(fila.get("f1")),
            "n_objetos": metrica(folds[k], clase, "n_truth"),
        }
        if _tiene_conteos(fila):
            tp_total += fila["tp"]
            fp_total += fila["fp"]
            n_total += fila["n_truth"]
        else:
            con_conteos = False
    conjunto: dict[str, float] = {}
    if con_conteos and n_total > 0:
        p = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else float("nan")
        r = tp_total / n_total
        conjunto = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if _valido(p) and p + r > 0 else 0.0,
            "tp": tp_total,
            "fp": fp_total,
            "n_objetos": n_total,
        }
    produccion = elegir_umbral(agregar_barridos([folds[v] for v in videos], clase), criterio)
    return {
        "clase": clase,
        "criterio": asdict(criterio),
        "por_fold": por_fold,
        **{m: _media_desv(por_fold[v][m] for v in videos) for m in ("precision", "recall", "f1")},
        "conjunto_fuera_de_fold": conjunto,
        "produccion": {
            "umbral": _num(produccion.get("conf")),
            "cumple": produccion["cumple"],
            "en_borde_de_grilla": produccion["en_borde_de_grilla"],
            "elegido_con": videos,
            "nota": "Rendimiento esperado: el fuera de fold, no el de estos mismos folds.",
        },
    }


# --- 5. Decisión -------------------------------------------------------------


def elegir_variante(
    datos: Mapping[str, Folds],
    comparaciones: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    regla: ReglaDecision,
) -> dict[str, Any]:
    """Aplica la regla de decisión del docstring del módulo."""
    folds_base = set(datos.get("base", {}))
    evaluacion: dict[str, Any] = {}
    candidatas: list[tuple[str, float]] = []
    for variante in [v for v in VARIANTES if v in comparaciones] + sorted(
        set(comparaciones) - set(VARIANTES)
    ):
        bull = comparaciones[variante]["bulldozer"]["ap50"]
        caex = comparaciones[variante]["caex"]["ap50"]
        motivos = []
        completa = set(datos[variante]) == folds_base and len(folds_base) >= 4
        if not completa:
            motivos.append(
                f"folds incompletos ({len(datos[variante])} contra {len(folds_base)} de base)"
            )
        if bull["veredicto"] != "mejora":
            motivos.append(f"AP@0.5 bulldozer: {bull['veredicto']}")
        if not caex["n_folds"] or caex["media"] < -regla.caida_media_caex:
            motivos.append(f"AP@0.5 caex cae en media más de {regla.caida_media_caex}")
        if caex["n_folds"] and caex["minimo"] < -regla.caida_fold_caex:
            motivos.append(f"AP@0.5 caex cae más de {regla.caida_fold_caex} en algún fold")
        evaluacion[variante] = {"candidata": not motivos, "motivos": motivos}
        if not motivos:
            candidatas.append((variante, bull["media"]))
    if not candidatas:
        ganadora = "base"
        motivo = "Ninguna variante cumple la regla: base por parsimonia."
    else:
        mejor = max(d for _, d in candidatas)
        # El orden de iteración ya es el de simplicidad: la primera dentro del empate.
        ganadora = next(v for v, d in candidatas if mejor - d <= regla.empate)
        motivo = (
            f"{ganadora} mejora el AP@0.5 del bulldozer de forma consistente sin degradar "
            "el del CAEX más de lo tolerado."
        )
    return {
        "ganadora": ganadora,
        "motivo": motivo,
        "regla": asdict(regla),
        "evaluacion": evaluacion,
    }


# --- Resumen completo --------------------------------------------------------


def resumir_lovo(
    datos: Mapping[str, Folds],
    criterio: CriterioUmbral | None = None,
    regla: ReglaDecision | None = None,
    clase_umbral: str = "bulldozer",
) -> dict[str, Any]:
    """Agrega, compara, elige variante y umbral. NaN donde la métrica no está definida."""
    criterio = criterio or CriterioUmbral()
    regla = regla or ReglaDecision()
    comparaciones = comparar_contra_base(datos, regla)
    decision = elegir_variante(datos, comparaciones, regla)
    umbrales = {v: umbral_fuera_de_fold(f, clase_umbral, criterio) for v, f in datos.items()}
    return {
        "variantes": {v: agregar_variante(f) for v, f in datos.items()},
        "diferencias_contra_base": comparaciones,
        "umbral_fuera_de_fold": umbrales,
        "decision": decision,
        "umbral_produccion": {
            "variante": decision["ganadora"],
            "clase": clase_umbral,
            **umbrales.get(decision["ganadora"], {}).get("produccion", {}),
        },
    }


# --- 4. Markdown -------------------------------------------------------------


def _f(x: Any, signo: bool = False) -> str:
    v = _num(x)
    if not _valido(v):
        return "—"
    return f"{v:+.3f}" if signo else f"{v:.3f}"


def _u(x: Any) -> str:
    v = _num(x)
    return "—" if not _valido(v) else f"{v:.2f}"


def _n(x: Any) -> str:
    v = _num(x)
    return "—" if not _valido(v) else str(round(v))


def _md(m: Mapping[str, Any]) -> str:
    if not m.get("n_folds"):
        return "—"
    texto = f"{_f(m['media'])} ± {_f(m['desviacion'])}"
    return texto if m["n_folds"] > 1 else f"{_f(m['media'])} (1 fold)"


NOMBRE_METRICA = {
    "ap50": "AP@0.5",
    "ap50_95": "AP@0.5:0.95",
    "precision": "Precisión",
    "recall": "Recall",
    "f1": "F1",
}


def a_markdown(resumen: Mapping[str, Any], comando: str = "") -> str:
    """Bloque para pegar en docs/resultados.md. Sólo números del resumen."""
    variantes = resumen["variantes"]
    lineas = ["### Validación cruzada dejando un video fuera (LOVO)", ""]
    lineas += [
        "**Pregunta.** ¿Alguna variante (copy-paste de bulldozers, preentrenamiento con "
        "datos públicos) detecta mejor el bulldozer que `base` en un video que no vio, "
        "sin empeorar el CAEX? Y, para la ganadora, ¿qué umbral de bulldozer usar?",
        "",
        "**Datos.** Un entrenamiento por variante y video excluido; cada fold se evalúa "
        "en el video que no vio (`docs/experimentos/lovo/{variante}_{video}.json`). "
        "Media ± desviación estándar muestral entre folds; precisión, recall y F1 en el "
        "umbral de operación de cada JSON.",
    ]
    if comando:
        lineas += ["", f"Reproducir: `{comando}`"]
    for clase in CLASES:
        lineas += ["", f"#### `{clase}`: media ± desviación entre folds", ""]
        lineas += ["| Variante | Folds | " + " | ".join(NOMBRE_METRICA.values()) + " |"]
        lineas += ["|---|---|" + "---|" * len(METRICAS)]
        for v, a in variantes.items():
            c = a["por_clase"][clase]
            lineas += [
                f"| `{v}` | {len(a['folds'])} | " + " | ".join(_md(c[m]) for m in METRICAS) + " |"
            ]
        lineas += ["", f"Detalle por fold de `{clase}` (objetos entre paréntesis):", ""]
        videos = sorted({x for a in variantes.values() for x in a["folds"]})
        lineas += ["| Variante | Métrica | " + " | ".join(videos) + " |"]
        lineas += ["|---|---|" + "---|" * len(videos)]
        for v, a in variantes.items():
            c = a["por_clase"][clase]
            for m in ("ap50", "ap50_95", "recall"):
                celdas = []
                for video in videos:
                    valor = c[m]["por_fold"].get(video)
                    n = c["n_objetos"].get(video)
                    celdas.append("—" if valor is None else f"{_f(valor)} ({_n(n)})")
                lineas += [f"| `{v}` | {NOMBRE_METRICA[m]} | " + " | ".join(celdas) + " |"]

    difs = resumen["diferencias_contra_base"]
    if difs:
        lineas += ["", "#### Diferencia pareada contra `base` (variante − base, por fold)", ""]
        lineas += [
            "| Variante | Clase | Métrica | Media | Mín | Máx | Folds mejor/peor/empate "
            "| p signos | Veredicto |"
        ]
        lineas += ["|---|---|---|---|---|---|---|---|---|"]
        for v, por_clase in difs.items():
            for clase in CLASES:
                for m in ("ap50", "ap50_95", "recall"):
                    d = por_clase[clase][m]
                    lineas += [
                        f"| `{v}` | `{clase}` | {NOMBRE_METRICA[m]} | {_f(d['media'], True)} "
                        f"| {_f(d['minimo'], True)} | {_f(d['maximo'], True)} "
                        f"| {d['folds_mejora']}/{d['folds_empeora']}/{d['folds_empate']} "
                        f"de {d['n_folds']} | {_f(d['p_signo'])} | {d['veredicto']} |"
                    ]
        regla = resumen["decision"]["regla"]
        lineas += [
            "",
            f"Criterio: un fold mejora si Δ > {regla['empate']}; la métrica mejora si lo "
            f"hacen al menos {regla['folds_minimos']} folds y la media es ≥ "
            f"{regla['efecto_minimo']}. Con 4 folds ni 4 de 4 es significativo al 5 % "
            "(p de signos bilateral = 0.125): es un criterio de consistencia, no una prueba.",
        ]

    dec = resumen["decision"]
    lineas += ["", "#### Decisión", "", f"**Variante elegida: `{dec['ganadora']}`.** "]
    lineas[-1] += dec["motivo"]
    for v, e in dec["evaluacion"].items():
        estado = "candidata" if e["candidata"] else "descartada: " + "; ".join(e["motivos"])
        lineas += [f"- `{v}`: {estado}."]

    umbrales = resumen["umbral_fuera_de_fold"]
    ganadora = dec["ganadora"]
    if ganadora in umbrales:
        u = umbrales[ganadora]
        cr = u["criterio"]
        lineas += [
            "",
            f"#### Umbral de `{u['clase']}` fuera de fold (`{ganadora}`)",
            "",
            f"Criterio: `{cr['tipo']}` con mínimo {cr['minimo']}. Para cada fold el umbral "
            "se elige con los barridos de los otros tres y se mide en el que queda fuera.",
            "",
            "| Fold | Umbral | Elegido con | Precisión | Recall | F1 | Objetos |",
            "|---|---|---|---|---|---|---|",
        ]
        for video, f in u["por_fold"].items():
            aviso = "" if f["cumple_en_otros"] else " (no cumple el mínimo)"
            aviso += " (borde de la grilla)" if f.get("en_borde_de_grilla") else ""
            lineas += [
                f"| {video} | {_u(f['umbral'])}{aviso} | {', '.join(f['elegido_con'])} "
                f"| {_f(f['precision'])} | {_f(f['recall'])} | {_f(f['f1'])} "
                f"| {_n(f['n_objetos'])} |"
            ]
        lineas += [
            f"| media ± desv. | | | {_md(u['precision'])} | {_md(u['recall'])} | {_md(u['f1'])} | |"
        ]
        cj = u["conjunto_fuera_de_fold"]
        if cj:
            lineas += [
                "",
                f"Sumando los cuatro folds, cada uno con su umbral fuera de fold: precisión "
                f"{_f(cj['precision'])}, recall {_f(cj['recall'])} "
                f"({_n(cj['tp'])} de {_n(cj['n_objetos'])} "
                "objetos).",
            ]
        p = u["produccion"]
        lineas += [
            "",
            f"**Umbral para producción**, elegido con los cuatro folds: **{_u(p['umbral'])}**"
            + ("" if p["cumple"] else " (ningún umbral cumple el mínimo; es el de máximo F1)")
            + (
                " (en el borde inferior de la grilla: el óptimo puede estar más abajo)"
                if p.get("en_borde_de_grilla")
                else ""
            )
            + f". Su rendimiento esperado es el fuera de fold de arriba. En `configs/`: "
            f"`detector.class_confidence: {{{u['clase']}: {_u(p['umbral'])}}}`.",
        ]
    lineas += [
        "",
        "**Qué no se concluye.** Cuatro videos generados no son una muestra de minas; una "
        "diferencia marcada como ruido no es evidencia de que las variantes sean iguales, "
        "sólo de que estos datos no las distinguen.",
    ]
    return "\n".join(lineas) + "\n"
