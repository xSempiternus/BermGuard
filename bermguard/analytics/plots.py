"""Generación de los gráficos técnicos del enunciado.

Produce las dos figuras exigidas, cada una en PNG y SVG, acompañadas del CSV con
los datos crudos que las originan.

## Decisiones de diseño visual, y por qué

**El riesgo se codifica por forma *y* color, no sólo por color.** El trío
verde/ámbar/rojo falla la separación bajo deuteranopía —la pareja rojo-verde da un
ΔE de 4.1, muy por debajo del umbral utilizable—, de modo que un lector con
deficiencia de visión al color no distinguiría un equipo seguro de uno en alerta
crítica. La forma del marcador resuelve el problema de raíz, y el color pasa a
reforzar en lugar de a informar en solitario.

**Las series de una sola variable no llevan leyenda.** El título nombra lo que se
grafica; una caja de leyenda con una entrada es ruido.

**Los huecos se dibujan como huecos.** Donde no hay medición, la línea se
interrumpe. Interpolar sobre un corte de escena uniría mediciones de cámaras
distintas con una recta que sugiere continuidad donde no la hay.

**Rejilla y ejes recesivos.** El dato es lo que tiene que destacar.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Sin servidor gráfico: el pipeline corre en un contenedor.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from bermguard.core.types import RiskLevel

logger = logging.getLogger(__name__)

# --- Paleta -------------------------------------------------------------------
# Instancia de la paleta de referencia validada. Superficie clara elegida
# deliberadamente: el entregable son imagenes estaticas que se leeran impresas o
# incrustadas en el reporte, no una pagina que siga el tema del lector.

SUPERFICIE = "#fcfcfb"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#8a8880"

SERIE_1 = "#2a78d6"
BANDA_1 = "#b7d3f6"

ESTADO = {
    RiskLevel.SAFE: ("#0ca30c", "o", "seguro"),
    RiskLevel.CAUTION: ("#fab219", "^", "precaucion"),
    RiskLevel.CRITICAL: ("#d03b3b", "s", "critico"),
}
"""Color, **forma** y etiqueta por nivel de riesgo.

La forma no es decorativa: el par rojo-verde de la paleta de estado no supera la
separación mínima bajo deuteranopía, así que el color no puede portar el
significado en solitario.
"""

RAMPA_SECUENCIAL = "Blues"
"""Rampa de un solo tono para magnitud. Nunca un arcoíris."""

MAX_EQUIPOS_EN_FIGURA = 8
"""Tope de equipos representados en las figuras.

No es estético. Un video con cortes de escena acumula decenas de identificadores
—el tracker no los reutiliza tras un reinicio, para no confundir dos máquinas
distintas bajo el mismo número—, y una matriz de 31×31 con celdas de pocos píxeles
y etiquetas solapadas no comunica nada. Se conservan los equipos con más
observaciones, que son los que estuvieron realmente en escena, y los omitidos se
declaran en la figura en lugar de desaparecer sin más.
"""


@dataclass
class VideoSeries:
    """Series acumuladas de un video, insumo de las figuras y de los CSV."""

    timestamps_s: list[float] = field(default_factory=list)
    height_m: list[float] = field(default_factory=list)
    height_lo_m: list[float] = field(default_factory=list)
    height_hi_m: list[float] = field(default_factory=list)
    shot: list[int] = field(default_factory=list)

    positions: list[tuple[int, float, float, RiskLevel]] = field(default_factory=list)
    """``(track_id, lateral_m, profundidad_m, riesgo)`` por observación."""

    min_distances: dict[tuple[int, int], float] = field(default_factory=dict)
    """Distancia mínima registrada por par de equipos a lo largo del video."""

    def principales(self, tope: int = MAX_EQUIPOS_EN_FIGURA) -> list[int]:
        """Equipos con más observaciones, hasta ``tope``.

        El conteo de observaciones separa los equipos que estuvieron en escena de
        los identificadores efímeros que produce un detector con falsos positivos y
        un video con cortes.
        """
        conteo: dict[int, int] = {}
        for track_id, _x, _z, _r in self.positions:
            conteo[track_id] = conteo.get(track_id, 0) + 1
        return sorted(sorted(conteo, key=lambda t: -conteo[t])[:tope])

    def record_frame(
        self,
        timestamp_s: float,
        shot: int,
        height_m: float | None,
        height_lo_m: float | None,
        height_hi_m: float | None,
    ) -> None:
        self.timestamps_s.append(timestamp_s)
        self.shot.append(shot)
        # NaN y no un cero de relleno: el cero se colaria como una medicion valida
        # y aplanaria la curva, mientras el NaN interrumpe la linea.
        self.height_m.append(height_m if height_m is not None else float("nan"))
        self.height_lo_m.append(height_lo_m if height_lo_m is not None else float("nan"))
        self.height_hi_m.append(height_hi_m if height_hi_m is not None else float("nan"))

    def record_positions(
        self,
        world_xy: Mapping[int, tuple[float, float]],
        risks: Mapping[int, RiskLevel],
    ) -> None:
        for track_id, (lateral, profundidad) in world_xy.items():
            self.positions.append(
                (track_id, lateral, profundidad, risks.get(track_id, RiskLevel.SAFE))
            )

    def record_distances(self, distances: Mapping[tuple[int, int], float]) -> None:
        for par, distancia in distances.items():
            previo = self.min_distances.get(par)
            if previo is None or distancia < previo:
                self.min_distances[par] = distancia


def _preparar_eje(ax: plt.Axes, titulo: str, xlabel: str, ylabel: str) -> None:
    """Aplica el estilo común: rejilla y ejes recesivos, sin marco superior."""
    ax.set_title(titulo, color=TINTA_PRIMARIA, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=TINTA_SECUNDARIA, fontsize=9)
    ax.set_ylabel(ylabel, color=TINTA_SECUNDARIA, fontsize=9)
    ax.grid(True, color=TINTA_TENUE, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_TENUE)
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=8, length=3, width=0.8)


def _guardar(fig: plt.Figure, destino: Path) -> None:
    """Escribe la figura en PNG y SVG.

    Ambos formatos porque el enunciado pide «imagen o vectorial»: el PNG sirve para
    incrustar y el SVG para imprimir o ampliar sin pérdida.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            destino.with_suffix(f".{extension}"),
            dpi=160,
            facecolor=SUPERFICIE,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_berm_height(series: VideoSeries, destino: Path, minimo_normativo_m: float = 1.5) -> None:
    """Curva temporal de la altura del pretil, con banda de incertidumbre.

    Una sola serie, así que no lleva leyenda: el título la nombra. La banda no es
    decorativa — sin calibración de cámara el valor puntual sobreestima lo que el
    método sabe, y publicar la línea sola afirmaría una precisión inexistente.
    """
    tiempos = np.asarray(series.timestamps_s)
    altura = np.asarray(series.height_m)
    if tiempos.size == 0 or not np.isfinite(altura).any():
        logger.info("Sin alturas medidas; se omite %s", destino.name)
        return

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)

    ax.fill_between(
        tiempos,
        np.asarray(series.height_lo_m),
        np.asarray(series.height_hi_m),
        color=BANDA_1,
        alpha=0.55,
        linewidth=0,
        label="_nolegend_",
    )
    ax.plot(tiempos, altura, color=SERIE_1, linewidth=2.0, solid_capstyle="round")

    # Fronteras de toma: la curva no es comparable a traves de un corte porque la
    # camara cambia. Marcarlas evita leer una discontinuidad como degradacion real.
    tomas = np.asarray(series.shot)
    for indice in np.flatnonzero(np.diff(tomas)) + 1:
        ax.axvline(tiempos[indice], color=TINTA_TENUE, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(
            f"corte · toma {tomas[indice]}",
            xy=(tiempos[indice], ax.get_ylim()[1]),
            xytext=(4, -10),
            textcoords="offset points",
            color=TINTA_SECUNDARIA,
            fontsize=7.5,
        )

    ax.axhline(minimo_normativo_m, color=TINTA_SECUNDARIA, linewidth=1.0, linestyle=(0, (2, 2)))
    ax.annotate(
        f"referencia normativa ~{minimo_normativo_m:g} m",
        xy=(tiempos[0], minimo_normativo_m),
        xytext=(2, 4),
        textcoords="offset points",
        color=TINTA_SECUNDARIA,
        fontsize=7.5,
    )

    mediana = float(np.nanmedian(altura))
    _preparar_eje(
        ax,
        f"Evolución de la altura del pretil  ·  mediana {mediana:.2f} m",
        "tiempo (s)",
        "altura sobre la rasante (m)",
    )
    ax.annotate(
        "banda: incertidumbre del ancla métrica",
        xy=(0.99, 0.03),
        xycoords="axes fraction",
        ha="right",
        color=TINTA_SECUNDARIA,
        fontsize=7.5,
    )
    _guardar(fig, destino)


def plot_spatial_and_distances(series: VideoSeries, destino: Path) -> None:
    """Dispersión espacial de la maquinaria y matriz de distancias mínimas.

    Dos paneles porque responden preguntas distintas: el mapa dice **dónde** ocurrió
    el riesgo, la matriz dice **entre quiénes** y cuánto se acercaron.
    """
    if not series.positions:
        logger.info("Sin posiciones proyectadas; se omite %s", destino.name)
        return

    fig, (ax_mapa, ax_matriz) = plt.subplots(
        1, 2, figsize=(13, 5.2), facecolor=SUPERFICIE, gridspec_kw={"width_ratios": [1.35, 1]}
    )
    for ax in (ax_mapa, ax_matriz):
        ax.set_facecolor(SUPERFICIE)

    # --- Panel A: vista cenital ---
    for nivel, (color, forma, etiqueta) in ESTADO.items():
        puntos = [(x, z) for _t, x, z, r in series.positions if r is nivel]
        if not puntos:
            continue
        xs, zs = zip(*puntos, strict=True)
        ax_mapa.scatter(
            xs,
            zs,
            s=34,
            marker=forma,
            facecolor=color,
            edgecolor=SUPERFICIE,
            linewidth=0.8,
            alpha=0.85,
            label=etiqueta,
        )

    # Etiqueta directa por equipo en su ultima posicion: la identidad no depende del
    # color, que aqui codifica el riesgo.
    principales = set(series.principales())
    ultimas: dict[int, tuple[float, float]] = {}
    for track_id, x, z, _r in series.positions:
        if track_id in principales:
            ultimas[track_id] = (x, z)
    for track_id, (x, z) in ultimas.items():
        ax_mapa.annotate(
            f"#{track_id}",
            xy=(x, z),
            xytext=(6, 5),
            textcoords="offset points",
            color=TINTA_PRIMARIA,
            fontsize=8,
            weight="medium",
        )

    ax_mapa.scatter([0], [0], marker="*", s=90, color=TINTA_SECUNDARIA, zorder=5)
    ax_mapa.annotate(
        "cámara",
        xy=(0, 0),
        xytext=(6, -10),
        textcoords="offset points",
        color=TINTA_SECUNDARIA,
        fontsize=7.5,
    )
    _preparar_eje(
        ax_mapa,
        "Distribución espacial de maquinaria (vista cenital)",
        "lateral respecto al eje óptico (m)",
        "profundidad (m)",
    )
    leyenda = ax_mapa.legend(
        frameon=False, fontsize=8, labelcolor=TINTA_SECUNDARIA, loc="upper left", title="riesgo"
    )
    leyenda.get_title().set_color(TINTA_SECUNDARIA)
    leyenda.get_title().set_fontsize(8)

    # --- Panel B: matriz de distancias minimas ---
    todos = sorted({t for t, _x, _z, _r in series.positions})
    ids = series.principales()
    omitidos = len(todos) - len(ids)
    matriz = np.full((len(ids), len(ids)), np.nan)
    for (a, b), distancia in series.min_distances.items():
        if a in ids and b in ids:
            i, j = ids.index(a), ids.index(b)
            matriz[i, j] = matriz[j, i] = distancia

    imagen = ax_matriz.imshow(matriz, cmap=RAMPA_SECUENCIAL, origin="upper")
    ax_matriz.set_xticks(range(len(ids)), [f"#{i}" for i in ids])
    ax_matriz.set_yticks(range(len(ids)), [f"#{i}" for i in ids])
    ax_matriz.grid(False)

    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j or not np.isfinite(matriz[i, j]):
                continue
            # El texto va en tinta y no en el color de la celda: sobre una rampa
            # secuencial hay que elegir claro u oscuro segun el fondo local.
            fondo_oscuro = matriz[i, j] < np.nanmax(matriz) * 0.45
            ax_matriz.text(
                j,
                i,
                f"{matriz[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color=SUPERFICIE if fondo_oscuro else TINTA_PRIMARIA,
            )

    titulo_matriz = "Distancia mínima registrada entre equipos"
    if omitidos:
        titulo_matriz += f"  ·  {len(ids)} de {len(todos)} equipos"
    _preparar_eje(ax_matriz, titulo_matriz, "", "")
    ax_matriz.set_xlabel("")
    barra = fig.colorbar(imagen, ax=ax_matriz, fraction=0.045, pad=0.04)
    barra.set_label("metros", color=TINTA_SECUNDARIA, fontsize=8)
    barra.ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=8)
    barra.outline.set_visible(False)

    nota = "celdas vacías: equipos nunca visibles a la vez"
    if omitidos:
        nota += f"  ·  {omitidos} equipo(s) de menor permanencia omitido(s)"
    ax_matriz.annotate(
        nota,
        xy=(0.0, -0.16),
        xycoords="axes fraction",
        color=TINTA_SECUNDARIA,
        fontsize=7.5,
    )

    fig.tight_layout()
    _guardar(fig, destino)


def distance_rows(series: VideoSeries) -> list[dict[str, object]]:
    """Filas de la matriz de distancias, para el CSV que acompaña a la figura."""
    return [
        {"track_a": a, "track_b": b, "min_distance_m": round(d, 2)}
        for (a, b), d in sorted(series.min_distances.items())
    ]


def _legend_handles() -> Sequence[Line2D]:
    """Manejadores de leyenda con forma y color, para reutilizar en el reporte."""
    return [
        Line2D([], [], marker=forma, color="none", markerfacecolor=color, label=etiqueta)
        for color, forma, etiqueta in ESTADO.values()
    ]
