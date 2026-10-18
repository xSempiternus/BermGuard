"""Genera las imágenes de vitrina de ``docs/showcase/img`` desde los datos del repo.

Cada figura sale de un JSON de ``docs/experimentos`` o de los videos y pesos, nunca de
números copiados a mano, así que se regeneran si cambian los resultados.

Uso:
    python scripts/make_showcase.py --raw data/raw            # todo
    python scripts/make_showcase.py --only sintetico lovo     # algunas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bermguard.analytics.plots import SUPERFICIE, TINTA_PRIMARIA, TINTA_SECUNDARIA, TINTA_TENUE

SALIDA = Path("docs/showcase/img")
EXPERIMENTOS = Path("docs/experimentos")
# Validados con el script de la skill de dataviz: CVD ΔE 29.6. El naranja queda bajo
# 3:1 contra la superficie, así que toda serie lleva etiqueta directa.
AZUL, NARANJA = "#2a6fdb", "#d97f00"


def _eje(ax: plt.Axes) -> None:
    ax.set_facecolor(SUPERFICIE)
    ax.grid(True, axis="y", color=TINTA_TENUE, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_TENUE)
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=8, length=3)


def _guardar(fig: plt.Figure, nombre: str) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(SALIDA / f"{nombre}.{ext}", dpi=160, facecolor=SUPERFICIE, bbox_inches="tight")
    plt.close(fig)
    print("escrito", SALIDA / nombre)


def sintetico() -> None:
    datos = json.loads((EXPERIMENTOS / "pretil_sintetico.json").read_text(encoding="utf-8"))
    contrastes = sorted({f["contraste"] for f in datos["filas"]})
    fig, ejes = plt.subplots(1, len(contrastes), figsize=(10, 3.2), sharey=True)
    fig.patch.set_facecolor(SUPERFICIE)
    for ax, c in zip(ejes, contrastes, strict=True):
        filas = [f for f in datos["filas"] if f["contraste"] == c]
        x = [f["huellas"] for f in filas]
        for clave, color, etiqueta in (
            ("camino_optimo", AZUL, "camino óptimo"),
            ("argmax", NARANJA, "argmax + filtro"),
        ):
            y = np.array([f[clave]["within_5"] for f in filas]) * 100
            se = np.array([f[clave]["within_5_se"] for f in filas]) * 100
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4)
            if c == contrastes[-1]:
                ax.annotate(
                    etiqueta,
                    (x[-1], y[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    color=TINTA_PRIMARIA,
                    fontsize=8,
                    va="center",
                )
        _eje(ax)
        ax.set_title(
            f"contraste de la cresta {c:.0f}", color=TINTA_PRIMARIA, fontsize=9, loc="left"
        )
        ax.set_xlabel("huellas de neumático en la escena", color=TINTA_SECUNDARIA, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
    ejes[0].set_ylabel("% de columnas a ≤5 px de la cresta", color=TINTA_SECUNDARIA, fontsize=8)
    fig.suptitle(
        "Exactitud del pretil en escenas sintéticas con la cresta conocida",
        x=0.01,
        y=1.04,
        ha="left",
        color=TINTA_PRIMARIA,
        fontsize=11,
    )
    fig.text(
        0.01,
        -0.04,
        f"{datos['semillas']} escenas por punto; banda: IC 95 % entre escenas.",
        color=TINTA_SECUNDARIA,
        fontsize=7.5,
    )
    _guardar(fig, "pretil_sintetico")


def separacion(raw: Path) -> None:
    """COCO contra detector propio en el frame 120 de video_02."""
    from ultralytics import YOLO

    captura = cv2.VideoCapture(str(raw / "video_02.mp4"))
    captura.set(cv2.CAP_PROP_POS_FRAMES, 120)
    ok, frame = captura.read()
    captura.release()
    if not ok:
        raise SystemExit("No se pudo leer el frame 120 de video_02")

    paneles = []
    for pesos, clases, titulo in (
        ("weights/yolo11s.pt", [7], "YOLO11s COCO: una caja para dos máquinas"),
        ("weights/detector_v1.pt", None, "Detector propio: dos máquinas, dos cajas"),
    ):
        r = YOLO(pesos).predict(
            frame, conf=0.25, iou=0.45, classes=clases, device="cpu", verbose=False
        )[0]
        img = frame.copy()
        for caja, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), strict=True):
            x1, y1, x2, y2 = caja.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (57, 255, 106), 3)
            cv2.putText(
                img,
                f"{conf:.2f}",
                (x1 + 4, y1 + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (57, 255, 106),
                2,
            )
        paneles.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), titulo))

    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.4))
    fig.patch.set_facecolor(SUPERFICIE)
    for ax, (img, titulo) in zip(ejes, paneles, strict=True):
        ax.imshow(img[140:620, 60:1000])
        ax.set_title(titulo, color=TINTA_PRIMARIA, fontsize=9.5, loc="left")
        ax.axis("off")
    _guardar(fig, "separacion_coco_vs_propio")


def auditoria() -> None:
    """Seis bulldozers que estaban etiquetados como caex."""
    datos = json.loads(Path("data/auditoria_etiquetas.json").read_text(encoding="utf-8"))
    elegidos = datos["cambios"][::5][:6]
    fig, ejes = plt.subplots(1, len(elegidos), figsize=(12, 2.4))
    fig.patch.set_facecolor(SUPERFICIE)
    for ax, cambio in zip(ejes, elegidos, strict=True):
        particion, nombre = cambio["archivo"].split("/")
        img = cv2.imread(f"data/dataset/images/{particion}/{Path(nombre).stem}.jpg")
        alto, ancho = img.shape[:2]
        _, cx, cy, w, h = (float(v) for v in cambio["original"].split())
        x1, y1 = int((cx - w / 2) * ancho), int((cy - h / 2) * alto)
        x2, y2 = int((cx + w / 2) * ancho), int((cy + h / 2) * alto)
        m = int(0.15 * (x2 - x1))
        recorte = img[max(0, y1 - m) : y2 + m, max(0, x1 - m) : x2 + m]
        ax.imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{nombre[:15]}\ncaex → bulldozer", color=TINTA_SECUNDARIA, fontsize=7.5)
        ax.axis("off")
    fig.suptitle(
        f"Auditoría de etiquetas: {len(datos['cambios'])} bulldozers estaban anotados como CAEX",
        x=0.01,
        ha="left",
        color=TINTA_PRIMARIA,
        fontsize=11,
        y=1.08,
    )
    _guardar(fig, "auditoria_etiquetas")


def lovo() -> None:
    """AP@0.5 por fold y clase, base contra copy-paste."""
    variantes = (("base", AZUL, "sin aumento"), ("copypaste", NARANJA, "con copy-paste"))
    folds = ("video_01", "video_02", "video_03", "video_04")
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    fig.patch.set_facecolor(SUPERFICIE)
    for ax, clase in zip(ejes, ("caex", "bulldozer"), strict=True):
        for k, (variante, color, etiqueta) in enumerate(variantes):
            valores, bajos, altos = [], [], []
            for fold in folds:
                archivo = EXPERIMENTOS / "lovo" / f"{variante}_{fold}.json"
                m = json.loads(archivo.read_text(encoding="utf-8"))["por_clase"][clase]
                valores.append(m["ap50"])
                bajos.append(m["ap50"] - m["ic95"]["ap50"][0])
                altos.append(m["ic95"]["ap50"][1] - m["ap50"])
            x = np.arange(len(folds)) + (k - 0.5) * 0.36
            ax.bar(x, valores, width=0.34, color=color, label=etiqueta)
            ax.errorbar(
                x,
                valores,
                yerr=[bajos, altos],
                fmt="none",
                ecolor=TINTA_SECUNDARIA,
                elinewidth=0.8,
                capsize=2,
            )
        _eje(ax)
        ax.set_xticks(np.arange(len(folds)), [f.replace("video_", "fuera: v") for f in folds])
        ax.set_title(clase, color=TINTA_PRIMARIA, fontsize=9.5, loc="left")
        ax.set_ylim(0, 1.05)
    ejes[0].set_ylabel("AP@0.5 en el video excluido", color=TINTA_SECUNDARIA, fontsize=8)
    ejes[1].legend(frameon=False, fontsize=8, labelcolor=TINTA_PRIMARIA)
    fig.suptitle(
        "Validación cruzada dejando un video fuera",
        x=0.01,
        ha="left",
        color=TINTA_PRIMARIA,
        fontsize=11,
    )
    _guardar(fig, "lovo_ap50")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    tareas = {
        "sintetico": sintetico,
        "separacion": lambda: separacion(args.raw),
        "auditoria": auditoria,
        "lovo": lovo,
    }
    for nombre, tarea in tareas.items():
        if args.only is None or nombre in args.only:
            tarea()


if __name__ == "__main__":
    main()
