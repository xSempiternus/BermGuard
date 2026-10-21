"""Sesión de GPU completa: chequeos previos, validación cruzada LOVO y resumen.

La GPU de desarrollo (4 GB) se comparte con el escritorio, así que el entrenamiento
corre en ventanas en que está libre. Este script junta lo que hay que hacer en esa
ventana en un solo comando y verifica antes de empezar lo que haría fallar una corrida
de horas a la mitad.

Se puede cortar con Ctrl+C: al volver a lanzarlo, ``cross_validate.py`` salta el
preentrenamiento y los folds terminados.

Uso:
    python scripts/gpu_session.py --check     # sólo los chequeos, sin tocar la GPU
    python scripts/gpu_session.py             # chequeos, LOVO (+ base2x si hace falta), resumen
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

VARIANTES = ("base", "copypaste", "pretrain")
VRAM_MINIMA_MB = 2800  # con batch 4 el entrenamiento usa ~2.5 GB
DISCO_MINIMO_GB = 5  # 13-17 entrenamientos (con base2x), checkpoints de 20-80 MB, folds
PROCESOS_PESADOS = ("blender", "rocketleague", "valorant", "obs64")
RESUMEN = Path("docs/experimentos/lovo/resumen.json")


def vram_libre_mb() -> tuple[int, list[str]]:
    """VRAM libre y procesos con contexto en la GPU, vía nvidia-smi (sin abrir CUDA)."""
    libre = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[0]
    procesos = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    return int(libre), [Path(p.strip()).name for p in procesos if p.strip()]


def chequear(args: argparse.Namespace) -> list[str]:
    problemas: list[str] = []

    imagenes = list(Path("data/dataset/images").glob("*/*.jpg"))
    etiquetas = list(Path("data/dataset/labels").glob("*/*.txt"))
    print(f"dataset propio: {len(imagenes)} frames, {len(etiquetas)} etiquetas")
    if not imagenes or len(imagenes) != len(etiquetas):
        problemas.append("data/dataset: faltan frames o no calzan con las etiquetas")

    if not Path(args.base_model).exists():
        problemas.append(f"falta {args.base_model} (pesos COCO de partida)")

    resumen = Path("data/pretrain/resumen.json")
    if not Path("data/pretrain/data.yaml").exists() or not resumen.exists():
        problemas.append("falta data/pretrain: correr prepare_pretrain.py (ver ADR 0004)")
    else:
        datos = json.loads(resumen.read_text(encoding="utf-8"))
        train = len(list(Path("data/pretrain/images/train").glob("*")))
        print(
            f"preentrenamiento: {train} imágenes de train, fuentes {list(datos['fuentes'])}, "
            f"exclusión {datos.get('excluir')}, repetición {datos.get('repetir')}"
        )
        if not datos.get("excluir") or not datos.get("repetir"):
            problemas.append(
                "data/pretrain se generó sin --exclude o sin --repeat: regenerar con el "
                "comando del ADR 0004"
            )

    disco_gb = shutil.disk_usage(".").free / 1e9
    print(f"disco libre: {disco_gb:.1f} GB")
    if disco_gb < DISCO_MINIMO_GB:
        # Una prueba de humo ya murió con 'No space left' al guardar un checkpoint.
        problemas.append(f"disco libre {disco_gb:.1f} GB < {DISCO_MINIMO_GB} GB")

    try:
        libre, procesos = vram_libre_mb()
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        problemas.append("nvidia-smi no responde: ¿hay GPU NVIDIA y driver?")
    else:
        print(f"VRAM libre: {libre} MB")
        pesados = sorted({p for p in procesos if p.lower().startswith(PROCESOS_PESADOS)})
        if pesados:
            problemas.append(f"procesos pesados usando la GPU: {', '.join(pesados)}")
        if libre < VRAM_MINIMA_MB:
            problemas.append(f"VRAM libre {libre} MB < {VRAM_MINIMA_MB} MB: cerrar programas")
    return problemas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Sólo chequear")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTES), choices=VARIANTES)
    parser.add_argument("--base-model", default="weights/yolo11s.pt")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--force", action="store_true", help="Seguir aunque haya avisos de VRAM o procesos"
    )
    args = parser.parse_args()

    problemas = chequear(args)
    for p in problemas:
        print(f"! {p}")
    if args.check:
        sys.exit(1 if problemas else 0)
    if problemas and not args.force:
        raise SystemExit("Chequeos fallidos; corregir o lanzar con --force.")

    python = sys.executable

    def lovo(variantes: list[str]) -> None:
        subprocess.run(
            [
                python,
                "scripts/cross_validate.py",
                "--variants",
                *variantes,
                "--batch",
                str(args.batch),
                "--base-model",
                args.base_model,
            ],
            check=True,
        )
        subprocess.run([python, "scripts/summarize_lovo.py"], check=True)

    lovo(args.variants)
    # Si copypaste gana, puede ser por hacer el doble de pasos: se entrena el control
    # base2x (unas 3,4 h más) y se vuelve a decidir. Ver ADR 0009.
    if json.loads(RESUMEN.read_text(encoding="utf-8"))["decision"].get("pendiente_control"):
        print("\ncopypaste es candidata contra base: entrenando el control base2x (mismos pasos)")
        lovo(["base2x"])
    print(
        "\nListo. Revisar docs/experimentos/lovo/resumen.md: variante ganadora y umbral "
        "de bulldozer fuera de fold. Lo que sigue (reentrenar el detector v2 con la "
        "ganadora) está en docs/entrenamiento/sesion_gpu.md."
    )


if __name__ == "__main__":
    main()
