"""Punto de entrada de BermGuard.

Interpreta la línea de comandos, resuelve la configuración y delega en el
orquestador. No contiene lógica de procesamiento: separarla permite ejecutar el
pipeline desde un test o un notebook sin pasar por ``argparse``.

El contrato de ejecución es el comando del enunciado, que debe funcionar tal cual:

    docker run --rm -v /ruta/test:/app/test -v /ruta/output:/app/output \\
      imagen \\
      python main.py --input /app/test --output /app/output --method 1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from bermguard.core.config import config_for_method
from bermguard.core.device import describe_runtime, resolve_device
from bermguard.core.exceptions import BermGuardError
from bermguard.io.artifacts import RunMetadata, code_version, write_json
from bermguard.io.video_reader import discover_videos
from bermguard.pipeline.factory import METODOS_DISPONIBLES, build_orchestrator

logger = logging.getLogger("bermguard")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Inspeccion de pretiles y monitoreo de proximidad en botaderos.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Directorio con videos")
    parser.add_argument("--output", type=Path, required=True, help="Directorio de salida")
    parser.add_argument(
        "--method",
        default="1",
        help=(
            "Metodo a ejecutar: "
            + ", ".join(str(m) for m in METODOS_DISPONIBLES)
            + ", o 'all' para ejecutarlos todos sobre cada video"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Dispositivo de computo. 'auto' usa CUDA si esta disponible",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs"),
        help="Directorio de archivos de configuracion",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Tope de frames por video, para pruebas rapidas",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def resolve_methods(spec: str) -> list[int]:
    """Traduce el argumento ``--method`` a la lista de métodos a ejecutar.

    ``all`` es el argumento «todo en 1» que exige el enunciado.
    """
    if spec.strip().lower() == "all":
        return list(METODOS_DISPONIBLES)
    try:
        metodo = int(spec)
    except ValueError:
        raise SystemExit(f"--method invalido: {spec!r}. Use un numero o 'all'.") from None
    if metodo not in METODOS_DISPONIBLES:
        raise SystemExit(f"Metodo {metodo} no disponible. Opciones: {METODOS_DISPONIBLES} o 'all'.")
    return [metodo]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    inicio = time.perf_counter()
    metodos = resolve_methods(args.method)
    device = resolve_device(args.device)

    try:
        videos = discover_videos(args.input)
    except BermGuardError as exc:
        logger.error("%s", exc)
        return 2

    if not videos:
        logger.error("No se encontraron videos en %s", args.input)
        return 2

    logger.info(
        "%d video(s) en %s | metodos: %s | dispositivo: %s",
        len(videos),
        args.input,
        metodos,
        device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    procesados = 0
    fallidos: list[dict[str, str]] = []

    for metodo in metodos:
        try:
            config = config_for_method(metodo, args.configs)
            orquestador = build_orchestrator(config, device)
        except BermGuardError as exc:
            logger.error("Metodo %d no se pudo inicializar: %s", metodo, exc)
            fallidos.append({"method": str(metodo), "video": "*", "error": str(exc)})
            continue

        for video in videos:
            destino = args.output / video.stem / f"method_{metodo}"
            try:
                orquestador.process_video(video, destino, max_frames=args.max_frames)
                procesados += 1
            except BermGuardError as exc:
                # Un video corrupto no puede tumbar el lote. Se registra el fallo
                # con su motivo y se continua con el siguiente.
                logger.error("Fallo %s con el metodo %d: %s", video.name, metodo, exc)
                fallidos.append({"method": str(metodo), "video": video.name, "error": str(exc)})
            except Exception:
                # Cualquier otra excepcion es un bug, no una condicion esperada. Se
                # registra con traza completa y se continua, para que un defecto en
                # un video no impida producir los artefactos de los demas.
                logger.exception("Error inesperado en %s con el metodo %d", video.name, metodo)
                fallidos.append(
                    {
                        "method": str(metodo),
                        "video": video.name,
                        "error": "excepcion inesperada, ver log",
                    }
                )

    transcurrido = time.perf_counter() - inicio
    write_json(
        RunMetadata(
            command=" ".join(sys.argv),
            input_dir=str(args.input),
            output_dir=str(args.output),
            methods=metodos,
            videos_found=len(videos),
            videos_processed=procesados,
            videos_failed=fallidos,
            total_seconds=transcurrido,
            runtime=describe_runtime(device),
            code_version=code_version(),
        ),
        args.output / "run_metadata.json",
    )

    logger.info(
        "Terminado en %.1fs: %d procesado(s), %d fallo(s)",
        transcurrido,
        procesados,
        len(fallidos),
    )
    # Se devuelve 0 mientras haya salida util: el lote cumplio su proposito aunque
    # algun video individual haya fallado, y el detalle queda en run_metadata.json.
    return 0 if procesados else 1


if __name__ == "__main__":
    raise SystemExit(main())
