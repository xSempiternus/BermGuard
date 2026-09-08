"""Evalúa si un detector open-vocabulary separa el CAEX del bulldozer.

Herramienta de desarrollo. Responde la pregunta que dejó abierta
``probe_detector.py``: el modelo preentrenado en COCO fusiona ambas máquinas en
una sola caja ``truck``, lo que inutiliza el módulo de proximidad y corrompe el
ancla métrica.

Un detector open-vocabulary no tiene una cabeza de clases fija: alinea regiones
de imagen contra embeddings de texto, así que las categorías se entregan como
prompts en tiempo de inferencia. La pregunta es si prompts del dominio
("bulldozer", "mining haul truck") logran la separación que COCO no da.

Guarda un frame anotado por cada configuración probada, porque los conteos por sí
solos ya demostraron ser engañosos: una caja que engloba dos máquinas cuenta como
una detección exitosa y sólo se descubre mirándola.

Uso:
    python scripts/probe_openvocab.py --video data/raw/video_02.mp4 --frame 120
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from bermguard.core.device import resolve_device
from bermguard.io.video_reader import VideoReader

PROMPT_SETS: dict[str, list[str]] = {
    # Especificidad creciente, con ambas máquinas compitiendo.
    "generico": ["truck", "bulldozer"],
    "dominio": ["mining haul truck", "bulldozer", "excavator"],
    "especifico": [
        "large yellow mining dump truck",
        "yellow tracked bulldozer with blade",
    ],
    # Aislamiento: el bulldozer solo, sin ninguna clase que le compita el NMS.
    # Si aun así no aparece, el problema no es la competencia entre cajas sino
    # que el embedding de texto no activa sobre esta imagen.
    "solo_dozer_1": ["bulldozer"],
    "solo_dozer_2": ["tracked bulldozer"],
    "solo_dozer_3": ["crawler dozer with blade"],
    # Categoría amplia, por si el modelo responde mejor a un término genérico.
    "amplio": ["construction vehicle"],
}
"""Prompts a evaluar, agrupados por hipótesis.

Los detectores open-vocabulary son sensibles a la redacción, y esa sensibilidad
es en sí misma un resultado: si la separación depende de acertar la frase exacta,
el método es frágil para producción y eso pertenece al reporte.

Los conjuntos ``solo_dozer_*`` existen para separar dos causas de fallo que los
conteos agregados confunden: que una caja fusionada le gane el NMS a las cajas
individuales, o que el modelo directamente no tenga representación de la máquina.
"""

COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255)]


def extract_frame(video: Path, index: int) -> cv2.typing.MatLike:
    """Devuelve un frame concreto, decodificando secuencialmente hasta él."""
    with VideoReader(video) as reader:
        for i, _timestamp, frame in reader.frames(max_frames=index + 1):
            if i == index:
                return frame.copy()
    raise SystemExit(f"El video {video} no llegó al frame {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/raw/video_02.mp4"))
    parser.add_argument("--frame", type=int, default=120)
    parser.add_argument("--weights", default="yolov8s-worldv2.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out", type=Path, default=Path("output/_probe"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from ultralytics import YOLOWorld

    device = resolve_device(args.device)  # type: ignore[arg-type]
    frame = extract_frame(args.video, args.frame)
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLOWorld(args.weights)
    model.to(device)

    for set_name, prompts in PROMPT_SETS.items():
        model.set_classes(prompts)
        result = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]

        print(f"\n=== prompts '{set_name}': {prompts} ===")
        annotated = frame.copy()
        if result.boxes is None or len(result.boxes) == 0:
            print("  sin detecciones")
        for box in result.boxes or []:
            cls = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            label = prompts[cls] if cls < len(prompts) else str(cls)
            print(f"  {label:<34} conf={conf:.2f} ancho={x2 - x1:.0f}px x=[{x1:.0f},{x2:.0f}]")

            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (int(x1), max(int(y1) - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        destination = args.out / f"openvocab_{set_name}.jpg"
        cv2.imwrite(str(destination), annotated)
        print(f"  -> {destination}")


if __name__ == "__main__":
    main()
