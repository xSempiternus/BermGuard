"""Detector de maquinaria basado en YOLO.

Envuelve a Ultralytics detrás de :class:`~bermguard.core.interfaces.IDetector`,
de modo que el resto del pipeline nunca la importa. Ese aislamiento es lo que
permite intercambiar un segundo detector para el benchmark sin tocar el
orquestador, y mantiene una dependencia pesada fuera de los módulos que deben
seguir siendo testeables sin ella.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

from bermguard.core.exceptions import ModelLoadError
from bermguard.core.types import BBox, Detection, ImageBGR, VehicleClass

logger = logging.getLogger(__name__)

DEFAULT_COCO_CLASS_MAP: Mapping[int, VehicleClass] = {
    7: VehicleClass.CAEX,  # "truck" en COCO
    5: VehicleClass.UNKNOWN,  # "bus" — la maquinaria grande suele leerse así
    6: VehicleClass.UNKNOWN,  # "train" — ídem, para cuerpos alargados con oruga
}
"""Mapeo de id de clase COCO a clase del dominio, para un modelo de estantería.

COCO no contiene maquinaria minera. Un camión de extracción de 300 toneladas no
es un camión de reparto, y un tractor de oruga no tiene ninguna clase en COCO, así
que este mapeo es una aproximación deliberada y su tasa de error pertenece al
reporte de benchmark. Separar bien CAEX de bulldozer requiere fine-tuning sobre
imágenes del dominio; hasta entonces se usa ``UNKNOWN`` con honestidad en vez de
adivinar una etiqueta.
"""


class YoloDetector:
    """Detector de una etapa, sobre un frame a la vez.

    Satisface :class:`~bermguard.core.interfaces.IDetector` estructuralmente; no
    hereda de él.
    """

    def __init__(
        self,
        weights: Path | str,
        device: str,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        half_precision: bool = True,
        class_map: Mapping[int, VehicleClass] | None = None,
    ) -> None:
        """
        Args:
            weights: Ruta a un checkpoint ``.pt``, o un nombre de modelo que
                Ultralytics sepa resolver. Dentro del contenedor es siempre una
                ruta ya horneada: no se descarga nada en tiempo de ejecución.
            device: ``"cuda"`` o ``"cpu"``, ya resuelto por
                :func:`~bermguard.core.device.resolve_device`.
            confidence: Puntaje mínimo para reportar una detección. Se mantiene
                bajo a propósito: el tracker recupera detecciones débiles que un
                umbral más alto descartaría, lo que importa con polvo y de noche.
            iou: Umbral de solape del NMS por encima del cual dos cajas se funden.
            image_size: Lado mayor que ve el modelo. Los frames se ajustan con
                letterbox a ese tamaño, así una fuente 1080p y una 720p se
                analizan a la misma escala.
            half_precision: Usar FP16. Se ignora en CPU, donde es más lento.
            class_map: Reemplaza a :data:`DEFAULT_COCO_CLASS_MAP`.

        Raises:
            ModelLoadError: Si el checkpoint no se puede cargar.
        """
        self._class_map = dict(class_map or DEFAULT_COCO_CLASS_MAP)
        self._confidence = confidence
        self._iou = iou
        self._image_size = image_size
        self._device = device
        self._half = half_precision and device == "cuda"
        self._weights = str(weights)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - fallo de entorno
            raise ModelLoadError("Ultralytics no está instalado") from exc

        try:
            self._model = YOLO(self._weights)
            self._model.to(device)
        except Exception as exc:
            raise ModelLoadError(
                f"No se pudieron cargar los pesos '{self._weights}' en el "
                f"dispositivo '{device}'"
            ) from exc

        logger.info(
            "Detector listo: %s en %s (fp16=%s, imgsz=%d, conf=%.2f)",
            Path(self._weights).name,
            device,
            self._half,
            image_size,
            confidence,
        )

    @property
    def name(self) -> str:
        return f"yolo:{Path(self._weights).stem}"

    def detect(self, frame: ImageBGR) -> Sequence[Detection]:
        """Detecta maquinaria en ``frame``.

        Returns:
            Detecciones con ``track_id`` en ``-1``. Asignar identidad es
            responsabilidad del tracker.
        """
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            iou=self._iou,
            imgsz=self._image_size,
            device=self._device,
            half=self._half,
            classes=list(self._class_map),
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        return [
            Detection(
                track_id=-1,
                vehicle_class=self._class_map.get(int(cls), VehicleClass.UNKNOWN),
                bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
            )
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confidences, class_ids)
        ]
