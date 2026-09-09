"""Lectura de fuentes de video.

Lo único que se asume del directorio de entrada es que es un directorio. La
resolución, la tasa de frames, el códec y el conteo de frames varían entre
archivos —sólo el set de muestra ya mezcla 1920x1080 a 30 fps con 1280x720 a
24 fps— y un directorio entregado para evaluación ciega puede además contener
archivos que no son video.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from bermguard.core.exceptions import VideoReadError
from bermguard.core.types import ImageBGR, VideoInfo

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
)
"""Extensiones tratadas como candidatas a video. Cualquier otra cosa en el
directorio de entrada se ignora en vez de intentarse: un README o un .DS_Store
perdido no debe aparecer como un fallo de procesamiento."""


def discover_videos(directory: Path) -> list[Path]:
    """Lista los archivos de video candidatos en ``directory``, ordenados por nombre.

    Ordenar hace determinista la corrida: el mismo directorio produce siempre el
    mismo orden de procesamiento, de modo que los logs y los metadatos son
    comparables entre ejecuciones.

    Raises:
        VideoReadError: Si ``directory`` no existe o no es un directorio.
    """
    if not directory.exists():
        raise VideoReadError(f"El directorio de entrada no existe: {directory}")
    if not directory.is_dir():
        raise VideoReadError(f"La ruta de entrada no es un directorio: {directory}")

    videos = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    skipped = sum(1 for p in directory.iterdir() if p.is_file()) - len(videos)
    if skipped:
        logger.info("Se ignoraron %d archivo(s) que no son video en %s", skipped, directory)
    return videos


class VideoReader:
    """Iterador perezoso de frames sobre un archivo de video.

    Se usa como context manager para que la captura se libere siempre, incluso si
    el pipeline lanza una excepción a mitad del archivo.

    Los frames se entregan de a uno y nunca se acumulan. Un clip 1080p de diez
    segundos son unos 1.8 GB decodificados; materializar un directorio completo
    agotaría la memoria de cualquier máquina sin ningún beneficio, ya que todas
    las etapas trabajan frame a frame.

    Example:
        >>> with VideoReader(Path("data/raw/video_01.mp4")) as reader:
        ...     for index, timestamp_s, frame in reader.frames():
        ...         ...
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._capture: cv2.VideoCapture | None = None
        self._info: VideoInfo | None = None

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        """Abre la captura y lee los metadatos del contenedor.

        Raises:
            VideoReadError: Si el archivo no se puede abrir, reporta un frame de
                tamaño cero, o falla al decodificar su primer frame.
        """
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise VideoReadError(f"No se pudo abrir el video: {self._path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            capture.release()
            raise VideoReadError(f"El video reporta un frame de tamaño cero: {self._path}")

        # Algunos contenedores reportan 0 o NaN. Una tasa de frames incorrecta
        # corrompe en silencio todos los timestamps río abajo, así que se corrige
        # acá, una sola vez, y se deja registrado.
        if not np.isfinite(fps) or fps <= 0:
            logger.warning(
                "El video %s reporta una tasa de frames inválida (%r); se asume 25 fps. "
                "Los timestamps de este archivo son aproximados.",
                self._path.name,
                fps,
            )
            fps = 25.0

        self._capture = capture
        self._info = VideoInfo(
            path=str(self._path),
            width=width,
            height=height,
            fps=fps,
            frame_count=max(declared_frames, 0),
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def info(self) -> VideoInfo:
        """Metadatos del contenedor. Disponibles sólo con el lector abierto."""
        if self._info is None:
            raise VideoReadError(f"El lector de {self._path} no está abierto")
        return self._info

    def frames(self, max_frames: int | None = None) -> Iterator[tuple[int, float, ImageBGR]]:
        """Entrega ``(index, timestamp_s, frame)`` hasta agotar la fuente.

        Args:
            max_frames: Tope opcional, para smoke tests e iteración rápida.
                ``None`` procesa el archivo completo.

        Note:
            La iteración se detiene en la primera lectura fallida y no al llegar
            al conteo de frames declarado por el contenedor, que habitualmente
            está mal o ausente. Un archivo truncado entrega entonces lo que tiene
            en vez de lanzar excepción, y el faltante queda en el log.

            Los timestamps se derivan como ``index / fps`` en vez de leerse de la
            posición del decodificador, que no es confiable en fuentes con tasa
            de frames variable.
        """
        if self._capture is None or self._info is None:
            raise VideoReadError(f"El lector de {self._path} no está abierto")

        fps = self._info.fps
        index = 0
        while max_frames is None or index < max_frames:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield index, index / fps, frame
            index += 1

        if index == 0:
            raise VideoReadError(f"El video decodificó cero frames: {self._path}")

        # Sólo tiene sentido en una lectura completa: un conteo corto bajo un
        # max_frames explícito es intención del llamador, no un archivo dañado.
        declared = self._info.frame_count
        if max_frames is None and declared and index < declared:
            logger.warning(
                "El video %s declaraba %d frames pero se decodificaron %d; "
                "la fuente puede estar truncada.",
                self._path.name,
                declared,
                index,
            )
