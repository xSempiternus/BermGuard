"""Escritura del video de salida anotado."""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType

import cv2

from bermguard.core.exceptions import VideoWriteError
from bermguard.core.types import ImageBGR

logger = logging.getLogger(__name__)

DEFAULT_FOURCC = "mp4v"
"""MPEG-4 Parte 2, el único códec siempre compilado dentro de las ruedas de OpenCV.

H.264 (``avc1``) comprime mejor pero depende de librerías del sistema que no están
garantizadas dentro de una imagen de contenedor liviana. Elegir un códec que
funciona siempre por sobre uno que funciona casi siempre es el intercambio
correcto para un entregable que tiene que correr sin retoques en la máquina de
otra persona.
"""


class VideoWriter:
    """Sumidero de frames para la salida anotada, se usa como context manager.

    Cada frame escrito debe coincidir con el ``size`` declarado en la
    construcción. OpenCV **descarta en silencio** los frames de tamaño distinto en
    vez de lanzar error, lo que produce un archivo de salida más corto que su
    entrada sin ningún error en ninguna parte — por eso el tamaño se valida acá.
    """

    def __init__(
        self,
        path: Path,
        fps: float,
        size: tuple[int, int],
        fourcc: str = DEFAULT_FOURCC,
    ) -> None:
        """
        Args:
            path: Archivo destino. Los directorios padre se crean solos.
            fps: Tasa de frames de la salida, normalmente la de la fuente.
            size: ``(ancho, alto)`` en píxeles.
            fourcc: Código de cuatro caracteres del códec.
        """
        self._path = path
        self._fps = fps
        self._size = size
        self._fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self._frames_written = 0

    def __enter__(self) -> VideoWriter:
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
        """Crea el archivo de salida.

        Raises:
            VideoWriteError: Si el codificador no se pudo inicializar.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self._path),
            cv2.VideoWriter_fourcc(*self._fourcc),  # type: ignore[attr-defined]
            self._fps,
            self._size,
        )
        if not writer.isOpened():
            writer.release()
            raise VideoWriteError(
                f"No se pudo abrir el escritor de video para {self._path} "
                f"(codec={self._fourcc}, size={self._size}, fps={self._fps})"
            )
        self._writer = writer

    def write(self, frame: ImageBGR) -> None:
        """Agrega un frame.

        Raises:
            VideoWriteError: Si el escritor está cerrado o el tamaño del frame no
                coincide con el declarado en la construcción.
        """
        if self._writer is None:
            raise VideoWriteError(f"El escritor de {self._path} no está abierto")

        height, width = frame.shape[:2]
        if (width, height) != self._size:
            raise VideoWriteError(
                f"El tamaño del frame {(width, height)} no coincide con el del "
                f"escritor {self._size}"
            )

        self._writer.write(frame)
        self._frames_written += 1

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.debug("Se escribieron %d frames en %s", self._frames_written, self._path)
