"""Segmentación del pretil por análisis de gradiente — Método 1.

Localiza la cresta y la base del pretil sin ningún modelo entrenado. Cada paso es
inspeccionable, lo que en seguridad industrial tiene valor propio: un supervisor
puede seguir por qué el sistema afirmó lo que afirmó.

El diseño responde a hechos medidos sobre el material, no a una receta general.
Están documentados en ``docs/analisis_material.md``:

1. **La energía de gradiente vertical es máxima en el primer plano**, no en el
   pretil. Las huellas de neumático y las sombras largas producen respuestas más
   fuertes que la propia cresta.
2. **Exigir coherencia horizontal no separa nada**, porque las huellas también son
   horizontales.
3. **El contraste del pretil es bajo**: un banco de tierra oscura sobre suelo del
   mismo color, no un borde limpio.

De ahí las cuatro decisiones que definen el método:

**Escala gruesa con signo.** Se deriva sobre una versión suavizada de la imagen,
lo que equivale a una derivada de gaussiana: un escalón estructural sobrevive y la
textura de alta frecuencia se cancela. Se conserva sólo el signo positivo, porque
la cresta oscurece hacia abajo mientras la textura oscila de signo.

**Exclusión de la maquinaria.** El pretil es terreno. Una máquina produce bordes
mucho más marcados que un banco de tierra, así que sus cajas se anulan en la
respuesta antes de buscar.

**Banda acotada por la rasante.** Los puntos de contacto de la maquinaria definen
el plano transitable, y el pretil está por encima de él en el espacio imagen.

**Continuidad impuesta durante la búsqueda, no después.** Un ``argmax`` por columna
decide cada columna de forma independiente y produce saltos que ningún filtro
posterior arregla, porque la información de continuidad ya se perdió. En su lugar
se busca el camino de máxima respuesta acumulada a través de las columnas, con el
salto vertical acotado entre columnas contiguas. Es programación dinámica sobre
una rejilla, y convierte el prior «el pretil es una estructura continua» en una
restricción del problema en vez de en un post-proceso.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter

from bermguard.core.types import BermPixels, Detection, FloatArray, ImageBGR

logger = logging.getLogger(__name__)


class ClassicalBermSegmenter:
    """Segmentador de pretil basado en gradiente a escala gruesa y camino óptimo.

    Satisface :class:`~bermguard.core.interfaces.IBermSegmenter`.

    Mantiene estado temporal —la cresta del frame anterior— para suavizar la serie
    entre frames. El orquestador invoca :meth:`reset` en cada corte de toma, ya que
    arrastrar ese estado a través de un corte mezclaría escenas distintas.
    """

    def __init__(
        self,
        clahe_clip: float = 2.5,
        blur_sigma: float = 6.0,
        max_step_px: int = 3,
        response_percentile: float = 55.0,
        smooth_window: int = 31,
        temporal_alpha: float = 0.3,
        sky_variance_threshold: float = 6.0,
        band_margin_frac: float = 0.04,
        box_dilation_px: int = 12,
    ) -> None:
        """
        Args:
            clahe_clip: Límite de contraste del CLAHE, aplicado sobre luminancia
                porque el pretil se distingue por brillo y no por color.
            blur_sigma: Escala del suavizado previo a derivar, en píxeles. Decide
                qué se considera estructura y qué textura: por debajo de 3 entra el
                ruido del terreno, por encima de 12 la cresta se difumina con él.
            max_step_px: Salto vertical máximo de la cresta entre dos columnas
                contiguas. Es el prior de continuidad expresado como restricción.
                Valores altos permiten seguir pretiles muy inclinados a costa de
                dejar entrar saltos hacia estructuras vecinas.
            response_percentile: Percentil de la respuesta a lo largo del camino
                por debajo del cual una columna se declara sin pretil. Umbral
                relativo y no absoluto, porque el contraste varía dos órdenes de
                magnitud entre día y noche.
            smooth_window: Ventana del filtro Savitzky-Golay sobre el perfil.
                Savitzky-Golay y no media móvil: ajusta un polinomio local, de modo
                que preserva la amplitud y la posición de las variaciones reales
                —una zona hundida del pretil es información— en vez de aplanarlas.
            temporal_alpha: Peso del frame actual en el suavizado exponencial. Bajo
                a propósito: el pretil cambia lentísimo y el ruido de medición es
                alto, así que conviene confiar más en la historia que en la
                observación instantánea. Es lo que elimina el parpadeo.
            sky_variance_threshold: Desviación estándar por fila por debajo de la
                cual se considera cielo. El cielo es la estructura horizontal más
                nítida de la escena y confundirlo con la cresta es el error fácil.
            band_margin_frac: Margen bajo la rasante, como fracción de la altura,
                para tolerar el error de la propia referencia de suelo.
            box_dilation_px: Cuánto se ensancha cada caja de detección al
                enmascararla. El borde de una máquina se extiende algo más allá de
                su caja por el desenfoque de movimiento y por el polvo que levanta.
        """
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        self._blur_sigma = blur_sigma
        self._max_step = max_step_px
        self._response_percentile = response_percentile
        self._smooth_window = smooth_window
        self._alpha = temporal_alpha
        self._sky_threshold = sky_variance_threshold
        self._band_margin = band_margin_frac
        self._box_dilation = box_dilation_px

        self._crest_previa: FloatArray | None = None

    @property
    def name(self) -> str:
        return "classical:coarse-gradient+dp"

    def reset(self) -> None:
        self._crest_previa = None

    # --- Contrato -------------------------------------------------------------

    def segment(
        self, frame: ImageBGR, detections: Sequence[Detection] = ()
    ) -> BermPixels | None:
        alto, ancho = frame.shape[:2]

        realzado = self._realzar(frame)
        respuesta = self._respuesta_estructural(realzado)
        self._anular_maquinaria(respuesta, detections)

        y_min, y_max = self._banda_de_busqueda(realzado, detections)
        if y_max - y_min < 16:
            logger.debug("Banda degenerada (%d-%d); sin pretil", y_min, y_max)
            return None

        banda = respuesta[y_min:y_max, :]
        pico = float(banda.max())
        if pico <= 0:
            return None

        cresta_local = self._camino_de_cresta(banda / pico)
        fuerza = banda[cresta_local, np.arange(ancho)]

        # Umbral relativo sobre la respuesta a lo largo del camino: donde el camino
        # pasa por zonas sin senal, no hay pretil que reportar. El camino existe
        # siempre —es un optimo global— y esta validacion es lo que impide que
        # devuelva una linea inventada sobre suelo liso.
        corte = float(np.percentile(fuerza, self._response_percentile))
        valida = fuerza > max(corte, pico * 0.05)
        if valida.sum() < ancho * 0.1:
            return None

        cresta = (cresta_local + y_min).astype(np.float32)
        cresta[~valida] = np.nan
        cresta = self._suavizar_perfil(cresta)
        cresta = self._suavizar_temporal(cresta)
        base = self._estimar_base(respuesta, cresta, y_max)

        cobertura = float(np.isfinite(cresta).sum() / ancho)
        return BermPixels(
            crest_y_px=cresta,
            base_y_px=base,
            coverage=cobertura,
            confidence=self._confianza(cobertura, cresta, detections),
        )

    # --- Etapas ---------------------------------------------------------------

    def _realzar(self, frame: ImageBGR) -> np.ndarray:
        """Convierte a luminancia y ecualiza el contraste localmente.

        CLAHE y no ecualización global: la escena nocturna tiene zonas saturadas
        por focos de faena junto a zonas casi negras, y un ajuste global arruina
        una u otra.
        """
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(gris)

    def _respuesta_estructural(self, realzado: np.ndarray) -> np.ndarray:
        """Gradiente vertical positivo a escala gruesa."""
        k = int(self._blur_sigma * 6) | 1
        suave = cv2.GaussianBlur(realzado, (k, k), self._blur_sigma)
        gy = cv2.Scharr(suave, cv2.CV_32F, 0, 1)
        return np.maximum(gy, 0.0)

    def _anular_maquinaria(
        self, respuesta: np.ndarray, detections: Sequence[Detection]
    ) -> None:
        """Pone a cero la respuesta dentro de las cajas de maquinaria, in situ."""
        alto, ancho = respuesta.shape
        d = self._box_dilation
        for deteccion in detections:
            x1, y1, x2, y2 = deteccion.bbox.as_tuple()
            respuesta[
                max(0, int(y1) - d) : min(alto, int(y2) + d),
                max(0, int(x1) - d) : min(ancho, int(x2) + d),
            ] = 0.0

    def _banda_de_busqueda(
        self, realzado: np.ndarray, detections: Sequence[Detection]
    ) -> tuple[int, int]:
        """Acota verticalmente la búsqueda entre el horizonte y la rasante."""
        alto = realzado.shape[0]

        # Limite superior: primera fila desde arriba con textura. El cielo tiene
        # varianza por fila muy baja y es el borde horizontal mas nitido de toda la
        # escena, asi que hay que excluirlo antes de buscar.
        varianza = realzado.std(axis=1)
        con_textura = np.flatnonzero(varianza > self._sky_threshold)
        y_min = int(con_textura[0]) if con_textura.size else 0

        # Limite inferior: el punto de contacto mas bajo de la maquinaria, que es
        # el de la maquina mas cercana. El pretil esta siempre por encima de el.
        # Es una cota deliberadamente conservadora: no excluye pretil real cuando
        # la unica maquina visible esta lejos.
        if detections:
            rasante = max(d.bbox.ground_point[1] for d in detections)
            y_max = int(min(alto, rasante + alto * self._band_margin))
        else:
            # Sin referencia se recorta el ultimo 20% del frame, donde la medicion
            # mostro que domina la textura del primer plano. Es grosero, y por eso
            # la confianza se penaliza cuando se recurre a esto.
            y_max = int(alto * 0.80)

        return y_min, max(y_min, y_max)

    def _camino_de_cresta(self, banda: np.ndarray) -> np.ndarray:
        """Camino de máxima respuesta acumulada, con salto vertical acotado.

        Programación dinámica clásica sobre la rejilla fila-columna. Hacia adelante
        se acumula, para cada fila de cada columna, el mejor puntaje alcanzable
        desde la columna anterior dentro de ``±max_step``; hacia atrás se recupera
        el camino óptimo.

        El paso hacia adelante usa un filtro de máximo deslizante en vez de un
        bucle sobre filas: la operación «mejor predecesor en una ventana» es
        exactamente eso, y expresarla así deja el coste en una pasada vectorizada
        por columna.

        Returns:
            Fila elegida por columna, relativa al inicio de la banda.
        """
        alto, ancho = banda.shape
        tam_ventana = 2 * self._max_step + 1

        acumulado = np.empty_like(banda)
        acumulado[:, 0] = banda[:, 0]
        for x in range(1, ancho):
            mejor_previo = maximum_filter1d(
                acumulado[:, x - 1], size=tam_ventana, mode="nearest"
            )
            acumulado[:, x] = banda[:, x] + mejor_previo

        camino = np.empty(ancho, dtype=np.int64)
        y = int(acumulado[:, -1].argmax())
        camino[-1] = y
        for x in range(ancho - 2, -1, -1):
            desde = max(0, y - self._max_step)
            hasta = min(alto, y + self._max_step + 1)
            y = desde + int(acumulado[desde:hasta, x].argmax())
            camino[x] = y
        return camino

    def _suavizar_perfil(self, cresta: FloatArray) -> FloatArray:
        """Suaviza el perfil por columna preservando sus variaciones reales."""
        indices = np.flatnonzero(np.isfinite(cresta))
        if indices.size < 7:
            return cresta

        largo = min(self._smooth_window | 1, (indices.size // 2) * 2 - 1)
        if largo < 5:
            return cresta

        salida = cresta.copy()
        salida[indices] = savgol_filter(cresta[indices], largo, 2, mode="nearest")
        return salida

    def _suavizar_temporal(self, cresta: FloatArray) -> FloatArray:
        """Suavizado exponencial contra el frame anterior.

        Es el mecanismo antiparpadeo del perfil. Las columnas con dato en sólo uno
        de los dos frames se toman del que lo tenga, en lugar de propagar ``NaN``:
        perder una columna por un frame ruidoso empeoraría la cobertura sin ganar
        precisión.
        """
        if self._crest_previa is None or self._crest_previa.shape != cresta.shape:
            self._crest_previa = cresta.copy()
            return cresta

        previa = self._crest_previa
        ambos = np.isfinite(cresta) & np.isfinite(previa)
        mezcla = np.where(
            ambos,
            self._alpha * cresta + (1.0 - self._alpha) * previa,
            np.where(np.isfinite(cresta), cresta, previa),
        ).astype(np.float32)

        self._crest_previa = mezcla
        return mezcla

    def _estimar_base(
        self, respuesta: np.ndarray, cresta: FloatArray, y_max: int
    ) -> FloatArray:
        """Localiza el pie del pretil descendiendo desde la cresta.

        La base es donde la cara del pretil se encuentra con la rasante: bajando
        desde la cresta, el punto en que la respuesta estructural vuelve a caer al
        nivel de fondo. Se busca sobre la respuesta y no sobre la imagen porque
        interesa dónde termina el escalón, no dónde cambia el brillo.

        Vectorizado sobre las columnas: la versión con un bucle en Python costaba
        unos 150 ms por frame en 1080p, el doble que la inferencia del detector, y
        un postproceso geométrico no puede ser el cuello de botella de un pipeline
        de video.
        """
        alto, ancho = respuesta.shape
        fin = min(alto, y_max + int(alto * 0.10))

        valida = np.isfinite(cresta)
        if not valida.any():
            return np.full_like(cresta, np.nan)

        filas = np.arange(alto, dtype=np.float32)[:, None]
        cresta_segura = np.where(valida, cresta, np.inf)

        # Nivel de fondo por columna. El percentil 25 y no el minimo: el minimo es
        # una muestra unica y por tanto ruidosa, mientras el cuartil describe el
        # nivel base de la columna.
        fondo = np.percentile(respuesta[:fin, :], 25, axis=0)
        indices = np.where(valida, np.nan_to_num(cresta, nan=0.0), 0).astype(int)
        pico = respuesta[np.clip(indices, 0, alto - 1), np.arange(ancho)]

        umbral = fondo + 0.5 * (pico - fondo)
        cae = (
            (respuesta < umbral[None, :])
            & (filas > cresta_segura[None, :])
            & (filas < fin)
        )

        primero = cae.argmax(axis=0).astype(np.float32)
        hay_cruce = cae.any(axis=0) & valida & (pico > fondo)
        return np.where(hay_cruce, primero, np.nan).astype(np.float32)

    def _confianza(
        self,
        cobertura: float,
        cresta: FloatArray,
        detections: Sequence[Detection],
    ) -> float:
        """Confianza propia del segmentador, en ``[0, 1]``.

        Combina tres señales. Ninguna es un ajuste aprendido: son penalizaciones
        declaradas sobre condiciones que sabemos que degradan el método.
        """
        # La cobertura domina: un perfil sobre el 20% de las columnas no describe
        # el pretil, por buenas que sean esas columnas.
        confianza = cobertura

        # Sin maquinaria visible la banda de busqueda es un recorte grosero, y la
        # medicion mostro que ahi el metodo se degrada de forma sustancial.
        if not detections:
            confianza *= 0.5

        # Un perfil que serpentea mucho mas de lo que serpentea un pretil real es
        # probablemente un enganche a textura del terreno.
        finitos = cresta[np.isfinite(cresta)]
        if finitos.size >= 3:
            rugosidad = float(np.std(np.diff(finitos)))
            confianza *= float(np.clip(1.0 - rugosidad / 8.0, 0.1, 1.0))

        return float(np.clip(confianza, 0.0, 1.0))
