"""Tests del tracker.

Se ejecutan sin GPU, sin modelo y sin archivo de video: el tracker opera sobre
``Detection``, que son dataclasses puras. Esa testeabilidad es la consecuencia
práctica de que el orquestador dependa de `Protocol` y no de implementaciones.

Cada test comprueba un comportamiento que el pipeline necesita, no un detalle de
implementación.
"""

from __future__ import annotations

import pytest

from bermguard.core.types import BBox, Detection, VehicleClass
from bermguard.vision.tracking import IouTracker, iou


def deteccion(x: float, y: float, ancho: float = 100, alto: float = 80, conf: float = 0.9):
    """Detección sin track asignado, en la posición dada."""
    return Detection(
        track_id=-1,
        vehicle_class=VehicleClass.CAEX,
        bbox=BBox(x, y, x + ancho, y + alto),
        confidence=conf,
    )


class TestIou:
    def test_cajas_identicas(self):
        caja = BBox(0, 0, 10, 10)
        assert iou(caja, caja) == pytest.approx(1.0)

    def test_sin_solape(self):
        assert iou(BBox(0, 0, 10, 10), BBox(20, 20, 30, 30)) == 0.0

    def test_solape_parcial(self):
        # Dos cuadrados de 10x10 desplazados 5 en ambos ejes: interseccion 25,
        # union 175.
        assert iou(BBox(0, 0, 10, 10), BBox(5, 5, 15, 15)) == pytest.approx(25 / 175)

    def test_contacto_sin_area(self):
        """Cajas que se tocan por el borde no se solapan."""
        assert iou(BBox(0, 0, 10, 10), BBox(10, 0, 20, 10)) == 0.0


class TestConfirmacion:
    """El filtro de falsos positivos, que es la razón de ser del ciclo de vida."""

    def test_no_reporta_antes_de_min_hits(self):
        tracker = IouTracker(min_hits=3)
        assert tracker.update([deteccion(0, 0)], 0) == []
        assert tracker.update([deteccion(0, 0)], 1) == []

    def test_reporta_al_alcanzar_min_hits(self):
        tracker = IouTracker(min_hits=3)
        for i in range(2):
            tracker.update([deteccion(0, 0)], i)
        salida = tracker.update([deteccion(0, 0)], 2)
        assert len(salida) == 1
        assert salida[0].track_id == 0

    def test_falso_positivo_de_un_frame_nunca_se_reporta(self):
        """El caso que motiva el diseño.

        El detector tiene precisión 0.297: emite falsos positivos sobre polvo y
        terreno. Un falso positivo aislado no reaparece en el frame siguiente, de
        modo que no llega a confirmarse y nunca alcanza el OSD ni la analítica de
        proximidad.
        """
        tracker = IouTracker(min_hits=3)
        assert tracker.update([deteccion(500, 500)], 0) == []
        # El frame siguiente no lo contiene: el track queda sin confirmar.
        for i in range(1, 5):
            assert tracker.update([], i) == []

    def test_identidad_estable_mientras_la_maquina_se_mueve(self):
        tracker = IouTracker(min_hits=2)
        ids = set()
        for i in range(10):
            # Desplazamiento de 8 px por frame: solape amplio entre frames vecinos.
            salida = tracker.update([deteccion(8 * i, 0)], i)
            ids.update(d.track_id for d in salida)
        assert ids == {0}, "la maquina deberia conservar un solo identificador"


class TestOclusiones:
    def test_el_track_sobrevive_un_hueco(self):
        """Sin esto la caja parpadea cada vez que el detector pierde un frame."""
        tracker = IouTracker(min_hits=2, max_age=5)
        for i in range(3):
            tracker.update([deteccion(8 * i, 0)], i)

        # Dos frames sin deteccion: el track se arrastra y sigue reportandose.
        for i in (3, 4):
            salida = tracker.update([], i)
            assert len(salida) == 1, "el track deberia arrastrarse durante la oclusion"

    def test_el_track_muere_pasado_max_age(self):
        tracker = IouTracker(min_hits=2, max_age=3)
        for i in range(3):
            tracker.update([deteccion(0, 0)], i)
        for i in range(3, 12):
            tracker.update([], i)
        assert tracker.update([], 12) == [], "un equipo que salio de escena no debe persistir"

    def test_arrastre_extrapola_el_movimiento(self):
        """La caja arrastrada avanza según la velocidad estimada, no se congela."""
        tracker = IouTracker(min_hits=2, max_age=5, velocity_smoothing=1.0)
        for i in range(4):
            tracker.update([deteccion(10 * i, 0)], i)
        ultima_vista = tracker.update([deteccion(40, 0)], 4)[0].bbox.x1

        arrastrada = tracker.update([], 5)[0].bbox.x1
        assert arrastrada > ultima_vista, (
            "la prediccion deberia avanzar en el sentido del movimiento"
        )


class TestSegundaRonda:
    def test_una_deteccion_debil_continua_un_track_existente(self):
        """La idea distintiva de ByteTrack.

        Con polvo o de noche un CAEX real cae a confianza baja. Un tracker que sólo
        mirase detecciones fuertes lo perdería; acá hay un track que lo explica.
        """
        tracker = IouTracker(min_hits=2, high_confidence=0.45)
        for i in range(2):
            tracker.update([deteccion(0, 0, conf=0.9)], i)

        salida = tracker.update([deteccion(4, 0, conf=0.2)], 2)
        assert len(salida) == 1
        assert salida[0].track_id == 0, "el track deberia continuar, no nacer de nuevo"

    def test_una_deteccion_debil_no_crea_un_track(self):
        tracker = IouTracker(min_hits=1, high_confidence=0.45)
        assert tracker.update([deteccion(0, 0, conf=0.2)], 0) == []


class TestReset:
    def test_reset_descarta_los_tracks(self):
        """El orquestador llama a reset en cada corte de toma (ADR 0003)."""
        tracker = IouTracker(min_hits=2)
        for i in range(3):
            tracker.update([deteccion(0, 0)], i)

        tracker.reset()
        assert tracker.update([deteccion(0, 0)], 3) == [], "tras un corte hay que reconfirmar"

    def test_reset_no_reutiliza_identificadores(self):
        """Dos equipos de tomas distintas no deben compartir número.

        Si los identificadores se reiniciaran, los artefactos mezclarían las
        trayectorias de dos máquinas distintas bajo el mismo track_id.
        """
        tracker = IouTracker(min_hits=1)
        primero = tracker.update([deteccion(0, 0)], 0)[0].track_id

        tracker.reset()
        segundo = tracker.update([deteccion(0, 0)], 1)[0].track_id
        assert segundo != primero


class TestVariasMaquinas:
    def test_dos_maquinas_reciben_identidades_distintas(self):
        tracker = IouTracker(min_hits=2)
        for i in range(3):
            salida = tracker.update([deteccion(0, 0), deteccion(400, 0)], i)
        assert len({d.track_id for d in salida}) == 2

    def test_maquinas_que_se_cruzan_conservan_su_identidad(self):
        """El motivo de usar asignación óptima y no emparejamiento greedy.

        Dos equipos que convergen: greedy podría asignar la primera pareja que
        encuentra y forzar a la segunda a un emparejamiento malo. El algoritmo
        húngaro minimiza el costo total.
        """
        tracker = IouTracker(min_hits=2)
        for i in range(3):
            tracker.update([deteccion(0, 0), deteccion(300, 0)], i)

        # Se acercan sin llegar a solaparse.
        salida = tracker.update([deteccion(20, 0), deteccion(280, 0)], 3)
        por_posicion = sorted(salida, key=lambda d: d.bbox.x1)
        assert por_posicion[0].track_id == 0
        assert por_posicion[1].track_id == 1
