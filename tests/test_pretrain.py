"""Tests del remapeo de datasets públicos a las clases del proyecto."""

from __future__ import annotations

import pytest

from bermguard.training.pretrain import (
    contiene,
    descartar_degeneradas,
    leer_exclusiones,
    nombres_repetidos,
    parse_class_map,
    parse_repeat,
    remap_labels,
)

NOMBRES = ["Bull_dozer", "excavator", "tippertruck"]


def test_variantes_de_nombre_son_la_misma_clase():
    mapa = parse_class_map(["bull dozer=bulldozer", "Tipper-Truck=caex"])
    assert mapa == {"bulldozer": 1, "tippertruck": 0}


def test_destino_desconocido_falla():
    with pytest.raises(ValueError):
        parse_class_map(["dozer=excavadora"])


def test_reasigna_indices_y_descarta_clases_sin_traduccion():
    mapa = parse_class_map(["bull_dozer=bulldozer", "tippertruck=caex"])
    texto = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n2 0.1 0.1 0.3 0.1 0.3 0.3\n"
    assert remap_labels(texto, NOMBRES, mapa) == "1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.3 0.1 0.3 0.3\n"


def test_imagen_sin_clases_traducidas_queda_vacia():
    mapa = parse_class_map(["tippertruck=caex"])
    assert remap_labels("1 0.5 0.5 0.2 0.2\n9 0.1 0.1 0.1 0.1\n", NOMBRES, mapa) == ""


def test_filtro_por_clase_requerida():
    assert contiene("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n", "bulldozer")
    assert not contiene("0 0.5 0.5 0.2 0.2\n", "bulldozer")
    assert not contiene("", "caex")


def test_quita_poligonos_y_cajas_degenerados():
    texto = (
        "1 0.39 0.44 0.392 0.445 0.391 0.446\n"  # polígono de un clic de más
        "1 0.1 0.15 0.87 0.15 0.87 0.99\n"
        "0 0.5 0.5 0.001 0.2\n"  # caja sin ancho
        "0 0.5 0.5 0.1 0.2\n"
    )
    limpio, quitadas = descartar_degeneradas(texto)
    assert quitadas == 2
    assert limpio == "1 0.1 0.15 0.87 0.15 0.87 0.99\n0 0.5 0.5 0.1 0.2\n"
    assert descartar_degeneradas("1 0.5 0.5 0.001 0.001\n") == ("", 1)


def test_lista_de_exclusion_ignora_comentarios_y_vacias():
    texto = "# cabecera\n\na.jpg  # duplicado de b.jpg\n  c.jpg\n#d.jpg\n"
    assert leer_exclusiones(texto) == {"a.jpg", "c.jpg"}


def test_repeticion_por_fuente():
    assert parse_repeat(["mines=5", "bulldozer=1"]) == {"mines": 5, "bulldozer": 1}
    for malo in ["mines", "mines=0", "mines=x", "=3"]:
        with pytest.raises(ValueError):
            parse_repeat([malo])


def test_copias_con_sufijo():
    assert nombres_repetidos("mines__IMG_1", 3) == [
        "mines__IMG_1",
        "mines__IMG_1_r1",
        "mines__IMG_1_r2",
    ]
    assert nombres_repetidos("mines__IMG_1", 1) == ["mines__IMG_1"]
