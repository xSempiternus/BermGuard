"""Tests del resumen LOVO con JSON sintéticos: agregación, umbral sin fuga, ruido, decisión."""

from __future__ import annotations

import copy
import json
import math

import pytest

from bermguard.benchmark.lovo import (
    CriterioUmbral,
    ReglaDecision,
    a_markdown,
    agregar_barridos,
    agregar_variante,
    cargar,
    comparar_contra_base,
    diferencia_pareada,
    elegir_umbral,
    elegir_variante,
    limpiar_nan,
    p_signo,
    resumir_lovo,
    umbral_fuera_de_fold,
)

VIDEOS = ("video_01", "video_02", "video_03", "video_04")


def fila(conf, tp, fp, n):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / n
    f1 = 2 * p * r / (p + r) if p == p and p + r > 0 else 0.0
    return {"conf": conf, "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "n_truth": n}


def barrido_simple(n=10):
    return [fila(0.05, 8, 2, n), fila(0.10, 6, 1, n), fila(0.15, 3, 0, n)]


def reporte(ap_bull=0.5, ap_caex=0.7, barrido_bull=None, n_bull=10, n_caex=30, p_bull=0.5):
    return {
        "pesos": "sintetico.pt",
        "imagenes": 40,
        "bloques": 8,
        "umbral_operacion": 0.25,
        "umbral_por_clase": {},
        "nms_iou": 0.45,
        "por_clase": {
            "caex": {
                "ap50": ap_caex,
                "ap50_95": ap_caex / 2,
                "precision": 0.6,
                "recall": 0.8,
                "f1": 0.686,
                "n_truth": float(n_caex),
            },
            "bulldozer": {
                "ap50": ap_bull,
                "ap50_95": ap_bull / 2,
                "precision": p_bull,
                "recall": 0.3,
                "f1": 0.375,
                "n_truth": float(n_bull),
            },
        },
        "barrido": {
            "bulldozer": barrido_bull if barrido_bull is not None else barrido_simple(n_bull),
            "caex": barrido_simple(n_caex),
        },
        "por_luz": {},
        "particion": {"train": 120, "val": 40, "pegados": 0},
    }


def variante(aps_bull, aps_caex=(0.7, 0.7, 0.7, 0.7)):
    return {v: reporte(b, c) for v, b, c in zip(VIDEOS, aps_bull, aps_caex, strict=True)}


# --- Agregación --------------------------------------------------------------


def test_media_desviacion_y_detalle_por_fold():
    a = agregar_variante(variante([0.2, 0.4, 0.6, 0.8]))
    ap = a["por_clase"]["bulldozer"]["ap50"]
    assert ap["media"] == pytest.approx(0.5)
    assert ap["desviacion"] == pytest.approx(0.258199, abs=1e-6)  # ddof=1
    assert ap["n_folds"] == 4
    assert ap["por_fold"]["video_03"] == pytest.approx(0.6)
    assert a["por_clase"]["bulldozer"]["n_objetos"]["video_01"] == 10
    assert a["por_clase"]["bulldozer"]["ap50_95"]["media"] == pytest.approx(0.25)
    assert list(a["por_clase"])[0] == "bulldozer"


def test_metrica_indefinida_queda_fuera_de_la_media():
    folds = variante([0.5] * 4)
    folds["video_02"]["por_clase"]["bulldozer"]["precision"] = float("nan")
    folds["video_03"]["por_clase"]["bulldozer"]["precision"] = None  # JSON ya limpiado
    p = agregar_variante(folds)["por_clase"]["bulldozer"]["precision"]
    assert p["n_folds"] == 2
    assert p["media"] == pytest.approx(0.5)


def test_metrica_ausente_en_json_viejo_es_nan():
    folds = variante([0.5] * 4)
    for r in folds.values():
        del r["por_clase"]["bulldozer"]["ap50_95"]
    m = agregar_variante(folds)["por_clase"]["bulldozer"]["ap50_95"]
    assert m["n_folds"] == 0
    assert math.isnan(m["media"])


# --- Diferencias pareadas y ruido --------------------------------------------


def test_diferencia_pareada_por_fold():
    base = variante([0.2, 0.4, 0.6, 0.8])
    cp = variante([0.3, 0.5, 0.65, 0.75])
    d = diferencia_pareada(base, cp, "bulldozer", "ap50", ReglaDecision())
    assert d["por_fold"]["video_01"] == pytest.approx(0.1)
    assert d["minimo"] == pytest.approx(-0.05)
    assert d["maximo"] == pytest.approx(0.1)
    assert d["media"] == pytest.approx(0.05)
    assert (d["folds_mejora"], d["folds_empeora"], d["folds_empate"]) == (3, 1, 0)
    assert d["veredicto"] == "mejora"


def test_signos_mezclados_es_ruido():
    base = variante([0.5] * 4)
    otra = variante([0.6, 0.42, 0.55, 0.48])
    d = diferencia_pareada(base, otra, "bulldozer", "ap50", ReglaDecision())
    assert d["folds_mejora"] == 2
    assert d["veredicto"] == "no distinguible del ruido"


def test_mejora_consistente_pero_minima_es_ruido():
    # 4 de 4 folds mejoran, pero 0.015 de media no llega al efecto mínimo de 0.02.
    base = variante([0.5] * 4)
    otra = variante([0.515] * 4)
    d = diferencia_pareada(base, otra, "bulldozer", "ap50", ReglaDecision())
    assert d["folds_mejora"] == 4
    assert d["veredicto"] == "no distinguible del ruido"


def test_diferencias_bajo_el_empate_no_cuentan():
    base = variante([0.5] * 4)
    otra = variante([0.505, 0.495, 0.5, 0.509])
    d = diferencia_pareada(base, otra, "bulldozer", "ap50", ReglaDecision(empate=0.01))
    assert d["folds_empate"] == 4
    assert d["p_signo"] == 1.0


def test_empeora_simetrico():
    base = variante([0.5] * 4)
    otra = variante([0.4, 0.45, 0.42, 0.55])
    d = diferencia_pareada(base, otra, "bulldozer", "ap50", ReglaDecision())
    assert d["veredicto"] == "empeora"


def test_prueba_de_signos_con_cuatro_folds_no_llega_al_5_por_ciento():
    assert p_signo(4, 0) == pytest.approx(0.125)
    assert p_signo(3, 1) == pytest.approx(0.625)
    assert p_signo(0, 0) == 1.0


# --- Umbral -------------------------------------------------------------------


def test_criterios_de_umbral():
    filas = [fila(0.05, 8, 12, 10), fila(0.10, 5, 3, 10), fila(0.15, 2, 0, 10)]
    assert elegir_umbral(filas, CriterioUmbral("precision_min", 0.5))["conf"] == 0.10
    assert elegir_umbral(filas, CriterioUmbral("recall_min", 0.5))["conf"] == 0.10
    assert elegir_umbral(filas, CriterioUmbral("recall_min", 0.1))["conf"] == 0.15
    assert elegir_umbral(filas, CriterioUmbral("max_f1"))["conf"] == 0.10


def test_si_ningun_umbral_cumple_cae_a_max_f1_y_lo_marca():
    filas = [fila(0.05, 8, 12, 10), fila(0.10, 5, 30, 10)]
    e = elegir_umbral(filas, CriterioUmbral("precision_min", 0.9))
    assert e["cumple"] is False
    assert e["conf"] == 0.05


def test_el_barrido_conjunto_suma_conteos():
    # Un fold de 2 objetos con precisión perfecta no pesa lo mismo que uno de 100.
    chico = {"barrido": {"bulldozer": [fila(0.1, 2, 0, 2)]}}
    grande = {"barrido": {"bulldozer": [fila(0.1, 10, 90, 100)]}}
    (f,) = agregar_barridos([chico, grande], "bulldozer")
    assert f["agregacion"] == "conteos"
    assert f["precision"] == pytest.approx(12 / 102)
    assert f["recall"] == pytest.approx(12 / 102)


def test_barridos_sin_conteos_se_promedian_en_los_umbrales_comunes():
    viejo = [{"conf": 0.05, "precision": 0.4, "recall": 0.6, "f1": 0.48}]
    otro = [
        {"conf": 0.05, "precision": 0.8, "recall": 0.2, "f1": 0.32},
        {"conf": 0.10, "precision": 1.0, "recall": 0.1, "f1": 0.18},
    ]
    filas = agregar_barridos(
        [{"barrido": {"bulldozer": viejo}}, {"barrido": {"bulldozer": otro}}], "bulldozer"
    )
    assert [f["conf"] for f in filas] == [0.05]
    assert filas[0]["agregacion"] == "promedio"
    assert filas[0]["precision"] == pytest.approx(0.6)


def folds_para_fuga():
    """Los folds 1-3 eligen 0.10; el fold 4, mucho más grande, elegiría 0.05 por sí solo."""
    otros = [fila(0.05, 8, 12, 10), fila(0.10, 5, 3, 10), fila(0.15, 2, 0, 10)]
    propio = [fila(0.05, 90, 10, 100), fila(0.10, 50, 0, 100), fila(0.15, 20, 0, 100)]
    folds = {v: reporte(barrido_bull=copy.deepcopy(otros)) for v in VIDEOS[:3]}
    folds["video_04"] = reporte(barrido_bull=propio, n_bull=100)
    return folds


def test_el_fold_k_no_usa_su_propio_barrido():
    folds = folds_para_fuga()
    criterio = CriterioUmbral("precision_min", 0.5)
    u = umbral_fuera_de_fold(folds, "bulldozer", criterio)
    k = u["por_fold"]["video_04"]
    assert "video_04" not in k["elegido_con"]
    assert k["umbral"] == pytest.approx(0.10)
    # Medido en el fold 4 con ese umbral: su propia fila de 0.10.
    assert k["recall"] == pytest.approx(0.5)
    assert k["precision"] == pytest.approx(1.0)
    # Con los cuatro folds (producción) el fold 4 sí entra, y el umbral cambia.
    assert u["produccion"]["umbral"] == pytest.approx(0.05)


def test_cambiar_el_barrido_de_k_no_cambia_el_umbral_de_k():
    folds = folds_para_fuga()
    criterio = CriterioUmbral("precision_min", 0.5)
    antes = umbral_fuera_de_fold(folds, "bulldozer", criterio)["por_fold"]
    for k in VIDEOS:
        alterado = copy.deepcopy(folds)
        alterado[k]["barrido"]["bulldozer"] = [
            fila(c, 0, 50, 10) if c != 0.15 else fila(c, 10, 0, 10) for c in (0.05, 0.10, 0.15)
        ]
        despues = umbral_fuera_de_fold(alterado, "bulldozer", criterio)["por_fold"]
        assert despues[k]["umbral"] == antes[k]["umbral"]
        assert despues[k]["elegido_con"] == [v for v in VIDEOS if v != k]


def test_conjunto_fuera_de_fold_suma_cada_fold_en_su_umbral():
    u = umbral_fuera_de_fold(folds_para_fuga(), "bulldozer", CriterioUmbral())
    tp = sum(
        next(f["tp"] for f in r["barrido"]["bulldozer"] if f["conf"] == u["por_fold"][v]["umbral"])
        for v, r in folds_para_fuga().items()
    )
    assert u["conjunto_fuera_de_fold"]["tp"] == tp
    assert u["conjunto_fuera_de_fold"]["n_objetos"] == 130


# --- Decisión -----------------------------------------------------------------


def decidir(datos, regla=None):
    regla = regla or ReglaDecision()
    return elegir_variante(datos, comparar_contra_base(datos, regla), regla)


def test_gana_la_que_mejora_el_bulldozer_sin_romper_caex():
    datos = {
        "base": variante([0.4, 0.5, 0.3, 0.6]),
        "copypaste": variante([0.5, 0.6, 0.35, 0.65]),
        # Mejora más el bulldozer, pero el CAEX cae 0.1 en un fold.
        "pretrain": variante([0.6, 0.7, 0.5, 0.8], aps_caex=(0.7, 0.6, 0.7, 0.7)),
        # El doble de épocas no le da nada a base: la mejora de copypaste no es de pasos.
        "base2x": variante([0.4, 0.5, 0.3, 0.6]),
    }
    d = decidir(datos)
    assert d["ganadora"] == "copypaste"
    assert d["evaluacion"]["pretrain"]["candidata"] is False
    assert any("caex" in m for m in d["evaluacion"]["pretrain"]["motivos"])


def test_si_ninguna_mejora_de_forma_consistente_gana_base():
    datos = {
        "base": variante([0.4, 0.5, 0.3, 0.6]),
        "copypaste": variante([0.6, 0.4, 0.35, 0.55]),  # 2 de 4
    }
    d = decidir(datos)
    assert d["ganadora"] == "base"
    assert "parsimonia" in d["motivo"]


def test_empate_entre_candidatas_gana_la_mas_simple():
    datos = {
        "base": variante([0.4] * 4),
        "pretrain": variante([0.505] * 4),
        "copypaste": variante([0.5] * 4),
        "base2x": variante([0.4] * 4),
    }
    assert decidir(datos)["ganadora"] == "copypaste"
    datos["pretrain"] = variante([0.6] * 4)
    assert decidir(datos)["ganadora"] == "pretrain"


def test_variante_con_folds_incompletos_no_es_candidata():
    datos = {"base": variante([0.4] * 4), "copypaste": variante([0.8] * 4)}
    del datos["copypaste"]["video_04"]
    d = decidir(datos)
    assert d["ganadora"] == "base"
    assert any("incompletos" in m for m in d["evaluacion"]["copypaste"]["motivos"])


def test_copypaste_sin_control_queda_pendiente():
    datos = {"base": variante([0.4] * 4), "copypaste": variante([0.6] * 4)}
    d = decidir(datos)
    assert d["pendiente_control"] is True
    assert d["ganadora"] == "base"
    assert any("base2x" in m for m in d["evaluacion"]["copypaste"]["motivos"])


def test_copypaste_que_no_supera_al_control_de_pasos_no_gana():
    datos = {
        "base": variante([0.4] * 4),
        "copypaste": variante([0.6] * 4),
        # Con los mismos pasos, base alcanza a copypaste: la mejora era de cómputo.
        "base2x": variante([0.6] * 4),
    }
    d = decidir(datos)
    assert d["pendiente_control"] is False
    assert d["ganadora"] == "base"
    assert any("pasos" in m for m in d["evaluacion"]["copypaste"]["motivos"])
    assert d["evaluacion"]["base2x"]["candidata"] is False


def test_si_base2x_mejora_a_base_se_recomiendan_epocas_dobles():
    datos = {
        "base": variante([0.4] * 4),
        "copypaste": variante([0.6] * 4),
        "base2x": variante([0.6] * 4),
    }
    d = decidir(datos)
    assert d["ganadora"] == "base"
    assert d["epocas_dobles"] is True
    assert "doble de épocas" in d["motivo"]
    assert d["evaluacion"]["copypaste"]["contra_control"]["veredicto"] != "mejora"
    assert (
        decidir({"base": variante([0.4] * 4), "base2x": variante([0.4] * 4)})["epocas_dobles"]
        is False
    )


def test_el_markdown_muestra_copypaste_contra_el_control():
    datos = {
        "base": variante([0.4] * 4),
        "copypaste": variante([0.6] * 4),
        "base2x": variante([0.45] * 4),
    }
    md = a_markdown(resumir_lovo(datos))
    assert "Variante elegida: `copypaste`" in md
    assert "contra `base2x` (mismos pasos): media +0.150" in md
    assert "`base2x`: control de cómputo de copypaste, no compite." in md


def test_el_control_no_afecta_a_pretrain():
    datos = {
        "base": variante([0.4] * 4),
        "pretrain": variante([0.6] * 4),
        "base2x": variante([0.7] * 4),
    }
    d = decidir(datos)
    assert d["ganadora"] == "pretrain"
    assert d["pendiente_control"] is False


# --- Resumen, JSON y markdown ------------------------------------------------


def test_resumen_de_punta_a_punta_desde_archivos(tmp_path):
    datos = {
        "base": variante([0.4, 0.5, 0.3, 0.6]),
        "copypaste": variante([0.5, 0.6, 0.35, 0.65]),
        "base2x": variante([0.4, 0.5, 0.3, 0.6]),
    }
    datos["base"]["video_02"]["por_clase"]["bulldozer"]["precision"] = float("nan")
    for v, folds in datos.items():
        for video, r in folds.items():
            (tmp_path / f"{v}_{video}.json").write_text(json.dumps(r), encoding="utf-8")
    (tmp_path / "resumen.json").write_text("{}", encoding="utf-8")  # no es un fold

    leidos = cargar(tmp_path)
    assert set(leidos) == {"base", "copypaste", "base2x"}  # el nombre con dígito se lee
    resumen = resumir_lovo(leidos)
    assert resumen["decision"]["ganadora"] == "copypaste"
    assert resumen["umbral_produccion"]["variante"] == "copypaste"
    json.dumps(limpiar_nan(resumen), allow_nan=False)  # JSON estándar, sin NaN

    md = a_markdown(resumen)
    assert "nan" not in md.lower()
    assert "Variante elegida: `copypaste`" in md
    assert "0.450 ± 0.129" in md  # AP@0.5 bulldozer de base: media y desviación
    assert "+0.075" in md  # media de la diferencia pareada
    # El markdown se puede regenerar desde el JSON ya escrito.
    assert a_markdown(json.loads(json.dumps(limpiar_nan(resumen)))) == md


def test_umbral_en_el_borde_de_la_grilla_se_marca():
    filas = [fila(0.01, 9, 1, 10), fila(0.05, 5, 0, 10)]
    assert elegir_umbral(filas, CriterioUmbral("precision_min", 0.5))["en_borde_de_grilla"]
    assert not elegir_umbral(filas, CriterioUmbral("recall_min", 0.5))["en_borde_de_grilla"]
