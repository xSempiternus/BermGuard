"""Tests de cómo prepare_v2 lee la decisión de LOVO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_v2 import EPOCAS, decision  # noqa: E402


def resumen(tmp_path: Path, **decision_: object) -> Path:
    ruta = tmp_path / "resumen.json"
    ruta.write_text(json.dumps({"decision": decision_}), encoding="utf-8")
    return ruta


def test_usa_la_ganadora_y_las_epocas_del_protocolo(tmp_path):
    assert decision(resumen(tmp_path, ganadora="pretrain")) == ("pretrain", EPOCAS)


def test_epocas_dobles_si_base2x_mejoro_a_base(tmp_path):
    ruta = resumen(tmp_path, ganadora="base", epocas_dobles=True)
    assert decision(ruta) == ("base", 2 * EPOCAS)


def test_una_decision_provisoria_no_se_usa(tmp_path):
    with pytest.raises(SystemExit, match="provisoria"):
        decision(resumen(tmp_path, ganadora="pretrain", pendiente_control=True))
