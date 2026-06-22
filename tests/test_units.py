"""Unit test: de-vig quote, blend mercato, normalizzazione nomi."""
import numpy as np

from previsore import odds
from previsore.blend import market_blend
from previsore.scorers import _norm_name


def test_shin_devig_sums_to_one_and_removes_margin():
    o = [2.10, 3.40, 3.50]
    p = odds.shin_devig(o)
    assert abs(sum(p) - 1.0) < 1e-9
    raw = np.array([1 / x for x in o])
    assert raw.sum() > 1.0                      # c'e overround
    assert all(0 < x < 1 for x in p)


def test_shin_devig_bad_input():
    assert odds.shin_devig([1.0, 2.0, 3.0]) is None      # quota <= 1
    assert odds.shin_devig([2.0, 3.0]) is None           # non 3 esiti


def test_market_blend_fallback():
    pm = [0.5, 0.3, 0.2]
    np.testing.assert_allclose(market_blend(pm, None, 0.5), pm)   # niente mercato -> modello
    np.testing.assert_allclose(market_blend(pm, [0.2, 0.3, 0.5], 0.0), pm)  # peso 0
    out = market_blend([0.6, 0.2, 0.2], [0.2, 0.2, 0.6], 0.5)
    assert abs(out.sum() - 1.0) < 1e-9 and out[0] < 0.6          # tirato verso il mercato


def test_norm_name_accent_dedup():
    assert _norm_name("Julián Álvarez") == _norm_name("Julian Alvarez")
    assert _norm_name("Kylian Mbappé") == _norm_name("kylian mbappe")
