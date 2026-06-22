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


def test_squad_alias_canonical_key():
    # heading Wikipedia 'Czechia' deve essere raggiungibile col nome martj42 'Czech Republic'
    import pandas as pd
    from previsore import squads
    df = pd.DataFrame({"team": ["Czechia", "Czechia"], "player": ["Tomáš Souček", "Patrik Schick"]})
    tmap = squads.tokens_by_team(df)
    assert squads.squad_tokens_for("Czech Republic", tmap)            # non None / non vuoto
    assert "soucek" in squads.squad_tokens_for("Czech Republic", tmap)


def test_lookup_market_neutral_orientation_and_bad_csv():
    import pandas as pd
    # riga col fixture INVERTITO (campo neutro): le prob devono essere scambiate
    t = pd.DataFrame([{"home_team": "Iran", "away_team": "Spain",
                       "odds_home": 4.0, "odds_draw": 3.5, "odds_away": 1.9}])
    p = odds.lookup_market(t, "Spain", "Iran")
    assert p is not None and p[0] > p[2]                              # Spain favorita
    # CSV senza le colonne attese -> None, niente crash
    assert odds.lookup_market(pd.DataFrame({"foo": [1]}), "Spain", "Iran") is None


def test_svg_escapes_hostile_names():
    import xml.dom.minidom as M
    from previsore.render import render_card_svg
    pred = {"home": "A & B", "away": "<X>", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
            "exact": (1, 1), "exact_prob": 0.11, "lambda_home": 1.4, "lambda_away": 1.1, "neutral": True}
    svg = render_card_svg(pred, [("Smith & Jones", 0.3, True)], [], {"tournament": "A<B>"})
    M.parseString(svg)                                                # solleva se malformato
