"""Output 'minimal-swiss': monocromo, due inchiostri, cifre tabellari.

Niente dipendenze (ANSI SGR grezzo + f-string). Rispetta NO_COLOR e i pipe
(isatty). Inchiostri: PRIMARY = default fg, MUTED = faint. Un solo separatore
(·), trattino lungo (–) per i punteggi, niente emoji. Esporta anche una card SVG.
"""
from __future__ import annotations

import html
import os
import sys

INSET = "  "
W = 60                 # misura contenuto
LBL = 16               # colonna etichette
VAL = W - LBL          # 44
FIG = " "         # figure space (larghezza cifra)

def _isatty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:                # stdout sostituito da un writer minimale
        return False


PLAIN = (not _isatty()) or bool(os.environ.get("NO_COLOR"))


def _sgr(s, code):
    return s if PLAIN else f"\x1b[{code}m{s}\x1b[0m"


def ink(s):
    return _sgr(s, "22")


def faint(s):
    return _sgr(s, "2")


def _fig(s, width):
    return str(s).rjust(width, FIG)


def _hr():
    return INSET + faint("─" * W)


def _row(label, value):
    return INSET + faint(label.ljust(LBL)) + value


def _flush(left_render, left_plain, right_render, right_plain, width=W):
    """Allinea a destra usando la larghezza VISIBILE (ignora i codici SGR)."""
    gap = max(1, width - len(left_plain) - len(right_plain))
    return INSET + left_render + (" " * gap) + right_render


def _three(a, b, c, fmt=ink):
    ws = [15, 14, 15]
    return "".join(fmt(s.center(w)) if fmt else s.center(w) for s, w in zip((a, b, c), ws))


def _meter(ph, pd_, pa, width=VAL):
    nh = round(width * ph)
    nd = round(width * pd_)
    na = max(0, width - nh - nd)
    return ink("█" * nh) + faint("▒" * nd) + faint("░" * na)


def _scorer_cols(home_team, sh, away_team, sa):
    left_w = 34
    lines = [INSET + faint(f"scorers · {home_team.lower()}".ljust(left_w))
             + faint(f"scorers · {away_team.lower()}")]
    for i in range(max(len(sh), len(sa))):
        left = _scorer_cell(sh[i]) if i < len(sh) else ""
        right = _scorer_cell(sa[i]) if i < len(sa) else ""
        lines.append(INSET + "  " + left.ljust(left_w - 2) + right)
    return lines


def _scorer_cell(item):
    name, p, is_pk = item
    tag = faint(" (p)") if is_pk else "    "
    pct = _fig(f"{round(p * 100)}%", 4)
    return f"{ink(name[:16].ljust(16))}{tag} {ink(pct)}"


def render_terminal(pred: dict, scorers_home=None, scorers_away=None, meta=None) -> str:
    meta = meta or {}
    home, away = pred["home"], pred["away"]
    ph, pd_, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
    x, y = pred["exact"]
    has_mkt = pred.get("p_market") is not None
    ctx = "neutral · group stage" if pred["neutral"] else "group stage"
    badge = "blend + market" if has_mkt else "blend"

    L = [_hr(), ""]
    # header
    title = f"{home.upper()}  ·  {away.upper()}"
    L.append(_flush(ink(title), title, faint(ctx), ctx))
    date = meta.get("date", "")
    sub = f"{meta.get('tournament', 'FIFA World Cup 2026')}" + (f" · {date}" if date else "")
    L.append(_flush(faint(sub), sub, faint(badge), badge))
    L += ["", _hr(), ""]
    # 1X2
    L.append(_row("the line", _three("1", "X", "2", fmt=faint)))
    L.append(_row("", _three(f"{ph * 100:.1f}%", f"{pd_ * 100:.1f}%", f"{pa * 100:.1f}%")))
    L.append(_row("", _meter(ph, pd_, pa)))
    L.append("")
    # gol attesi + scoreline
    L.append(_row("expected goals", ink(f"{pred['lambda_home']:.2f}") + faint("  ·  ") + ink(f"{pred['lambda_away']:.2f}")))
    sline_plain = f"{x}–{y}"
    ep = pred["exact_prob"]
    one_in = round(1 / ep) if ep > 0 else 0
    modal_plain = f"~{ep * 100:.0f}% · 1 su {one_in}"
    modal_render = faint("~") + ink(f"{ep * 100:.0f}%") + faint(f" · 1 su {one_in}")
    lbl = "likely score".ljust(LBL)
    L.append(_flush(faint(lbl) + ink(sline_plain), lbl + sline_plain, modal_render, modal_plain))
    L.append("")
    # also (top risultati 2..5): la distribuzione, non un punto
    tops = pred["top_scores"][1:5]
    if tops:
        def cell(t):
            (a, b), p = t
            return f"{ink(f'{a}–{b}')}   {ink(_fig(f'{p * 100:.1f}', 4))}"
        rowpairs = [tops[i:i + 2] for i in range(0, len(tops), 2)]
        for j, rp in enumerate(rowpairs):
            rl = "also" if j == 0 else ""
            seg = ("   ".join(cell(t) for t in rp))
            L.append(_row(rl, seg))
        L.append("")
    # mercati derivati (dalla matrice, gratis)
    m = pred.get("markets")
    if m:
        L.append(_row("goals o/u 2.5", ink(f"over {m['over25'] * 100:.0f}%") + faint("  ·  ")
                      + ink(f"under {m['under25'] * 100:.0f}%")))
        L.append(_row("both score", ink(f"yes {m['btts'] * 100:.0f}%") + faint("  ·  ")
                      + ink(f"no {m['btts_no'] * 100:.0f}%")))
        L.append(_row("double chance", faint("1X ") + ink(f"{m['dc_1x'] * 100:.0f}%")
                      + faint("   12 ") + ink(f"{m['dc_12'] * 100:.0f}%")
                      + faint("   X2 ") + ink(f"{m['dc_x2'] * 100:.0f}%")))
        L.append(_row("clean sheet", faint(f"{home.lower()} ") + ink(f"{m['clean_home'] * 100:.0f}%")
                      + faint(f"   {away.lower()} ") + ink(f"{m['clean_away'] * 100:.0f}%")))
        L.append("")
    # marcatori
    if scorers_home or scorers_away:
        L.append(_hr())
        L.append("")
        L += _scorer_cols(home, scorers_home or [], away, scorers_away or [])
        L.append("")
    # provenance
    L.append(_hr())
    L.append("")
    prov = meta.get("prov", {})
    if prov.get("model"):
        L.append(_row("model", faint(prov["model"])))
    if has_mkt and prov.get("market"):
        L.append(_row("market", faint(prov["market"])))
    if prov.get("squad"):
        L.append(_row("squad", faint(prov["squad"])))
    if prov.get("fit"):
        L.append(_row("fit", faint(prov["fit"])))
    L.append("")
    L.append(_hr())
    return "\n".join(L)


# ----------------------------------------------------------------- card SVG
def render_card_svg(pred: dict, scorers_home=None, scorers_away=None, meta=None) -> str:
    meta = meta or {}
    esc = html.escape                                  # XML-safe & < > " '
    home, away = esc(pred["home"].upper()), esc(pred["away"].upper())
    tour = esc(str(meta.get("tournament", "FIFA WORLD CUP 2026")))
    prov_model = esc(str(meta.get("prov", {}).get("model", "Dixon–Coles + Elo")))
    prov_fit = esc(str(meta.get("prov", {}).get("fit", "")))
    ph, pd_, pa = pred["p_home"], pred["p_draw"], pred["p_away"]
    x, y = pred["exact"]
    W_, H_ = 1080, 1350
    INK, MUT, PAP = "#111111", "#999999", "#ffffff"
    ax = 96
    bar_w = W_ - 2 * ax
    seg_h = ph * bar_w
    seg_d = pd_ * bar_w

    def scorer_rows(sc, x0, y0):
        out = []
        for i, (name, p, pk) in enumerate(sc[:4]):
            yy = y0 + i * 46
            mark = " (p)" if pk else ""
            out.append(f'<text x="{x0}" y="{yy}" font-size="30" fill="{INK}">{html.escape(name[:18])}{mark}</text>')
            out.append(f'<text x="{x0 + 300}" y="{yy}" font-size="30" fill="{MUT}" text-anchor="end">{round(p*100)}%</text>')
        return "\n".join(out)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{H_}" viewBox="0 0 {W_} {H_}" font-family="ui-monospace, Menlo, monospace">
<rect width="{W_}" height="{H_}" fill="{PAP}"/>
<line x1="{ax}" y1="80" x2="{W_-ax}" y2="80" stroke="{INK}" stroke-width="2"/>
<text x="{ax}" y="180" font-size="64" font-weight="600" fill="{INK}">{home} · {away}</text>
<text x="{ax}" y="226" font-size="26" fill="{MUT}" letter-spacing="3">{tour} · {'NEUTRAL · ' if pred['neutral'] else ''}{esc(str(meta.get('date','')))}</text>
<text x="{ax}" y="380" font-size="22" fill="{MUT}">1</text>
<text x="{ax + bar_w/2:.0f}" y="380" font-size="22" fill="{MUT}" text-anchor="middle">X</text>
<text x="{W_-ax}" y="380" font-size="22" fill="{MUT}" text-anchor="end">2</text>
<text x="{ax}" y="440" font-size="56" fill="{INK}">{ph*100:.1f}</text>
<text x="{ax + bar_w/2:.0f}" y="440" font-size="56" fill="{MUT}" text-anchor="middle">{pd_*100:.1f}</text>
<text x="{W_-ax}" y="440" font-size="56" fill="{INK}" text-anchor="end">{pa*100:.1f}</text>
<rect x="{ax}" y="470" width="{seg_h:.1f}" height="10" fill="{INK}"/>
<rect x="{ax+seg_h:.1f}" y="470" width="{seg_d:.1f}" height="10" fill="{MUT}"/>
<rect x="{ax+seg_h+seg_d:.1f}" y="470" width="{bar_w-seg_h-seg_d:.1f}" height="10" fill="#cccccc"/>
<text x="{ax}" y="660" font-size="160" font-weight="700" fill="{INK}">{x}–{y}</text>
<text x="{W_-ax}" y="640" font-size="28" fill="{MUT}" text-anchor="end">modal · {pred['exact_prob']*100:.1f}%</text>
<text x="{W_-ax}" y="680" font-size="28" fill="{MUT}" text-anchor="end">xG {pred['lambda_home']:.2f} · {pred['lambda_away']:.2f}</text>
<line x1="{ax}" y1="760" x2="{W_-ax}" y2="760" stroke="{MUT}" stroke-width="1"/>
<text x="{ax}" y="810" font-size="24" fill="{MUT}" letter-spacing="2">SCORERS · {home}</text>
<text x="{ax+460}" y="810" font-size="24" fill="{MUT}" letter-spacing="2">SCORERS · {away}</text>
{scorer_rows(scorers_home or [], ax, 870)}
{scorer_rows(scorers_away or [], ax+460, 870)}
<line x1="{ax}" y1="1180" x2="{W_-ax}" y2="1180" stroke="{MUT}" stroke-width="1"/>
<text x="{ax}" y="1230" font-size="22" fill="{MUT}">{prov_model}</text>
<text x="{ax}" y="1262" font-size="22" fill="{MUT}">{prov_fit}</text>
</svg>'''
