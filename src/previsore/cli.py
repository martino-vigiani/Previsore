"""Interfaccia a riga di comando: update / fit / predict / backtest."""
from __future__ import annotations

import argparse
import sys

from . import data as datamod
from .model import DixonColes

MODEL_PATH = datamod.DATA_DIR / "model.json"


def _load_model(played, refit: bool = False, **kw) -> DixonColes:
    if MODEL_PATH.exists() and not refit:
        return DixonColes.from_json(MODEL_PATH)
    m = DixonColes.fit(played, **kw)
    m.to_json(MODEL_PATH)
    return m


def _fmt_scorers(model, goals, pred) -> str:
    from . import scorers
    lines = []
    for side, team, lam in (("casa", pred["home"], pred["lambda_home"]),
                            ("ospite", pred["away"], pred["lambda_away"])):
        sh = scorers.player_shares(goals, team, model.ref_date)
        top = scorers.predict_scorers(lam, sh, topn=4)
        if top:
            txt = ", ".join(f"{p} {pa * 100:.0f}%" for p, pa, _ in top)
        else:
            txt = "(nessun dato gol recente)"
        lines.append(f"  marcatori {side} ({team}): {txt}")
    return "\n".join(lines)


def _fmt(pred: dict) -> str:
    x, y = pred["exact"]
    neutro = "  (campo neutro)" if pred["neutral"] else ""
    tops = ", ".join(f"{a}-{b} {p * 100:.1f}%" for (a, b), p in pred["top_scores"])
    return "\n".join([
        f"  {pred['home']} {x}-{y} {pred['away']}{neutro}",
        f"  gol attesi:  {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}",
        f"  1X2:         1={pred['p_home'] * 100:4.1f}%   X={pred['p_draw'] * 100:4.1f}%   2={pred['p_away'] * 100:4.1f}%",
        f"  esatto piu probabile: {x}-{y} ({pred['exact_prob'] * 100:.1f}%)",
        f"  top risultati: {tops}",
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="previsore",
        description="Predittore risultato esatto + 1X2 (Dixon-Coles). Dati CC0, solo CPU.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("update", help="scarica/aggiorna i dati (martj42, CC0)")

    pf = sub.add_parser("fit", help="addestra e salva il modello")
    pf.add_argument("--half-life-days", type=float, default=730.0)
    pf.add_argument("--window-years", type=int, default=12)
    pf.add_argument("--min-matches", type=int, default=25)

    pp = sub.add_parser("predict", help="predici una partita o le prossime fixture")
    pp.add_argument("--home")
    pp.add_argument("--away")
    pp.add_argument("--neutral", action="store_true", help="campo neutro (default Mondiale)")
    pp.add_argument("--scorers", action="store_true", help="aggiungi marcatori probabili (euristica)")
    pp.add_argument("--upcoming", action="store_true", help="predici tutte le fixture future nei dati")
    pp.add_argument("--limit", type=int, default=20)
    pp.add_argument("--refit", action="store_true", help="riaddestra anche se esiste un modello salvato")

    pb = sub.add_parser("backtest", help="backtest temporale (RPS, 1X2, esatto)")
    pb.add_argument("--cutoff", default="2024-01-01")

    args = ap.parse_args(argv)

    if args.cmd == "update":
        datamod.update()
        return 0

    df = datamod.load_results()
    pl = datamod.played(df)

    if args.cmd == "fit":
        m = DixonColes.fit(pl, half_life_days=args.half_life_days,
                           window_years=args.window_years, min_matches=args.min_matches)
        m.to_json(MODEL_PATH)
        print(f"modello salvato -> {MODEL_PATH}")
        return 0

    if args.cmd == "predict":
        m = _load_model(pl, refit=args.refit)
        goals = datamod.load_goalscorers() if args.scorers else None
        if args.upcoming:
            fut = datamod.future(df).sort_values("date")
            shown = 0
            for r in fut.itertuples(index=False):
                if r.home_team not in m.attack or r.away_team not in m.attack:
                    continue
                pred = m.predict(r.home_team, r.away_team, bool(r.neutral))
                print(f"\n{r.date.date()}  [{r.tournament}]")
                print(_fmt(pred))
                if args.scorers:
                    print(_fmt_scorers(m, goals, pred))
                shown += 1
                if shown >= args.limit:
                    break
            if shown == 0:
                print("Nessuna fixture futura predicibile nei dati.")
            return 0
        if not args.home or not args.away:
            print("Serve --home e --away, oppure --upcoming.", file=sys.stderr)
            return 2
        pred = m.predict(args.home, args.away, args.neutral)
        print(_fmt(pred))
        if args.scorers:
            print(_fmt_scorers(m, goals, pred))
        return 0

    if args.cmd == "backtest":
        from . import backtest
        res = backtest.run(pl, args.cutoff)
        if res["n"] == 0:
            print("Nessuna partita di test nel periodo scelto.")
            return 0
        print(f"Backtest da {args.cutoff} - {res['n']} partite di test")
        print(f"  RPS         DC={res['rps_dc']:.4f}   Elo={res['rps_elo']:.4f}   (piu basso = meglio; bookie ~0.20)")
        print(f"  Acc 1X2     DC={res['acc1x2_dc'] * 100:.1f}%   Elo={res['acc1x2_elo'] * 100:.1f}%")
        print(f"  Esatto      DC={res['exact_dc'] * 100:.1f}%   (tetto reale ~9-15%)")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
