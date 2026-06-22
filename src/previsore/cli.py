"""Interfaccia a riga di comando.

Comandi: update / fit / predict / evaluate / walkforward / backtest / odds / squads.
Default (niente chiave quote, niente cache rose) = identico a prima e offline.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import blend
from . import data as datamod
from . import render
from .model import DixonColes

MODEL_PATH = datamod.DATA_DIR / "model.json"
CONFIG_PATH = datamod.DATA_DIR / "config.json"

# numeri di validazione (walk-forward 2018-2026), mostrati come provenienza
_FIT_LINE = "7875 gp · RPS 0.169 · log-loss 0.867 · ECE 1.7%"


def _load_predictor(played, cfg, use_odds=False, refit=False, **kw) -> blend.Predictor:
    if MODEL_PATH.exists() and not refit:
        m = DixonColes.from_json(MODEL_PATH)
    else:
        m = DixonColes.fit(played, **kw)
        m.to_json(MODEL_PATH)
    R = blend.elo_ratings(played)
    market_w, odds_table = 0.0, None
    if use_odds:                              # solo con --odds, non dal solo env
        from . import odds
        odds_table = odds.get_odds_table()
        if odds_table is not None:
            mw = cfg.get("market_w")          # rispetta uno 0.0 esplicito; assente -> 0.5
            market_w = mw if mw is not None else 0.5
    return blend.Predictor(m, R, w=cfg.get("ensemble_w", 1.0), T=cfg.get("temperature", 1.0),
                           market_w=market_w, odds_table=odds_table)


def _prov(predictor, has_market: bool, has_squad: bool) -> dict:
    p = {"model": f"Dixon–Coles + Elo · cal T {predictor.T} · ref {predictor.dc.ref_date}",
         "fit": _FIT_LINE}
    if has_market:
        p["market"] = f"the-odds-api · Shin de-vig · anchor w {predictor.market_w:.2f}"
    if has_squad:
        p["squad"] = "26-man finals · Wikipedia · penalty (p) routed"
    return p


def _scorers_for(goals, team, lam, tmap, ref):
    from . import scorers, squads
    toks = squads.squad_tokens_for(team, tmap) if tmap else None
    return scorers.scorer_probs(goals, team, ref, lam, squad_tokens=toks, topn=4), (toks is not None)


def _emit(predictor, pred, goals, tmap, date, tournament, card_path=None):
    sh = sa = None
    has_squad = False
    if goals is not None:
        ref = predictor.dc.ref_date
        sh, hs = _scorers_for(goals, pred["home"], pred["lambda_home"], tmap, ref)
        sa, as_ = _scorers_for(goals, pred["away"], pred["lambda_away"], tmap, ref)
        has_squad = hs or as_
    meta = {"date": date, "tournament": tournament,
            "prov": _prov(predictor, pred.get("p_market") is not None, has_squad)}
    print(render.render_terminal(pred, sh, sa, meta))
    if card_path:
        from pathlib import Path
        Path(card_path).write_text(render.render_card_svg(pred, sh, sa, meta))
        print(f"\n  card -> {card_path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="previsore",
        description="Predittore risultato esatto + 1X2 (Dixon-Coles + Elo + mercato). Dati CC0, CPU.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("update", help="scarica/aggiorna i dati (martj42, CC0)")
    sub.add_parser("odds", help="aggiorna le quote (the-odds-api, serve PREVISORE_ODDS_API_KEY)")
    sub.add_parser("squads", help="scarica le rose 26 reali da Wikipedia")

    pf = sub.add_parser("fit", help="addestra, tara il blend e salva modello + config")
    pf.add_argument("--half-life-days", type=float, default=1095.0)
    pf.add_argument("--window-years", type=int, default=12)
    pf.add_argument("--min-matches", type=int, default=15)
    pf.add_argument("--reg", type=float, default=1.0)
    pf.add_argument("--tune-cutoff", default="2023-01-01")
    pf.add_argument("--no-tune", action="store_true")

    pp = sub.add_parser("predict", help="predici una partita o le prossime fixture")
    pp.add_argument("--home")
    pp.add_argument("--away")
    pp.add_argument("--neutral", action="store_true")
    pp.add_argument("--scorers", action="store_true", help="aggiungi marcatori probabili")
    pp.add_argument("--odds", action="store_true", help="ancora al mercato (quote bookmaker)")
    pp.add_argument("--card", metavar="FILE.svg", help="esporta una card SVG condivisibile")
    pp.add_argument("--upcoming", action="store_true")
    pp.add_argument("--limit", type=int, default=20)
    pp.add_argument("--refit", action="store_true")
    pp.add_argument("--all", action="store_true", help="con --upcoming: mostra anche date passate")

    pb = sub.add_parser("backtest", help="backtest split singolo")
    pb.add_argument("--cutoff", default="2024-01-01")

    pe = sub.add_parser("evaluate", help="predizioni vs partite GIA giocate (out-of-sample)")
    pe.add_argument("--cutoff", default="2026-06-11")
    pe.add_argument("--tournament", default="FIFA World Cup")
    pe.add_argument("--since", default=None)
    pe.add_argument("--scorers", action="store_true")
    pe.add_argument("--examples", type=int, default=12)

    pw = sub.add_parser("walkforward", help="validazione onesta a finestra espandente")
    pw.add_argument("--start", default="2018-01-01")
    pw.add_argument("--end", default="2026-06-11")

    args = ap.parse_args(argv)

    if args.cmd == "update":
        datamod.update()
        return 0
    if args.cmd == "odds":
        from . import odds
        t = odds.get_odds_table()
        print(f"quote: {len(t)} partite -> {odds.ODDS_CSV}" if t is not None
              else "nessuna quota (serve PREVISORE_ODDS_API_KEY o data/odds.csv).")
        return 0
    if args.cmd == "squads":
        from . import squads
        squads.update_squads()
        print(f"rose -> {squads.SQUADS_CSV}")
        return 0

    df = datamod.load_results()
    pl = datamod.played(df)

    if args.cmd == "fit":
        fit_kw = dict(half_life_days=args.half_life_days, window_years=args.window_years,
                      min_matches=args.min_matches, reg=args.reg)
        m = DixonColes.fit(pl, **fit_kw)
        m.to_json(MODEL_PATH)
        cfg = {"ensemble_w": 1.0, "temperature": 1.0, **fit_kw}
        if not args.no_tune:
            print(f"taro il blend (val dal {args.tune_cutoff})...")
            tuned = blend.tune(pl, val_cutoff=args.tune_cutoff, **fit_kw)
            cfg.update(tuned)
            print(f"  ensemble_w={tuned['ensemble_w']}  temperature={tuned['temperature']}  "
                  f"(val log-loss={tuned['val_logloss']:.4f} su {tuned['val_n']} partite)")
        blend.save_config(CONFIG_PATH, cfg)
        print(f"modello -> {MODEL_PATH}\nconfig  -> {CONFIG_PATH}")
        return 0

    if args.cmd == "predict":
        cfg = blend.load_config(CONFIG_PATH)
        pe_ = _load_predictor(pl, cfg, use_odds=args.odds, refit=args.refit)
        goals = datamod.load_goalscorers() if args.scorers else None
        tmap = {}
        if args.scorers:
            from . import squads
            tmap = squads.tokens_by_team()
        if args.upcoming:
            if args.card:
                print("nota: --card vale solo per una partita singola, ignorato con --upcoming",
                      file=sys.stderr)
            fut = datamod.future(df).sort_values("date")
            if not args.all:
                fut = fut[fut["date"] >= pd.Timestamp.today().normalize()]
            shown = 0
            for r in fut.itertuples(index=False):
                if r.home_team not in pe_.dc.attack or r.away_team not in pe_.dc.attack:
                    continue
                pred = pe_.predict(r.home_team, r.away_team, bool(r.neutral))
                _emit(pe_, pred, goals, tmap, str(r.date.date()), r.tournament)
                print()
                shown += 1
                if shown >= args.limit:
                    break
            if shown == 0:
                print("Nessuna fixture futura predicibile nei dati.")
            return 0
        if not args.home or not args.away:
            print("Serve --home e --away, oppure --upcoming.", file=sys.stderr)
            return 2
        pred = pe_.predict(args.home, args.away, args.neutral)
        _emit(pe_, pred, goals, tmap, "", "FIFA World Cup 2026", card_path=args.card)
        return 0

    if args.cmd == "backtest":
        from . import backtest
        res = backtest.run(pl, args.cutoff)
        if res["n"] == 0:
            print("Nessuna partita di test nel periodo scelto.")
            return 0
        print(f"Backtest da {args.cutoff} - {res['n']} partite")
        print(f"  RPS     DC={res['rps_dc']:.4f}  Elo={res['rps_elo']:.4f}")
        print(f"  Acc1X2  DC={res['acc1x2_dc']*100:.1f}%  Elo={res['acc1x2_elo']*100:.1f}%")
        print(f"  Esatto  DC={res['exact_dc']*100:.1f}%")
        return 0

    if args.cmd == "evaluate":
        from . import evaluate
        cfg = blend.load_config(CONFIG_PATH)
        goals = datamod.load_goalscorers() if args.scorers else None
        res = evaluate.run(df, cutoff=args.cutoff, tournament=args.tournament, since=args.since,
                           goals=goals, w=cfg.get("ensemble_w", 1.0), T=cfg.get("temperature", 1.0))
        if res["n"] == 0:
            print("Nessuna partita giocata da valutare.")
            return 0
        since = args.since or args.cutoff
        print(f"VALIDAZIONE '{args.tournament}' dal {since} - {res['n']} partite (addestrato < {args.cutoff})")
        print(f"  {'metrica':<10} {'blend':>8} {'DC':>8} {'Elo':>8}")
        for key, lab in (("acc", "Acc 1X2"), ("rps", "RPS"), ("logloss", "log-loss"), ("brier", "Brier")):
            sc = 100 if key == "acc" else 1
            su = "%" if key == "acc" else ""
            print(f"  {lab:<10} {res['blend'][key]*sc:>7.3f}{su} {res['dc'][key]*sc:>7.3f}{su} {res['elo'][key]*sc:>7.3f}{su}")
        print(f"  Risultato esatto (DC): {res['exact']*100:.1f}%")
        ci = res["ci_blend_minus_elo_logloss"]
        print(f"  CI95 log-loss (blend - Elo): [{ci[0]:+.4f}, {ci[1]:+.4f}]"
              f"{'  (non significativo a questo n)' if ci[0] < 0 < ci[1] else '  (significativo)'}")
        if "sc_n" in res:
            print(f"  Marcatori ({res['sc_n']}): top-1 {res['sc_t1']*100:.1f}%  top-3 {res['sc_t3']*100:.1f}%")
        if "sc_cal" in res:
            c = res["sc_cal"]
            print(f"  Calibrazione marcatori: Brier {c['brier']:.4f}  ECE {c['ece']*100:.1f}%  "
                  f"(massa predetta {c['mass_pred']:.0f} vs reale {c['mass_real']})")
            for lo, hi, pm, rm, n in c["bands"]:
                flag = "  <-- gonfiato" if pm - rm > 0.06 else ""
                print(f"    {int(lo*100):2}-{int(hi*100):3}%: predetto {pm*100:4.1f}%  reale {rm*100:4.1f}%  (n={n}){flag}")
        if args.examples:
            print("\n  data        predetto -> reale            esito esatto")
            for dte, pr, real, o, e in res["examples"][:args.examples]:
                print(f"  {dte}  {pr:<28} {real:<6} {'OK' if o else '. '}    {'OK' if e else '.'}")
        return 0

    if args.cmd == "walkforward":
        from . import backtest
        cfg = blend.load_config(CONFIG_PATH)
        print(f"walk-forward {args.start} -> {args.end} (refit annuale, w={cfg.get('ensemble_w', 1.0)}, "
              f"T={cfg.get('temperature', 1.0)})... ~1 min")
        res = backtest.walk_forward(pl, start=args.start, end=args.end,
                                    w=cfg.get("ensemble_w", 1.0), T=cfg.get("temperature", 1.0))
        if res["n"] == 0:
            print("Nessuna partita valutabile.")
            return 0
        print(f"\n{res['n']} partite out-of-sample")
        print(f"  {'metrica':<10} {'blend':>8} {'DC':>8} {'Elo':>8}")
        for key, lab in (("acc", "Acc 1X2"), ("rps", "RPS"), ("logloss", "log-loss"), ("brier", "Brier"), ("ece", "ECE")):
            sc = 100 if key in ("acc", "ece") else 1
            su = "%" if key in ("acc", "ece") else ""
            print(f"  {lab:<10} {res[key]['blend']*sc:>7.3f}{su} {res[key]['dc']*sc:>7.3f}{su} {res[key]['elo']*sc:>7.3f}{su}")
        lo, hi = res["ci_logloss_blend_minus_elo"]
        print(f"  CI95 log-loss (blend - Elo): [{lo:+.4f}, {hi:+.4f}]  "
              f"{'blend meglio (significativo)' if hi < 0 else 'non conclusivo' if lo < 0 else 'Elo meglio'}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
