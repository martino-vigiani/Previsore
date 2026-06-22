"""Interfaccia a riga di comando: update / fit / predict / backtest / evaluate / walkforward."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import blend
from . import data as datamod
from .model import DixonColes

MODEL_PATH = datamod.DATA_DIR / "model.json"
CONFIG_PATH = datamod.DATA_DIR / "config.json"


def _load_predictor(played, refit: bool = False, **kw) -> blend.Predictor:
    if MODEL_PATH.exists() and not refit:
        m = DixonColes.from_json(MODEL_PATH)
    else:
        m = DixonColes.fit(played, **kw)
        m.to_json(MODEL_PATH)
    cfg = blend.load_config(CONFIG_PATH)
    R = blend.elo_ratings(played)
    return blend.Predictor(m, R, w=cfg.get("ensemble_w", 1.0), T=cfg.get("temperature", 1.0))


def _fmt_scorers(ref_date, goals, pred) -> str:
    from . import scorers
    lines = []
    for side, team, lam in (("casa", pred["home"], pred["lambda_home"]),
                            ("ospite", pred["away"], pred["lambda_away"])):
        sh = scorers.player_shares(goals, team, ref_date)
        top = scorers.predict_scorers(lam, sh, topn=4)
        txt = ", ".join(f"{p} {pa * 100:.0f}%" for p, pa, _ in top) if top else "(nessun marcatore attivo recente)"
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
        description="Predittore risultato esatto + 1X2 (Dixon-Coles + ensemble Elo). Dati CC0, solo CPU.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("update", help="scarica/aggiorna i dati (martj42, CC0)")

    pf = sub.add_parser("fit", help="addestra, tara il blend e salva modello + config")
    pf.add_argument("--half-life-days", type=float, default=1095.0)
    pf.add_argument("--window-years", type=int, default=12)
    pf.add_argument("--min-matches", type=int, default=15)
    pf.add_argument("--reg", type=float, default=1.0, help="forza ridge (pooling minnow)")
    pf.add_argument("--tune-cutoff", default="2023-01-01", help="validazione per tarare w,T")
    pf.add_argument("--no-tune", action="store_true", help="non tarare il blend (w=1, T=1)")

    pp = sub.add_parser("predict", help="predici una partita o le prossime fixture")
    pp.add_argument("--home")
    pp.add_argument("--away")
    pp.add_argument("--neutral", action="store_true", help="campo neutro (default Mondiale)")
    pp.add_argument("--scorers", action="store_true", help="aggiungi marcatori probabili")
    pp.add_argument("--upcoming", action="store_true", help="predici le fixture future nei dati")
    pp.add_argument("--limit", type=int, default=20)
    pp.add_argument("--refit", action="store_true", help="riaddestra anche se esiste un modello salvato")
    pp.add_argument("--all", action="store_true", help="con --upcoming: mostra anche date passate")

    pb = sub.add_parser("backtest", help="backtest temporale split singolo (RPS, 1X2, esatto)")
    pb.add_argument("--cutoff", default="2024-01-01")

    pe = sub.add_parser("evaluate", help="confronta predizioni vs partite GIA giocate (out-of-sample)")
    pe.add_argument("--cutoff", default="2026-06-11", help="addestra solo su dati prima di questa data")
    pe.add_argument("--tournament", default="FIFA World Cup")
    pe.add_argument("--since", default=None, help="valuta partite dal (default = cutoff)")
    pe.add_argument("--scorers", action="store_true", help="valuta anche i marcatori")
    pe.add_argument("--examples", type=int, default=12)

    pw = sub.add_parser("walkforward", help="valutazione onesta a finestra espandente (refit annuale)")
    pw.add_argument("--start", default="2018-01-01")
    pw.add_argument("--end", default="2026-06-11")

    args = ap.parse_args(argv)

    if args.cmd == "update":
        datamod.update()
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
        pe_ = _load_predictor(pl, refit=args.refit)
        goals = datamod.load_goalscorers() if args.scorers else None
        ref = pe_.dc.ref_date
        if args.upcoming:
            fut = datamod.future(df).sort_values("date")
            if not args.all:
                fut = fut[fut["date"] >= pd.Timestamp.today().normalize()]
            shown = 0
            for r in fut.itertuples(index=False):
                if r.home_team not in pe_.dc.attack or r.away_team not in pe_.dc.attack:
                    continue
                pred = pe_.predict(r.home_team, r.away_team, bool(r.neutral))
                print(f"\n{r.date.date()}  [{r.tournament}]")
                print(_fmt(pred))
                if args.scorers:
                    print(_fmt_scorers(ref, goals, pred))
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
        print(_fmt(pred))
        if args.scorers:
            print(_fmt_scorers(ref, goals, pred))
        return 0

    if args.cmd == "backtest":
        from . import backtest
        res = backtest.run(pl, args.cutoff)
        if res["n"] == 0:
            print("Nessuna partita di test nel periodo scelto.")
            return 0
        print(f"Backtest da {args.cutoff} - {res['n']} partite di test")
        print(f"  RPS         DC={res['rps_dc']:.4f}   Elo={res['rps_elo']:.4f}   (piu basso = meglio)")
        print(f"  Acc 1X2     DC={res['acc1x2_dc'] * 100:.1f}%   Elo={res['acc1x2_elo'] * 100:.1f}%")
        print(f"  Esatto      DC={res['exact_dc'] * 100:.1f}%   (tetto reale ~9-15%)")
        return 0

    if args.cmd == "evaluate":
        from . import evaluate
        cfg = blend.load_config(CONFIG_PATH)
        goals = datamod.load_goalscorers() if args.scorers else None
        res = evaluate.run(df, cutoff=args.cutoff, tournament=args.tournament, since=args.since,
                           goals=goals, w=cfg.get("ensemble_w", 1.0), T=cfg.get("temperature", 1.0))
        if res["n"] == 0:
            print("Nessuna partita giocata da valutare nel periodo/torneo scelto.")
            return 0
        since = args.since or args.cutoff
        print(f"VALIDAZIONE '{args.tournament}' dal {since} - {res['n']} partite giocate "
              f"(addestrato su dati < {args.cutoff}, niente leakage)")
        print(f"  {'metrica':<10} {'blend':>8} {'DC':>8} {'Elo':>8}")
        for key, lab in (("acc", "Acc 1X2"), ("rps", "RPS"), ("logloss", "log-loss"), ("brier", "Brier")):
            scale = 100 if key == "acc" else 1
            suf = "%" if key == "acc" else ""
            print(f"  {lab:<10} {res['blend'][key] * scale:>7.3f}{suf} {res['dc'][key] * scale:>7.3f}{suf} "
                  f"{res['elo'][key] * scale:>7.3f}{suf}")
        print(f"  Risultato esatto (DC): {res['exact'] * 100:.1f}%")
        ci = res["ci_blend_minus_elo_logloss"]
        sig = "" if ci[0] < 0 < ci[1] else "  (significativo)"
        print(f"  CI95 log-loss (blend - Elo): [{ci[0]:+.4f}, {ci[1]:+.4f}]{sig or '  (attraversa 0: non significativo a questo n)'}")
        if "sc_n" in res:
            print(f"  Marcatori ({res['sc_n']} squadre-partita): top-1 {res['sc_t1'] * 100:.1f}%  top-3 {res['sc_t3'] * 100:.1f}%")
        if args.examples:
            print("\n  data        predetto -> reale            esito esatto")
            for dte, pred, real, o, e in res["examples"][:args.examples]:
                print(f"  {dte}  {pred:<28} {real:<6} {'OK' if o else '. '}    {'OK' if e else '.'}")
        return 0

    if args.cmd == "walkforward":
        from . import backtest
        cfg = blend.load_config(CONFIG_PATH)
        print(f"walk-forward {args.start} -> {args.end} (refit annuale, w={cfg.get('ensemble_w', 1.0)}, "
              f"T={cfg.get('temperature', 1.0)})... puo richiedere ~1 min")
        res = backtest.walk_forward(pl, start=args.start, end=args.end,
                                    w=cfg.get("ensemble_w", 1.0), T=cfg.get("temperature", 1.0))
        if res["n"] == 0:
            print("Nessuna partita valutabile.")
            return 0
        print(f"\n{res['n']} partite valutate out-of-sample")
        print(f"  {'metrica':<10} {'blend':>8} {'DC':>8} {'Elo':>8}")
        for key, lab in (("acc", "Acc 1X2"), ("rps", "RPS"), ("logloss", "log-loss"), ("brier", "Brier"), ("ece", "ECE")):
            scale = 100 if key in ("acc", "ece") else 1
            suf = "%" if key in ("acc", "ece") else ""
            print(f"  {lab:<10} {res[key]['blend'] * scale:>7.3f}{suf} {res[key]['dc'] * scale:>7.3f}{suf} "
                  f"{res[key]['elo'] * scale:>7.3f}{suf}")
        lo, hi = res["ci_logloss_blend_minus_elo"]
        print(f"  CI95 log-loss (blend - Elo): [{lo:+.4f}, {hi:+.4f}]  "
              f"{'blend meglio (significativo)' if hi < 0 else 'non conclusivo' if lo < 0 else 'Elo meglio'}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
