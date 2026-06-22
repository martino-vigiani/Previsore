#!/usr/bin/env bash
# Aggiornamento + riaddestramento + previsioni giornaliere (autonomia via cron).
# Esempio crontab (ogni giorno alle 08:00):
#   0 8 * * * /Users/martinovigiani/lab/Previsore/scripts/daily.sh >> /tmp/previsore.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== $(date) ==="
previsore update                 # pull dati CC0 (martj42)
previsore squads || true         # rose 26 reali (Wikipedia, opt-in; non bloccare se fallisce)
previsore odds || true           # quote se PREVISORE_ODDS_API_KEY impostata
previsore fit                    # riaddestra (secondi)
previsore predict --upcoming --limit 32 --scorers   # previsioni + marcatori
