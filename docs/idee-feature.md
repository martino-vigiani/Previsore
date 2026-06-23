# Idee feature — previsioni "context-aware"

Idee per arricchire le previsioni con il **contesto della competizione**, non
solo con la forza relativa delle squadre (Dixon–Coles + Elo + mercato). L'idea
di fondo: l'esito di una partita non dipende solo da "chi è più forte", ma da
**cosa si gioca** ciascuna squadra in quel momento.

## 1. Valutare la competizione

- Pesare il tipo di torneo: amichevole vs qualificazioni vs fase finale.
- Nelle amichevoli la motivazione/turnover abbassa la prevedibilità → la
  previsione dovrebbe essere più "morbida" (probabilità più vicine al 50/50,
  varianza gol più alta).
- Stage della competizione (gironi / ottavi / quarti / … / finale) come
  feature: incide su intensità e gestione forze.

## 2. Posizione nel girone / classifica

- Leggere la classifica corrente del girone prima della partita.
- Capire cosa serve a ciascuna squadra: già qualificata, deve vincere, le
  basta il pari, è già eliminata.
- Una squadra "salva" o già qualificata tende a gestire → abbassa l'attesa
  offensiva; una che "deve vincere o esce" tende a sbilanciarsi → più gol,
  più varianza.

## 3. Differenza reti (goal difference) e incentivi

- Calcolare la GD attuale di ciascuna squadra nel girone.
- Casi tipici:
  - chi è dietro nella GD e deve recuperare → spinge per il largo (più gol
    attesi, più rischio).
  - scontro diretto tra **prima e seconda** già qualificate → spesso partita
    di gestione, GD già alta, poco da giocarsi → meno gol attesi.
- La GD può cambiare l'incentivo a segnare ancora anche a risultato acquisito
  (es. serve +1 sulla differenza reti per superare una terza squadra).

## 4. Lato del tabellone (bracket) della fase finale

- Capire in quale lato del tabellone si finisce in base al piazzamento nel
  girone (1ª vs 2ª pescano avversari diversi agli ottavi).
- Da qui derivano scenari di **convenienza**:
  - "biscotto" / risultato che fa comodo a entrambe (pareggio o esito
    concordato che qualifica entrambe e/o evita l'incrocio peggiore).
  - scelta di arrivare 2ª per pescare un lato di tabellone più morbido.
- Feature che stima la *probabilità di gestione/biscotto* e corregge la
  previsione di conseguenza (meno gol, esito più "comodo" più probabile).

## Note di implementazione (da valutare)

- Serve uno stato "classifica girone + calendario residuo" derivato dai
  risultati → modulo nuovo (es. `standings.py`) che alimenta le feature.
- Le correzioni sono **moltiplicatori sui λ attesi** (xG home/away) prima del
  Dixon–Coles, oppure uno shift sulle probabilità 1X2 finali.
- Rischio: features di "motivazione" sono rumorose e difficili da calibrare →
  introdurle solo con backtest dedicato, altrimenti peggiorano RPS/log-loss.
- Lo scenario "biscotto" è raro e politicamente sensibile: trattarlo come
  flag/avviso più che come correzione forte.
