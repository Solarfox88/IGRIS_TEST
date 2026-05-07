# Claude Cowork Onboarding — IGRIS_GPT

Questo documento serve ad allineare una sessione Claude Cowork al progetto IGRIS_GPT.

Cowork deve agire come cabina di regia strategica: roadmap, analisi report, priorità, prompt per Claude Code, review architetturale e supervisione dei Rank test.

Claude Code resta l'esecutore operativo sulla VM.

## Fonti principali

- Repository: `Solarfox88/IGRIS_GPT`
- Roadmap: `docs/IGRIS_GPT_MASTER_ROADMAP.md`
- PR critica recente: `#77` — destructive write guard, safe edit actions, patch-first policy

## Ruoli

### Cowork

- ragionamento strategico;
- revisione roadmap;
- analisi dei report di Claude Code;
- preparazione mandati operativi;
- valutazione Rank C/B/A/S;
- identificazione rischi e prossimi epic.

### Claude Code

- lavoro operativo sulla VM;
- test;
- modifiche codice;
- GitHub workflow;
- report operativo.

## Stato recente da conoscere

PR #77 risulta mergiata e dovrebbe includere:

- destructive write guard;
- safe edit actions;
- patch-first policy;
- fix del test di modifica file;
- fix del test doctor su root inesistente;
- full pytest dichiarato verde nel commit finale.

La prima validazione utile è confermare su VM:

- main aggiornato;
- full pytest verde;
- servizio sano;
- Rank C confermato;
- Rank B tentato solo dopo i controlli di sicurezza.

## Rank system

- C: installazione, test e benchmark sani;
- B: micro-feature reale completata da IGRIS in autonomia con test e senza danni;
- A: task multi-file reale;
- S: missione end-to-end con GitHub, safety, rollback e report.

## Output richiesto a Cowork

Cowork deve sempre produrre:

- cosa ha capito;
- accessi disponibili;
- stato stimato di IGRIS;
- rischi;
- prompt o assignment per Claude Code;
- una prossima azione consigliata.

## Prompt completo

Il prompt completo di onboarding può essere mantenuto nella cabina strategica esterna e aggiornato in base ai report di Claude Code.