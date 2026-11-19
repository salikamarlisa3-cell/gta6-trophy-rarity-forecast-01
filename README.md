# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/salikamarlisa3-cell/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/salikamarlisa3-cell/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A refresh-aware forecaster for GTA VI's PS5 trophy earn rates on Exophase, one week after launch. Exophase only republishes a game's counts at its own refreshes, hours apart, so the program first collapses a case's read history into the distinct page states it shows. With a dynamic history it forecasts the tracked-player count with a saturating-growth fit, then each trophy's earners two ways — a players-elasticity fit and a logit progression trend, both with slopes shrunk toward tier means — and blends the medians on the logit scale, with a spread built from fit residuals, horizon and player growth. With a flat history (the page has not refreshed inside the window) it applies the dilution the competition documents for launch night: the universal trophy stays at 100%, every other trophy's integer earners scale by the refreshed earned ratio, and the player base by the refreshed players ratio. Pure Python 3 standard library, deterministic, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ntk41cb8&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
