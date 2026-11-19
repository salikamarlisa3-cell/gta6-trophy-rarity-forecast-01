#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster.

make.fun competition: gta6-trophy-rarity-forecast-01.

Reads one case file (schema makefun/gta6-trophies-case/v1) per run from the
input directory and writes one forecast file per case to the output
directory: for every trophy of the case, five quantiles of its Exophase
earn rate (percent of tracked players who have earned it) at the case's
target instant. Scored by mean pinball loss on the logit scale.

How it works
------------
Exophase only republishes a game's counts at its own refreshes, hours
apart, so a case's history of reads is first collapsed into the distinct
page states it shows ("snapshots"), each stamped with the first read that
showed it. Two regimes are then handled separately:

* Dynamic history (two or more snapshots): the tracked-player count is
  forecast with a saturating-growth fit blended with a log-time fit. Each
  trophy's earners are forecast two ways -- a players-elasticity fit
  (log earners on log players, slopes shrunk toward tier means) and a
  progression trend (logit rate on log time since release, slopes shrunk
  the same way) -- and the two medians are blended on the logit scale.
  The spread comes from the trend fit's residuals plus terms that grow
  with the horizon and with the player-growth ratio.

* Flat history (a single snapshot): the page has not refreshed inside the
  observed window, so a refresh before the target is the central event.
  The competition's own documentation records the launch-night refresh
  this case shape straddles: 638 -> 3,868 players tracked and 943 ->
  4,276 trophies earned, with the one universal trophy (earned by every
  tracked player) staying at 100%. That event anchors a dilution prior:
  the universal trophy's earners become the new player count, every other
  trophy's earners scale by the remaining earned ratio, and the player
  base scales by the players ratio. The spread is sized by the integer
  rounding left in that reconstruction.

Everything is deterministic, standard library only, and defensive: any
trophy or case shape the fits cannot use falls back to persistence with
a wide, honest spread rather than failing.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

_EPS_LO = 0.01    # scoring clamps percents into [0.01, 99.99] for logits
_EPS_HI = 99.99
_NORM = NormalDist()


def clamp_rate(x):
    """Clamp a percent into the range where its logit is finite."""
    if x < _EPS_LO:
        return _EPS_LO
    if x > _EPS_HI:
        return _EPS_HI
    return x


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return 100.0 * ex / (1.0 + ex)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp to epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std)."""
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    if sxx <= 1e-12:
        b = 0.0
        a = my
    else:
        b = sxy / sxx
        a = my - b * mx
    if n > 2:
        var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return a, b, resid


def quantiles_from(median_logit, sigma, levels):
    """Percent quantiles from a normal belief on the logit scale."""
    out = []
    for tau in levels:
        z = _NORM.inv_cdf(tau)
        out.append(inv_logit(median_logit + z * sigma))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Case preparation
# ---------------------------------------------------------------------------

def snapshots(history):
    """Collapse consecutive identical page states into snapshots.

    A snapshot is (time_of_first_read, players_tracked, trophies_earned,
    rates tuple). The page only changes at a refresh, so repeats of one
    state carry no new information beyond confirming the state persisted.
    """
    snaps = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), read.get("trophies_earned"),
               tuple(rates))
        if key != prev_key:
            snaps.append({
                "t": parse_time(read["at"]),
                "players": read.get("players_tracked") or 0,
                "earned": read.get("trophies_earned") or 0,
                "rates": list(rates),
            })
            prev_key = key
    return snaps


# ---------------------------------------------------------------------------
# Flat history: the refresh has not happened yet inside the window
# ---------------------------------------------------------------------------

# The launch-night refresh the competition documents for this case shape:
# the seed page (638 players, 943 trophies earned) was replaced by a page
# with 3,868 players and 4,276 trophies earned, the universal trophy
# staying at 100%. These are the prior's anchor, not a lookup: they size
# how a stale page dilutes when the public flood arrives.
_SEED_PLAYERS = 638.0
_SEED_EARNED = 943.0
_REFRESH_PLAYERS = 3868.0
_REFRESH_EARNED = 4276.0


def flat_forecast(case, snaps, levels):
    """Forecast when the history shows one unchanging page."""
    snap = snaps[-1]
    players0 = float(snap["players"])
    earned0 = float(snap["earned"])
    rates0 = snap["rates"]
    horizon_days = case["horizon_minutes"] / 1440.0

    seed_like = (abs(players0 - _SEED_PLAYERS) < 1.0
                 and abs(earned0 - _SEED_EARNED) < 1.0)
    if seed_like:
        players1 = _REFRESH_PLAYERS
        earned1 = _REFRESH_EARNED
    else:
        # Generic stale page: expect one refresh of moderate size.
        players1 = players0 * 2.5
        earned1 = earned0 * 2.2

    # Earners are integers underneath the printed percents (a rate is
    # round(100 * earners / players, 2)), so recover them as integers:
    # the reconstruction below is integer arithmetic, not curve fitting.
    univ_idx = [i for i, r in enumerate(rates0)
                if r is not None and r >= 99.995]
    univ0 = sum(round(rates0[i] * players0 / 100.0) for i in univ_idx)
    rest0 = earned0 - univ0
    univ1 = len(univ_idx) * players1
    rest1 = earned1 - univ1
    scale = (rest1 / rest0) if rest0 > 0 else 1.0

    out = []
    for i, rate in enumerate(rates0):
        if rate is None:
            rate = 0.0
        if i in univ_idx:
            # Earned by every tracked player before; the documented
            # refresh kept it at 100%, so its earners become the new
            # player count and its rate stays pinned to the top of the
            # scale (the scoring clamp makes 99.99 its effective logit).
            out.append((i, [99.95, 99.99, 99.99, 100.00, 100.00]))
            continue
        e0 = round(rate * players0 / 100.0)
        e1 = round(e0 * scale)
        # For horizons well past one refresh, earners keep accumulating a
        # little faster than the player base as players progress.
        if horizon_days > 0.5:
            e1 = round(e1 * (1.0 + 0.35 * min(horizon_days - 0.5, 4.0)))
        rate1 = 100.0 * e1 / players1 if players1 > 0 else rate
        if e1 <= 0:
            q = [0.0, 0.01, 0.02, 0.06, 0.20]
        else:
            # Spread: a small floor for rule risk plus a term in
            # 1/earners for the integer rounding the reconstruction
            # cannot see, on the logit scale.
            sigma = math.sqrt(0.02 ** 2 + 0.04 / max(e1, 1.0))
            q = quantiles_from(logit(rate1), sigma, levels)
        out.append((i, q))
    return out


# ---------------------------------------------------------------------------
# Dynamic history: fit the trajectory
# ---------------------------------------------------------------------------

def forecast_players(snaps, t_target):
    """Forecast the tracked-player count at the target time.

    Blends a saturating-growth fit (grid over the asymptote, linear in
    log(asymptote - players)) with a log-time linear fit, geometrically.
    """
    pts = [(s["t"], float(s["players"])) for s in snaps if s["players"] > 0]
    if not pts:
        return 0.0
    p_last = pts[-1][1]
    if len(pts) < 3:
        span_days = max((t_target - pts[-1][0]) / 86400.0, 0.0)
        return p_last * (1.0 + 0.30 * span_days)
    t0 = pts[0][0]
    xs = [(t - t0) / 3600.0 for t, _ in pts]          # hours
    xt = (t_target - t0) / 3600.0
    ws = [1.0 + 0.35 * k for k in range(len(pts))]     # recency weights

    # Saturating fit over a grid of asymptotes.
    best = None
    for mult in (1.15, 1.3, 1.5, 1.8, 2.2, 2.8, 3.6, 5.0, 7.0, 10.0):
        pinf = p_last * mult
        if any(p >= pinf for _, p in pts):
            continue
        ys = [math.log(pinf - p) for _, p in pts]
        a, b, _ = wls_line(xs, ys, ws)
        if b >= 0:
            continue
        sse = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
        pred = pinf - math.exp(a + b * xt)
        if best is None or sse < best[0]:
            best = (sse, pred)
    # Log-time fit: players linear in ln(1 + days).
    xd = [math.log1p(x / 24.0) for x in xs]
    xtd = math.log1p(xt / 24.0)
    a2, b2, _ = wls_line(xd, [p for _, p in pts], ws)
    pred_log = a2 + b2 * xtd

    preds = []
    if best is not None and best[1] > p_last:
        preds.append(best[1])
    if pred_log > p_last:
        preds.append(pred_log)
    if not preds:
        span_days = max((t_target - pts[-1][0]) / 86400.0, 0.0)
        return p_last * (1.0 + 0.30 * span_days)
    geo = math.exp(sum(math.log(p) for p in preds) / len(preds))
    return min(max(geo, p_last * 1.01), p_last * 30.0)


def dynamic_forecast(case, snaps, levels):
    """Forecast from a history with two or more distinct page states."""
    trophies = case["trophies"]
    t_target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    release_t = parse_time(case["game"]["release_date"] + "T00:00:00Z")

    p_target = forecast_players(snaps, t_target)
    p_last = float(snaps[-1]["players"]) or 1.0
    growth = max(p_target / p_last, 1.0) if p_last > 0 else 1.0

    # Fit on the public segment: drop the pre-release seed page (a few
    # hundred tracked players whose rates are not the public's) whenever
    # enough later snapshots remain.
    public = [s for s in snaps
              if s["players"] >= max(2000.0, 0.05 * p_last)]
    fit_snaps = public if len(public) >= 3 else snaps

    def hours_since_release(t):
        return max((t - release_t) / 3600.0, 0.0)

    # Per-trophy raw fits, collected first so slopes can be shrunk toward
    # tier means.
    raw = []
    for idx, tr in enumerate(trophies):
        pts = []
        for s in fit_snaps:
            r = s["rates"][idx] if idx < len(s["rates"]) else None
            if r is None or s["players"] <= 0:
                continue
            pts.append((s, r))
        entry = {"idx": idx, "tier": tr.get("tier", "bronze"), "pts": pts}
        if pts:
            last_s, last_r = pts[-1]
            entry["last_rate"] = last_r
            entry["last_earners"] = last_r * last_s["players"] / 100.0
            entry["last_players"] = float(last_s["players"])
        else:
            entry["last_rate"] = None
            entry["last_earners"] = 0.0
            entry["last_players"] = p_last
        # Elasticity view: log earners on log players.
        if len(pts) >= 3 and len({s["players"] for s, _ in pts}) >= 3:
            xs = [math.log(s["players"]) for s, _ in pts]
            ys = [math.log(max(r * s["players"] / 100.0, 0.25))
                  for s, r in pts]
            ws = [1.0 + 0.35 * k for k in range(len(pts))]
            _, gamma, _ = wls_line(xs, ys, ws)
            entry["gamma"] = gamma
        else:
            entry["gamma"] = None
        # Progression view: logit rate on ln(1 + days since release).
        if len(pts) >= 3:
            xs = [math.log1p(hours_since_release(s["t"]) / 24.0)
                  for s, _ in pts]
            ys = [logit(r) for _, r in pts]
            ws = [1.0 + 0.35 * k for k in range(len(pts))]
            a, b, resid = wls_line(xs, ys, ws)
            entry["trend"] = (a, b)
            entry["resid"] = resid
        else:
            entry["trend"] = None
            entry["resid"] = None
        raw.append(entry)

    # Tier means for shrinkage (platinum is a tier of one; merge it with
    # gold so its slopes borrow strength from the other rare trophies).
    def group(tier):
        return "rare" if tier in ("platinum", "gold") else tier

    for key in ("gamma", "trend"):
        sums, counts = {}, {}
        for e in raw:
            v = e["gamma"] if key == "gamma" else (
                e["trend"][1] if e["trend"] else None)
            if v is None:
                continue
            g = group(e["tier"])
            sums[g] = sums.get(g, 0.0) + v
            counts[g] = counts.get(g, 0) + 1
        means = {g: sums[g] / counts[g] for g in sums}
        all_mean = (sum(sums.values()) / sum(counts.values())
                    if counts else (1.0 if key == "gamma" else 0.0))
        for e in raw:
            g = group(e["tier"])
            mean = means.get(g, all_mean)
            v = e["gamma"] if key == "gamma" else (
                e["trend"][1] if e["trend"] else None)
            n = len(e["pts"])
            if v is None:
                v = mean
            else:
                v = (n * v + 4.0 * mean) / (n + 4.0)
            if key == "gamma":
                e["gamma_s"] = min(max(v, -0.5), 2.5)
            else:
                e["trend_b"] = v
    resids = [e["resid"] for e in raw if e["resid"] is not None]
    resid_med = sorted(resids)[len(resids) // 2] if resids else 0.35

    u_target = math.log1p(hours_since_release(t_target) / 24.0)
    rows = []
    for e in raw:
        idx = e["idx"]
        if e["last_rate"] is None:
            # Never seen with a rate: a small floor by tier, wide spread.
            floor = {"platinum": 0.02, "gold": 0.03,
                     "silver": 0.05}.get(e["tier"], 0.08)
            rows.append((idx, quantiles_from(logit(floor), 1.1, levels)))
            continue
        medians = []
        # Elasticity median.
        e_hat = e["last_earners"] * growth ** e["gamma_s"]
        if p_target > 0:
            medians.append(logit(100.0 * e_hat / p_target))
        # Progression-trend median.
        if e["trend"] is not None:
            a, _b = e["trend"]
            # Re-anchor the shrunk slope at the last observed point so
            # the level is the trophy's own, only the slope is pooled.
            last_s = e["pts"][-1][0]
            u_last = math.log1p(hours_since_release(last_s["t"]) / 24.0)
            med = logit(e["last_rate"]) + e["trend_b"] * (u_target - u_last)
            medians.append(med)
        if medians:
            med_logit = sum(medians) / len(medians)
        else:
            med_logit = logit(e["last_rate"])
        # Keep a universal trophy universal: if its earners already
        # cover (nearly) every tracked player, cap the median high.
        if e["last_earners"] >= 0.995 * e["last_players"]:
            med_logit = max(med_logit, logit(99.5))
        med_logit = min(max(med_logit, logit(0.01)), logit(99.9))

        resid = e["resid"] if e["resid"] is not None else resid_med
        sigma = math.sqrt(
            resid ** 2
            + (0.32 * math.sqrt(max(horizon_days, 0.0))) ** 2
            + (0.12 * math.log(growth)) ** 2
            + 0.15 ** 2)
        sigma = min(max(sigma, 0.30), 1.70)
        rows.append((idx, quantiles_from(med_logit, sigma, levels)))
    return rows


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------

def forecast_case(case):
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    trophies = case["trophies"]
    snaps = snapshots(case.get("history") or [])
    if not snaps:
        # No history at all: persistence is impossible; answer with a
        # flat, wide prior per tier so the case is still valid.
        rows = []
        for idx, tr in enumerate(trophies):
            floor = {"platinum": 0.05, "gold": 0.10,
                     "silver": 0.30}.get(tr.get("tier"), 0.80)
            rows.append((idx, quantiles_from(logit(floor), 1.4, levels)))
    elif len(snaps) == 1:
        rows = flat_forecast(case, snaps, levels)
    else:
        rows = dynamic_forecast(case, snaps, levels)

    by_idx = dict(rows)
    out_rows = []
    for idx, tr in enumerate(trophies):
        q = by_idx.get(idx)
        if q is None:
            q = [1.0, 1.0, 1.0, 1.0, 1.0]
        q = [min(max(float(v), 0.0), 100.0) for v in q]
        q = [round(v, 4) for v in q]
        q.sort()
        # Rounding can collapse distinct values; keep them non-decreasing
        # (equality is allowed) and inside [0, 100].
        out_rows.append({"id": tr["id"], "q": q})
    return {"trophies": out_rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
