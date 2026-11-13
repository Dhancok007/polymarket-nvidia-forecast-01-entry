import json, math, os, sys

# Forecaster v3 for Polymarket "Next CEO of NVIDIA?" minute prices.
# Same ensemble skeleton as candidate2 (absolute + logit space, EWMA
# mean-abs vol, pooled empirical quantiles of standardised changes,
# fast-pass blend, baseline blend, horizon tail shaping, rvol boost),
# with a rebuilt quantile-table stage and retuned constants:
#  - table samples are drawn only from the last RECENT=2160 minutes of
#    each history (older regimes, including the market-open price
#    discovery, no longer contaminate the tails);
#  - standardised samples are winsorized at +-10 before quantiles are
#    taken (one-off repricing crashes reached -1200 sigma raw);
#  - spread squeezed (SCALE 0.65, CAL (.5,.9,1,.9,.5), stronger tail
#    shaping) to bring 90% coverage from ~0.94 toward ~0.92;
#  - vol halflife 540 and fast-pass weight 0.35.
# Tested and rejected this round: per-market tables shrunk to pooled,
# vol-state conditioned scaling, robust-center (TWAP/median) shifts.
# Final entry: this v3 model replaces candidate2 (official dev loss
# 0.271305, coverage_90 0.9448 flagged by the platform) after beating it
# on practice, extra, late-extra, and the equal-weighted mean locally.

BUCKETS = (10, 20, 40, 80, 160, 240)
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
CAL = (0.5, 0.9, 1.0, 0.9, 0.5)
SCALE = 0.65
HALFLIFE = 540.0
HL_FAST = 300.0
W_FAST = 0.35
BLEND_BASE = 0.1
EPS = 0.0005
ZNORM = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
# per horizon bucket (1-60, 61-120, 121-180, 181-240): multipliers on
# (q_j - median) for j = 0..4
SHAPE_G = ((1.0, 1.0, 1.0, 1.0, 1.0),
           (0.82, 0.93, 1.0, 0.93, 0.82),
           (0.75, 0.91, 1.0, 0.91, 0.75),
           (0.78, 0.92, 1.0, 0.92, 0.78))
RVOL_BETA = 0.3
RVOL_WIN = 90
RVOL_CAP = 2.0
RECENT = 2160        # table samples only from the last RECENT minutes of history
WINSOR = 10.0        # clip standardised samples to +-WINSOR when building tables
SHRINK_K = 0.0       # per-market table shrinkage toward pooled (0 = pooled only; tested, no gain)


def clamp6(v):
    return round(min(1.0, max(0.0, v)), 6) + 0


def volatility(prices, halflife):
    """EWMA of |1-minute changes| with the given halflife (minutes)."""
    n = len(prices)
    if n < 3:
        return 0.0
    lam = 0.5 ** (1.0 / halflife)
    s = 0.0
    w = 0.0
    lw = 1.0
    for i in range(n - 1, 0, -1):
        s += lw * abs(prices[i] - prices[i - 1])
        w += lw
        lw *= lam
        if lw < 1e-9:
            break
    return s / w if w > 0 else 0.0


def quantile(sorted_vals, tau):
    pos = tau * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def std_quantiles(histories, vols):
    """Per-market quantile tables of standardised changes, per horizon
    bucket. Samples come from the last RECENT minutes of each history,
    are winsorized at +-WINSOR, and each market's own quantiles are
    shrunk toward the pooled quantiles with weight SHRINK_K."""
    pooled = {H: [] for H in BUCKETS}
    own = []
    for prices, s in zip(histories, vols):
        per_bucket = {H: [] for H in BUCKETS}
        if s > 0:
            n = len(prices)
            for H in BUCKETS:
                if n <= H + 1:
                    continue
                denom = s * math.sqrt(H)
                t0 = max(H, n - RECENT) if RECENT else H
                for t in range(t0, n, 10):
                    u = (prices[t] - prices[t - H]) / denom
                    if WINSOR is not None:
                        u = max(-WINSOR, min(WINSOR, u))
                    per_bucket[H].append(u)
                    pooled[H].append(u)
        own.append(per_bucket)
    pooled_q = {}
    for H, us in pooled.items():
        if len(us) < 50:
            continue
        us.sort()
        pooled_q[H] = [quantile(us, tau) for tau in TAUS]
    out = []
    for per_bucket in own:
        tab = {}
        for H, us in per_bucket.items():
            if H not in pooled_q:
                continue
            if SHRINK_K and len(us) >= 50:
                us = sorted(us)
                qo = [quantile(us, tau) for tau in TAUS]
                cnt = float(len(us))
                w = cnt / (cnt + SHRINK_K)
                tab[H] = [(w * qo[j] + (1.0 - w) * pooled_q[H][j]) * CAL[j]
                          for j in range(5)]
            else:
                tab[H] = [pooled_q[H][j] * CAL[j] for j in range(5)]
        out.append(tab)
    return out


def forecast_market(prices, s, qtab, horizon, clamp=True):
    x = prices[-1]
    s *= SCALE
    Hs = sorted(qtab.keys())
    rows = []
    for h in range(1, horizon + 1):
        if h <= Hs[0]:
            m = qtab[Hs[0]]
        elif h >= Hs[-1]:
            m = qtab[Hs[-1]]
        else:
            m = None
            for i in range(len(Hs) - 1):
                if Hs[i] <= h <= Hs[i + 1]:
                    f = (math.log(h) - math.log(Hs[i])) / (math.log(Hs[i + 1]) - math.log(Hs[i]))
                    m = [qtab[Hs[i]][j] + (qtab[Hs[i + 1]][j] - qtab[Hs[i]][j]) * f for j in range(5)]
                    break
        sh = s * math.sqrt(h)
        if clamp:
            rows.append([clamp6(x + m[j] * sh) for j in range(5)])
        else:
            rows.append([x + m[j] * sh for j in range(5)])
    return rows


def logit(p):
    p = min(1.0 - EPS, max(EPS, p))
    return math.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def ensemble_rows(prices, s_abs, s_log, qtab_abs, qtab_log, horizon):
    """Average of the absolute-space and logit-space forecasts."""
    ra = forecast_market(prices, s_abs, qtab_abs, horizon)
    tp = [logit(p) for p in prices]
    rl = forecast_market(tp, s_log, qtab_log, horizon, clamp=False)
    rows = []
    for k in range(horizon):
        row = [clamp6((ra[k][j] + sigmoid(rl[k][j])) / 2.0) for j in range(5)]
        rows.append(row)
    return rows


def baseline_rows(x, horizon):
    rows = []
    for h in range(1, horizon + 1):
        sh = 0.0025 * math.sqrt(h)
        rows.append([clamp6(x + z * sh) for z in ZNORM])
    return rows


def ensemble_pass(histories, halflife, horizon):
    """One full ensemble pass at a given vol halflife. Returns a list of
    240x5 row-lists aligned with histories, or None if no quantile table
    could be built in either space."""
    vols = [volatility(p, halflife) for p in histories]
    pos = sorted(v for v in vols if v > 0)
    med = pos[len(pos) // 2] if pos else 0.002
    vols = [v if v > 0 else med for v in vols]
    qtab = std_quantiles(histories, vols)
    thist = [[logit(p) for p in h] for h in histories]
    vols_log = [volatility(t, halflife) for t in thist]
    pos_log = sorted(v for v in vols_log if v > 0)
    med_log = pos_log[len(pos_log) // 2] if pos_log else 0.02
    vols_log = [v if v > 0 else med_log for v in vols_log]
    qtab_log = std_quantiles(thist, vols_log)
    if not any(qtab) and not any(qtab_log):
        return None
    out = []
    for mi, prices in enumerate(histories):
        qt = qtab[mi] if qtab else {}
        ql = qtab_log[mi] if qtab_log else {}
        if qt and ql:
            rows = ensemble_rows(prices, vols[mi], vols_log[mi], qt, ql, horizon)
        elif qt:
            rows = forecast_market(prices, vols[mi], qt, horizon)
        else:
            rl = forecast_market(thist[mi], vols_log[mi], ql, horizon, clamp=False)
            rows = [[clamp6(sigmoid(v)) for v in row] for row in rl]
        for row in rows:
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
        out.append(rows)
    return out


def hbucket(h):
    if h <= 60:
        return 0
    if h <= 120:
        return 1
    if h <= 180:
        return 2
    return 3


def run_case(case):
    horizon = int(case.get('horizon_minutes', 240))
    markets = case['markets']
    histories = [[float(v) for v in m['prices']] for m in markets]
    main_rows = ensemble_pass(histories, HALFLIFE, horizon)
    if main_rows is None:
        main_rows = [baseline_rows(p[-1], horizon) for p in histories]
    else:
        fast_rows = ensemble_pass(histories, HL_FAST, horizon)
        if fast_rows is not None:
            for mi in range(len(histories)):
                for k in range(horizon):
                    main_rows[mi][k] = [(1.0 - W_FAST) * main_rows[mi][k][j]
                                        + W_FAST * fast_rows[mi][k][j] for j in range(5)]
    forecasts = []
    for mi, m in enumerate(markets):
        prices = histories[mi]
        x = prices[-1]
        rows = main_rows[mi]
        # blend toward house baseline rows (spread floor)
        for k in range(horizon):
            sh = 0.0025 * math.sqrt(k + 1)
            for j in range(5):
                bv = min(1.0, max(0.0, x + ZNORM[j] * sh))
                rows[k][j] = (1.0 - BLEND_BASE) * rows[k][j] + BLEND_BASE * bv
        # horizon-bucket tail shaping around the median
        for k in range(horizon):
            gb = SHAPE_G[hbucket(k + 1)]
            med = rows[k][2]
            rows[k] = [med + (rows[k][j] - med) * gb[j] for j in range(5)]
        # recent-volatility spread boost for this market
        if len(prices) > RVOL_WIN + 2:
            rec = sum(abs(prices[i] - prices[i - 1])
                      for i in range(len(prices) - RVOL_WIN, len(prices))) / RVOL_WIN
            lng = volatility(prices, HALFLIFE)
            if lng > 0:
                boost = min(RVOL_CAP, 1.0 + RVOL_BETA * max(0.0, rec / lng - 1.0))
                if boost > 1.0:
                    for k in range(horizon):
                        med = rows[k][2]
                        rows[k] = [med + (rows[k][j] - med) * boost for j in range(5)]
        # final clamp + monotonicity repair
        for k in range(horizon):
            r = [min(1.0, max(0.0, round(v, 6))) for v in rows[k]]
            for j in range(1, 5):
                if r[j] < r[j - 1]:
                    r[j] = r[j - 1]
            rows[k] = r
        forecasts.append({'id': m['id'], 'q': rows})
    return {'forecasts': forecasts}


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith('.in'):
            continue
        with open(os.path.join(in_dir, fname), 'r', encoding='utf-8') as f:
            case = json.load(f)
        result = run_case(case)
        out_path = os.path.join(out_dir, fname[:-3] + '.out')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    main()
