import json, math, os, sys

# Forecaster for Polymarket "Next CEO of NVIDIA?" minute prices.
# Base: ensemble of absolute-space and logit-space forecasts. Per-market
# volatility = EWMA (halflife 720 min) of mean absolute 1-minute changes;
# the shape of future moves is the pooled empirical quantiles of
# standardised h-minute changes from the case itself (per horizon bucket),
# outer multipliers shrunk (CAL) and spread scaled by 0.7.
# Round-2 additions, all operating on the ensemble output:
#  - a second ensemble pass with a faster vol halflife (300 min), blended
#    in at weight 0.25;
#  - 10% blend toward the house baseline rows (a spread floor);
#  - horizon-bucket tail shaping: at longer horizons the inner/outer
#    quantiles are pulled toward the median (the raw ensemble over-covers
#    there: coverage90 0.93-0.95 vs 0.90 nominal);
#  - per-market recent-volatility boost: markets whose last 90 minutes were
#    much more active than their EWMA level get a wider spread (news moves
#    cluster), capped at 2x.
# Final entry for the competition: the round-2 model above replaces the
# round-1 ensemble (dev loss 0.275417) after beating it on every local
# validation set (practice, extra, and the equal-weighted mean of both).

BUCKETS = (10, 20, 40, 80, 160, 240)
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
CAL = (0.55, 0.95, 1.0, 0.95, 0.55)
SCALE = 0.7
HALFLIFE = 720.0
HL_FAST = 300.0
W_FAST = 0.25
BLEND_BASE = 0.1
EPS = 0.0005
ZNORM = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
# per horizon bucket (1-60, 61-120, 121-180, 181-240): multipliers on
# (q_j - median) for j = 0..4
SHAPE_G = ((1.0, 1.0, 1.0, 1.0, 1.0),
           (0.85, 0.94, 1.0, 0.94, 0.85),
           (0.78, 0.92, 1.0, 0.92, 0.78),
           (0.81, 0.93, 1.0, 0.93, 0.81))
RVOL_BETA = 0.3
RVOL_WIN = 90
RVOL_CAP = 2.0


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
    """Pooled empirical quantiles of standardised changes, per horizon bucket."""
    per_bucket = {H: [] for H in BUCKETS}
    for prices, s in zip(histories, vols):
        if s <= 0:
            continue
        n = len(prices)
        for H in BUCKETS:
            if n <= H + 1:
                continue
            denom = s * math.sqrt(H)
            for t in range(H, n, 10):
                per_bucket[H].append((prices[t] - prices[t - H]) / denom)
    out = {}
    for H, us in per_bucket.items():
        if len(us) < 50:
            continue
        us.sort()
        out[H] = [quantile(us, tau) * CAL[j] for j, tau in enumerate(TAUS)]
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
    if not qtab and not qtab_log:
        return None
    out = []
    for mi, prices in enumerate(histories):
        if qtab and qtab_log:
            rows = ensemble_rows(prices, vols[mi], vols_log[mi], qtab, qtab_log, horizon)
        elif qtab:
            rows = forecast_market(prices, vols[mi], qtab, horizon)
        else:
            rl = forecast_market(thist[mi], vols_log[mi], qtab_log, horizon, clamp=False)
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
