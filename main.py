import json, math, os, sys

# Forecaster for Polymarket "Next CEO of NVIDIA?" minute prices.
# Method: per-market volatility = EWMA (halflife 1440 min) of mean absolute
# 1-minute changes. The shape of future moves is taken from the case itself:
# h-minute changes of every market in the case, standardised by that market's
# volatility times sqrt(h), are pooled per horizon bucket and their empirical
# 5/25/50/75/95% quantiles become the forecast's quantile multipliers.
# Outer multipliers are shrunk by 0.85 and the whole spread scaled by 0.9
# (calibrated on the event's own history). Median = last price (persistence).

BUCKETS = (10, 20, 40, 80, 160, 240)
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
CAL = (0.55, 0.95, 1.0, 0.95, 0.55)
SCALE = 0.7
HALFLIFE = 720.0
EPS = 0.0005
ZNORM = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)


def clamp6(v):
    return round(min(1.0, max(0.0, v)), 6) + 0


def volatility(prices):
    """EWMA of |1-minute changes| with halflife HALFLIFE minutes."""
    n = len(prices)
    if n < 3:
        return 0.0
    lam = 0.5 ** (1.0 / HALFLIFE)
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


def run_case(case):
    horizon = int(case.get('horizon_minutes', 240))
    markets = case['markets']
    histories = [[float(v) for v in m['prices']] for m in markets]
    vols = [volatility(p) for p in histories]
    pos = sorted(v for v in vols if v > 0)
    med = pos[len(pos) // 2] if pos else 0.002
    vols = [v if v > 0 else med for v in vols]
    qtab = std_quantiles(histories, vols)
    thist = [[logit(p) for p in h] for h in histories]
    vols_log = [volatility(t) for t in thist]
    pos_log = sorted(v for v in vols_log if v > 0)
    med_log = pos_log[len(pos_log) // 2] if pos_log else 0.02
    vols_log = [v if v > 0 else med_log for v in vols_log]
    qtab_log = std_quantiles(thist, vols_log)
    forecasts = []
    for mi, (m, prices, s) in enumerate(zip(markets, histories, vols)):
        if qtab and qtab_log:
            rows = ensemble_rows(prices, s, vols_log[mi], qtab, qtab_log, horizon)
        elif qtab:
            rows = forecast_market(prices, s, qtab, horizon)
        elif qtab_log:
            rl = forecast_market(thist[mi], vols_log[mi], qtab_log, horizon, clamp=False)
            rows = [[clamp6(sigmoid(v)) for v in row] for row in rl]
        else:
            rows = baseline_rows(prices[-1], horizon)
        for row in rows:
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
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
