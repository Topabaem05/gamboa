"""Fresh-noise stress test on held-out vessels.
# Run from the repo root: python experiments/stress.py --fold 0   (fold bundles cached in $STRESS_DIR)

For each fold: train on the other folds (cached to disk), then evaluate the held-out vessels
(a) as given, (b) with freshly generated typos for Latin names (same edit-count / edit-type
distribution as train), (c) with tonnage resampled from the train noise distribution,
(d) with flag dropped. Prints accuracy per scenario and every miss.
"""
import sys, os, time, random, argparse, joblib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import vessel_matcher as vm
from vessel_matcher import *
from vessel_matcher import _tonnage_noise_pairs

ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, default=0); ap.add_argument('--reps', type=int, default=3); ap.add_argument('--tag', default='v6'); ap.add_argument('--hw', type=float, default=1.0); ap.add_argument('--only', default='')
args = ap.parse_args()
S = os.environ.get('STRESS_DIR', 'experiments/out')
tr = pd.read_csv('data/train.csv'); cat = pd.read_csv('data/vessel_catalog.csv')
catn = cat.set_index('vessel_id'); idx = CatalogIndex(cat)
a, b = list(GroupKFold(5).split(tr, groups=tr.target_vessel_id))[args.fold]
trn, va = tr.iloc[a], tr.iloc[b].copy()
path = f'{S}/fold{args.fold}_bundle.pkl' if args.tag == 'v6' else f'{S}/fold{args.fold}_bundle_{args.tag}.pkl'
if os.path.exists(path):
    bundle = joblib.load(path)
else:
    t = time.time(); bundle = fit(trn, cat, index=idx, hangul_weight=args.hw); joblib.dump(bundle, path, compress=3)
    print(f'trained fold {args.fold} in {time.time()-t:.0f}s', flush=True)

# empirical noise
m = tr.join(catn, on='target_vessel_id', rsuffix='_c')
rel = ((m.gross_tonnage - m.gross_tonnage_c) / m.gross_tonnage_c).dropna().to_numpy()
EDITS = [0] * 6892 + [1] * 2785 + [2] * 1271 + [3] * 79

def typo(name: str, rng: random.Random) -> str:
    s = name.upper()
    for _ in range(rng.choice(EDITS)):
        pos = [i for i, ch in enumerate(s) if ch.isalnum()]
        if len(pos) < 3:
            break
        i = rng.choice(pos); op = rng.random()
        if op < 0.47:   s = s[:i] + s[i + 1:]                       # delete
        elif op < 0.89: s = s[:i] + s[i] + s[i:]                    # duplicate
        elif i + 1 < len(s): s = s[:i] + s[i + 1] + s[i] + s[i + 2:]  # transpose
    return s

def run(df, label):
    ids, _ = predict_ids(bundle, df, idx, use_answers=False, n_jobs=int(os.environ.get('MATCHER_JOBS', 2)))
    ok = np.asarray(ids) == df.target_vessel_id.to_numpy(); h = df.vessel_name.map(is_hangul).to_numpy()
    w = 0.7 * ok[h].mean() + 0.3 * ok[~h].mean()
    print(f'[fold{args.fold} {label}] hangul={ok[h].mean():.4f} ({(~ok[h]).sum()} miss) non={ok[~h].mean():.4f} ({(~ok[~h]).sum()} miss) weighted={w:.4f}', flush=True)
    for r, p in zip(df[~ok].to_dict('records'), np.asarray(ids)[~ok]):
        print(f'    miss {label}: q={r["vessel_name"]!r} gt={r["gross_tonnage"]} intl={r["international_gross_tonnage"]} flag={r["flag_name"]} type={r["vessel_type"]} '
              f'| true={catn.loc[r["target_vessel_id"]].vessel_name!r} gt={catn.loc[r["target_vessel_id"]].gross_tonnage} {catn.loc[r["target_vessel_id"]].flag_name} '
              f'| pred={catn.loc[p].vessel_name!r} gt={catn.loc[p].gross_tonnage} {catn.loc[p].flag_name}', flush=True)
    return w

run(va, 'asis')
for rep in range(args.reps if args.only != 'asis' else 0):
    rng = random.Random(1000 * args.fold + rep); nr = np.random.default_rng(1000 * args.fold + rep)
    df = va.copy()
    lat = ~df.vessel_name.map(is_hangul)
    names = {}
    for vid in df.loc[lat, 'target_vessel_id'].unique():
        names[vid] = typo(catn.loc[vid].vessel_name, rng)          # one query name per vessel, as in train
    df.loc[lat, 'vessel_name'] = df.loc[lat, 'target_vessel_id'].map(names)
    if args.only != 'gt': run(df, f'freshtypo{rep}')
    d2 = va.copy(); cgt = d2.target_vessel_id.map(catn.gross_tonnage).astype(float)
    pairs = _tonnage_noise_pairs(tr, cat); draw = pairs[nr.integers(0, len(pairs), size=len(d2))]
    has = (cgt > 0).to_numpy()
    d2.loc[has, 'gross_tonnage'] = np.round(cgt[has].to_numpy() * (1 + draw[has, 0]))
    d2.loc[has, 'international_gross_tonnage'] = np.round(cgt[has].to_numpy() * (1 + draw[has, 1]))
    run(d2, f'freshgt{rep}')
d3 = va.copy(); d3['flag_name'] = np.nan; run(d3, 'noflag')
d4 = va.copy(); d4['gross_tonnage'] = np.nan; d4['international_gross_tonnage'] = np.nan; run(d4, 'notonnage')
