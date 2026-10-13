"""Hangul variant stress test: re-spell each held-out Hangul vessel name with 1-2 realistic
# Run from the repo root after experiments/stress.py has cached the fold bundle.
variations observed in train (져/저, 니아/니어, 썬/선, 갤/겔(ㅐ/ㅔ), 스피리트/스피릿, 랴/리야, 톈/텐,
샨/산, 호 suffix, spacing, word-final 버/바) and evaluate the fold model on them."""
import sys, os, random, argparse, joblib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from vessel_matcher import *

ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, default=0); ap.add_argument('--reps', type=int, default=4); ap.add_argument('--tag', default='v6')
args = ap.parse_args()
S = os.environ.get('STRESS_DIR', 'experiments/out')
tr = pd.read_csv('data/train.csv'); cat = pd.read_csv('data/vessel_catalog.csv')
catn = cat.set_index('vessel_id'); idx = CatalogIndex(cat)
a, b = list(GroupKFold(5).split(tr, groups=tr.target_vessel_id))[args.fold]
va = tr.iloc[b]; va = va[va.vessel_name.map(is_hangul)].copy()
bundle = joblib.load(f'{S}/fold{args.fold}_bundle.pkl' if args.tag == 'v6' else f'{S}/fold{args.fold}_bundle_{args.tag}.pkl')

def dec(ch):
    c = ord(ch) - 0xAC00
    return (c // 588, (c % 588) // 28, c % 28) if 0 <= c <= 11171 else None
def enc(i, m, f): return chr(0xAC00 + i * 588 + m * 28 + f)

MED_SWAP = {1: 5, 5: 1,      # ㅐ <-> ㅔ
            4: 0}            # ㅓ -> ㅏ (word-final: 하버 -> 하바)
GLIDE = {6: 4, 4: 6, 18: 18, 12: 8, 8: 12, 17: 13, 13: 17}   # ㅕ<->ㅓ, ㅛ<->ㅗ, ㅠ<->ㅜ (after ㅈ/ㅊ)
DOUBLE = {10: 9, 9: 10, 4: 3, 3: 4}                         # ㅆ<->ㅅ, ㄸ<->ㄷ

def variants(word: str, rng: random.Random) -> str:
    chars = list(word)
    ops = []
    for k, ch in enumerate(chars):
        d = dec(ch)
        if not d: continue
        i, mm, f = d
        if mm in (1, 5): ops.append(('ae_e', k))
        if mm == 4 and f == 0 and k == len(chars) - 1: ops.append(('eo_a', k))
        if i in (12, 14) and mm in GLIDE: ops.append(('glide', k))
        if i in DOUBLE: ops.append(('double', k))
        if ch == '트' and k == len(chars) - 1 and k > 0 and dec(chars[k - 1]) and dec(chars[k - 1])[2] == 0: ops.append(('teu_final', k))
        if f == 19 and k == len(chars) - 1: ops.append(('final_s_teu', k))
        if ch in ('아', '어') and k == len(chars) - 1 and k > 0 and chars[k - 1] in '니리': ops.append(('nia', k))
        if i in (5, 9, 2) and mm in (2, 6) : ops.append(('ya', k))
    if not ops: return word
    op, k = rng.choice(ops); i, mm, f = dec(chars[k])
    if op == 'ae_e': chars[k] = enc(i, MED_SWAP[mm], f)
    elif op == 'eo_a': chars[k] = enc(i, 0, f)
    elif op == 'glide': chars[k] = enc(i, GLIDE[mm], f)
    elif op == 'double': chars[k] = enc(DOUBLE[i], mm, f)
    elif op == 'teu_final':
        pi, pm, _ = dec(chars[k - 1]); chars[k - 1] = enc(pi, pm, 19); del chars[k]
    elif op == 'final_s_teu': chars[k] = enc(i, mm, 0); chars.insert(k + 1, '트')
    elif op == 'nia': chars[k] = '어' if chars[k] == '아' else '아'
    elif op == 'ya': chars[k] = enc(i, 0 if mm == 2 else 4, f)          # 샨->산, 셴->선
    return ''.join(chars)

def respell(name: str, rng: random.Random) -> str:
    toks = name.split()
    for _ in range(rng.choice([1, 1, 2])):
        r = rng.random()
        if r < 0.15 and len(toks) >= 2:                      # join two tokens (spacing)
            j = rng.randrange(len(toks) - 1); toks[j:j + 2] = [toks[j] + toks[j + 1]]
        elif r < 0.25:                                       # add / drop 호 suffix
            toks[-1] = toks[-1][:-1] if toks[-1].endswith('호') and len(toks[-1]) > 2 else toks[-1] + '호'
        else:
            j = rng.randrange(len(toks)); toks[j] = variants(toks[j], rng)
    return ' '.join(toks)

def run(df, label):
    ids, _ = predict_ids(bundle, df, idx, use_answers=False, n_jobs=int(os.environ.get('MATCHER_JOBS', 2)))
    ok = np.asarray(ids) == df.target_vessel_id.to_numpy()
    print(f'[fold{args.fold} {label}] hangul rows={len(df)} acc={ok.mean():.4f} misses={(~ok).sum()}', flush=True)
    for r, p in zip(df[~ok].to_dict('records'), np.asarray(ids)[~ok]):
        t = catn.loc[r['target_vessel_id']]; q = catn.loc[p]
        print(f'    miss {label}: q={r["vessel_name"]!r} gt={r["gross_tonnage"]} flag={r["flag_name"]} | true={t.vessel_name!r} gt={t.gross_tonnage} {t.flag_name} | pred={q.vessel_name!r} gt={q.gross_tonnage} {q.flag_name}', flush=True)
    return ok.mean()

run(va, 'asis')
accs = []
for rep in range(args.reps):
    rng = random.Random(9_000_077 + 1000 * args.fold + rep)   # seeds disjoint from training augmentation; df = va.copy()
    names = {v: respell(n, rng) for v, n in df.groupby('target_vessel_id').vessel_name.first().items()}
    changed = sum(names[v] != n for v, n in df.groupby('target_vessel_id').vessel_name.first().items())
    df['vessel_name'] = df.target_vessel_id.map(names)
    print(f'rep{rep}: {changed} of {len(names)} vessel names re-spelled; e.g. {list(names.values())[:6]}', flush=True)
    accs.append(run(df, f'respell{rep}'))
print(f'[fold{args.fold}] mean respelled hangul acc = {np.mean(accs):.4f}')
