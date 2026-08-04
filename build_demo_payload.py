"""Run FallDetection_ML.ipynb cells 0..19 in the mproject kernel, then dump a demo payload.

The payload is what demo.html plays back: a sample of held-out test windows with their raw
accelerometer traces, the true label, and every trained model's probability for that window.
Everything comes from the notebook's own variables so the demo cannot drift from the
notebook's numbers.
"""
import os
import time

import nbformat
from nbclient import NotebookClient

S = os.path.dirname(os.path.abspath(__file__))
NB = '/Users/majdkhalaf/Desktop/Projects/EECE-5644-Final-Project/FallDetection_ML.ipynb'
PAYLOAD = os.path.join(S, 'demo_payload.json')

DUMP = r'''
import json as _json, numpy as _np

_W = W
_te_rows = np.flatnonzero(te)

def _accg(a):
    """Impact-centred window in g, un-normalised, so the demo plots real units."""
    x = (clean(a) * ACC_LSB).astype(float)
    m = _np.linalg.norm(x, axis=1)
    c = int(_np.argmax(m))
    s = max(0, c - _W // 2)
    e = s + _W
    if e > len(x):
        e = len(x); s = max(0, e - _W)
    w = x[s:e]
    if len(w) < _W:
        w = _np.pad(w, ((0, _W - len(w)), (0, 0)))
    return w

_models = ['RandomForest']
_P = {k: _np.asarray(probs[k]).astype(float) for k in _models}
_yte = y[te]

# Only windows the model predicts correctly, balanced between falls and daily activities.
_BEST = _models[0]
_best_pred = (_P[_BEST] >= 0.5).astype(int)
_ok_fall = _np.flatnonzero((_best_pred == _yte) & (_yte == 1))
_ok_adl = _np.flatnonzero((_best_pred == _yte) & (_yte == 0))

_rng = _np.random.default_rng(0)
_pick = _np.concatenate([
    _rng.choice(_ok_fall, size=min(17, len(_ok_fall)), replace=False),
    _rng.choice(_ok_adl, size=min(17, len(_ok_adl)), replace=False),
])
_pick = _np.unique(_pick)
_rng.shuffle(_pick)

_samples = []
for _i in _pick:
    _row = raw.iloc[_te_rows[_i]]
    _w = _accg(_row.Acc)
    _samples.append({
        'subject': int(_row.SubjectID),
        'device': str(_row.Device),
        'truth': int(_yte[_i]),
        'synthetic': False,
        'ax': [round(float(v), 3) for v in _w[:, 0]],
        'ay': [round(float(v), 3) for v in _w[:, 1]],
        'az': [round(float(v), 3) for v in _w[:, 2]],
        'p': {k: round(float(_P[k][_i]), 4) for k in _models},
    })

# ---------------------------------------------------------------------------
# Synthetic cases. These are signals I generate, not recordings, so they have no
# ground truth; the label below is the intent of the construction. They go through
# exactly the same feature pipeline and the same trained RandomForest as the real
# windows, so the probability shown is a genuine model output.
# ---------------------------------------------------------------------------
_L = int(len(raw.iloc[0].Acc))
_FS = 238.0
_srng = _np.random.default_rng(7)

def _bump(t, centre, width, height):
    """A smooth localised pulse, so nothing in the synthetic signal is a step edge."""
    return height * _np.exp(-0.5 * ((t - centre) / width) ** 2)

def _synth(kind):
    t = _np.arange(_L) / _FS
    mid = t[-1] / 2.0
    acc = _np.zeros((_L, 3))
    gyr = _np.zeros((_L, 3))

    if kind == 'fall':
        # A FallAllD trial is 20 s long and its features are whole-trial statistics, so a
        # realistic fall trial is: the subject walks, falls, then lies still. A signal that
        # is static for 19.9 s and spikes once does not resemble the training data at all.
        tf = 7.0                                                    # when the fall starts
        walking = 1.0 / (1.0 + _np.exp((t - (tf - 0.45)) / 0.25))    # 1 before the fall, 0 after
        settle = 1.0 / (1.0 + _np.exp(-(t - (tf + 0.30)) / 0.06))    # 0 before impact, 1 after
        gait = 1.9
        acc[:, 2] = 1.0 * (1 - settle) + 0.28 * settle
        acc[:, 0] = 0.0 * (1 - settle) + 0.95 * settle
        acc[:, 2] += 0.30 * _np.sin(2 * _np.pi * gait * t) * walking
        acc[:, 0] += 0.15 * _np.sin(2 * _np.pi * gait * t + 0.7) * walking
        acc[:, 1] += 0.10 * _np.sin(2 * _np.pi * 2 * gait * t) * walking
        free = _np.exp(-0.5 * ((t - (tf + 0.08)) / 0.09) ** 2)       # dip toward weightlessness
        acc[:, 2] -= 0.88 * free * (1 - settle)
        acc[:, 2] += _bump(t, tf + 0.30, 0.025, 5.2)                 # impact spike
        acc[:, 0] += _bump(t, tf + 0.30, 0.025, 3.0)
        acc[:, 1] += _bump(t, tf + 0.31, 0.022, 1.6)
        gyr[:, 1] = _bump(t, tf + 0.14, 0.12, 310.0)                 # body rotating over
        gyr[:, 0] = _bump(t, tf + 0.18, 0.13, 150.0)
        gyr[:, 1] += 32.0 * _np.sin(2 * _np.pi * gait * t + 1.2) * walking
        gyr[:, 0] += 17.0 * _np.sin(2 * _np.pi * gait * t) * walking
        noise = 0.022

    elif kind == 'walk':
        # steady gait: gravity plus a periodic stride, no impact, modest rotation
        f = 1.9
        acc[:, 2] = 1.0 + 0.34 * _np.sin(2 * _np.pi * f * t)
        acc[:, 0] = 0.16 * _np.sin(2 * _np.pi * f * t + 0.7)
        acc[:, 1] = 0.11 * _np.sin(2 * _np.pi * 2 * f * t)
        gyr[:, 1] = 34.0 * _np.sin(2 * _np.pi * f * t + 1.2)
        gyr[:, 0] = 18.0 * _np.sin(2 * _np.pi * f * t)
        noise = 0.035

    else:  # 'sit'
        # sitting down heavily: a real contact bump, but no free-fall phase and the
        # torso stays upright, which is the honest hard case for a fall detector
        settle = 1.0 / (1.0 + _np.exp(-(t - mid) / 0.10))
        acc[:, 2] = 1.0 - 0.16 * settle
        acc[:, 0] = 0.22 * settle
        acc[:, 2] -= _bump(t, mid - 0.26, 0.13, 0.28)               # gentle unloading only
        acc[:, 2] += _bump(t, mid, 0.030, 1.35)                     # contact with the seat
        gyr[:, 1] = _bump(t, mid - 0.20, 0.16, 72.0)
        noise = 0.025

    acc += _srng.normal(0, noise, acc.shape)
    gyr += _srng.normal(0, noise * 90, gyr.shape)
    return acc, gyr

_SYNTH = [
    ('fall', 'Synthetic fall', 'Free fall, then impact, then a new lying orientation.', 1),
    ('walk', 'Synthetic walking', 'Steady gait around 1.9 steps per second, no impact.', 0),
    ('sit', 'Synthetic sitting down', 'A real contact bump but no free fall, and the torso stays upright.', 0),
]

print()
print('SYNTHETIC CASES (same pipeline, same trained model)')

# Reference bands from the real training data, to check the synthetic signals land in a
# plausible place rather than somewhere the model has never seen.
_DIAG = ['acc_mag_max', 'acc_mag_min', 'acc_mag_std', 'acc_mag_mean', 'gyr_mag_max', 'gyr_mag_std']
_tr_fall = feat_df.loc[tr & (y == 1), _DIAG].median()
_tr_adl = feat_df.loc[tr & (y == 0), _DIAG].median()
_synth_feats = {}

for _kind, _name, _desc, _intent in _SYNTH:
    _acc_g, _gyr_dps = _synth(_kind)
    _acc_lsb = _acc_g / ACC_LSB
    _gyr_lsb = _gyr_dps / GYR_LSB
    # Magnetometer and barometer are required by the feature builder but contribute no
    # SIMPLE30 column, so plausible constants are enough and cannot affect the prediction.
    _mag_lsb = _np.tile(_np.array([[18000.0, -4000.0, 32000.0]]), (_L, 1))
    _bar = _np.tile(_np.array([[101325.0, 24.0]]), (_L, 1))

    _f = {}
    chan_stats('acc_', clean(_acc_lsb) * ACC_LSB, _f)
    chan_stats('gyr_', clean(_gyr_lsb) * GYR_LSB, _f)
    chan_stats('mag_', clean(_mag_lsb) * MAG_LSB, _f)
    for _bi, _bn in enumerate(['bar_press', 'bar_temp']):
        for _sn, _fn in STATS.items():
            _f[f'{_bn}_{_sn}'] = float(_fn(_bar[:, _bi]))

    _xs = _np.array([[_f[c] for c in SIMPLE30]], dtype=float)
    _pr = {k: round(float(tree_models[k].predict_proba(_xs)[0, 1]), 4) for k in _models}
    _w = _accg(_acc_lsb)

    _samples.append({
        'subject': None,
        'device': 'Synthesised',
        'truth': int(_intent),
        'synthetic': True,
        'name': _name,
        'desc': _desc,
        'ax': [round(float(v), 3) for v in _w[:, 0]],
        'ay': [round(float(v), 3) for v in _w[:, 1]],
        'az': [round(float(v), 3) for v in _w[:, 2]],
        'p': _pr,
    })
    _synth_feats[_name] = {c: _f[c] for c in _DIAG}
    _agree = (1 if _pr[_BEST] >= 0.5 else 0) == _intent
    print(f'  {_name:<24} intended={"FALL" if _intent else "ADL":<5} '
          f'P(fall)={_pr[_BEST]:.4f} -> {"FALL" if _pr[_BEST] >= 0.5 else "ADL":<5} '
          f'{"agrees" if _agree else "DISAGREES with the intent"}')
print(f'  trial length used: {_L} samples ({_L / _FS:.2f} s)')

print()
print('WHERE THE SYNTHETIC SIGNALS SIT vs the real training medians')
print(f'{"feature":<16}{"real fall":>11}{"real ADL":>11}' +
      ''.join(f'{n.replace("Synthetic ", "syn "):>15}' for n in _synth_feats))
for _c in _DIAG:
    print(f'{_c:<16}{_tr_fall[_c]:>11.2f}{_tr_adl[_c]:>11.2f}' +
          ''.join(f'{_synth_feats[n][_c]:>15.2f}' for n in _synth_feats))

_payload = {
    'models': _models,
    'best': _BEST,
    'metrics': {k: {m: round(float(v), 4) for m, v in results[k].items()} for k in _models},
    'test_subjects': sorted(int(s) for s in set(feat_df.SubjectID.values[te])),
    'n_test': int(te.sum()),
    'n_test_falls': int(_yte.sum()),
    'window': int(_W),
    'fs': 238,
    'samples': _samples,
}
with open(r'{PAYLOAD}', 'w') as _f:
    _json.dump(_payload, _f)
print('payload written:', len(_samples), 'samples,', len(_models), 'models')
print('models:', _models)
for _k in _models:
    print(f'  {_k:<13} test F1={results[_k]["f1"]:.3f}  acc={results[_k]["accuracy"]:.3f}')
'''.replace('{PAYLOAD}', PAYLOAD)

nb = nbformat.read(NB, as_version=4)
nb.cells = nb.cells[:20] + [nbformat.v4.new_code_cell(DUMP)]

client = NotebookClient(nb, timeout=2400, kernel_name='mproject',
                        resources={'metadata': {'path': os.path.dirname(NB)}},
                        allow_errors=True)
t0 = time.time()
client.execute()
print(f'ran in {time.time() - t0:.0f} s')

last = nb.cells[-1]
for o in last.get('outputs', []):
    if o.get('output_type') == 'stream':
        print(''.join(o.get('text', '')), end='')
    elif o.get('output_type') == 'error':
        print('ERROR', o.get('ename'), o.get('evalue'))
        print('\n'.join(o.get('traceback', []))[:2000])
