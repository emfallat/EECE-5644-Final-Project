"""Regenerate demo.html from demo_payload.json.

The payload is produced by running FallDetection_ML.ipynb up to the tree-ensemble cell and
dumping the test-window traces plus every model's probability. This script only does the
inlining, so the demo is a single file with no network dependency: open demo.html directly.

usage: python build_demo.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, 'demo_template.html')
PAYLOAD = os.path.join(HERE, 'demo_payload.json')
OUT = os.path.join(HERE, 'demo.html')

tpl = open(TEMPLATE, encoding='utf-8').read()
payload = open(PAYLOAD, encoding='utf-8').read()

# The payload is inlined into a <script type="application/json"> block, so it must not be
# able to close that block early.
if '</script' in payload.lower():
    raise SystemExit('payload contains a closing script tag')
if '__PAYLOAD__' not in tpl:
    raise SystemExit('template is missing the __PAYLOAD__ placeholder')

html = tpl.replace('__PAYLOAD__', payload)
open(OUT, 'w', encoding='utf-8').write(html)

d = json.loads(payload)
external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
print(f'wrote {OUT}  ({len(html) / 1e6:.2f} MB)')
print(f'  {len(d["samples"])} windows, {len(d["models"])} models, best = {d["best"]}')
print(f'  external references: {external or "none"}')
