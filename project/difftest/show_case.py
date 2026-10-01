#!/usr/bin/env python3
"""Pretty-prints one recorded difftest case: show_case.py <category> <line number>."""
import json
import os
import sys

cat, no = sys.argv[1], int(sys.argv[2])
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_difftest as g  # noqa: E402
line = open(f'src/test/resources/difftest/{cat}.jsonl').read().splitlines()[no-1]
c = json.loads(line)
for k, v in c.items():
    if isinstance(v, dict) and 'd' in v:
        a = g.dec_array(v)
        print(k, '=', repr(a), 'view steps', v.get('w'))
    elif k == 'arrs':
        for e in v: print('arr', repr(g.dec_array(e)), e.get('w'))
    elif k in ('idx',):
        print('idx =', g.dec_index(v))
    elif k == 'r':
        print('r =', str(v)[:500])
    else:
        print(k, '=', str(v)[:2000])
