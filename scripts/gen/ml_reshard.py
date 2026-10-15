#!/usr/bin/env python3
"""Split ml checks files so each stays inside one file's comptime fuel.

The correctly rounded math functions cost far more fuel than the fdlibm
kernels did, so a generated file may no longer pass in one budget. For
each tests/ml/<lib>_checks*.rce named, this tests the file as check.sh
does (package mode) and, while it runs out of fuel, halves its check
lines (the second half into <stem>_s<n>.rce) (the helper head is copied into each).
A single check that alone exhausts the fuel is reported, not split.
Usage: RCE=rce python3 scripts/gen/ml_reshard.py tests/ml/ad_checks_0.rce ..."""
import os, subprocess, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uses
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
RCE = os.environ.get('RCE', 'rce')
BASE = ['']
def run(lib, text):
    d = tempfile.mkdtemp()
    try:
        for u in uses.deps(lib):
            shutil.copy(os.path.join(ROOT, 'lib', u + '.rce'), d)
        open(os.path.join(d, os.path.basename(lib)), 'w').write(open(lib).read() + '\n' + text)
        r = subprocess.run([RCE, 'test', d], capture_output=True, text=True)
        return r.stdout + r.stderr
    finally:
        shutil.rmtree(d)
def split(lines):
    k = next(i for i, l in enumerate(lines) if l.startswith('check '))
    return lines[:k], lines[k:]
def go(lib, path, out):
    lines = open(path).read().rstrip('\n').split('\n')
    head, checks = split(lines)
    res = run(lib, '\n'.join(lines) + '\n')
    if 'fuel exhausted' not in res:
        if ' 0 failed' not in res:
            print('FAIL (not fuel)', path, res[-400:]); sys.exit(1)
        out.append((path, lines)); return
    if len(checks) == 1:
        print('ALONE OUT OF FUEL', path, checks[0][:120]); sys.exit(1)
    # the first half stays in path (check.sh and the gates name the base
    # files), the second goes to a fresh <base>_s<n>.rce
    h = len(checks) // 2
    n = 1
    while os.path.exists('%s_s%d.rce' % (BASE[0], n)): n += 1
    p = '%s_s%d.rce' % (BASE[0], n)
    open(p, 'w').write('\n'.join(head + checks[h:]) + '\n')
    open(path, 'w').write('\n'.join(head + checks[:h]) + '\n')
    go(lib, path, out)
    go(lib, p, out)
for path in sys.argv[1:]:
    name = os.path.basename(path).split('_checks')[0]
    lib = os.path.join(ROOT, 'lib', 'ml', name + '.rce')
    out = []
    BASE[:] = [path[:-4]]
    go(lib, path, out)
    print(path, '->', len(out), 'files', sum(len(split(l)[1]) for _, l in out), 'checks')
