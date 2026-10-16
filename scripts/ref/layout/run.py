#!/usr/bin/env python3
# run.py CAT [limit] [-v]: run vectors of one category via rce run, in chunks, in parallel
import sys, subprocess, os, concurrent.futures as cf
sys.path.insert(0, '/home/box/work/stdlib/wt-layout/scripts/gen')
from rcelit import lit
RCE = '/tmp/rce-ide-0.1.3-x86_64-unknown-linux-gnu/rce'
cat = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10**9
verbose = '-v' in sys.argv
filt = os.environ.get('F', '')
mod = open('/tmp/l50/m.rce').read()
rows = [l.rstrip('\n').split('\t') for l in open('/home/box/work/stdlib/wt-layout/vectors/ui/layout/%s.tsv' % cat)]
rows = [r for r in rows if filt in r[0]][:lim]
def chunk(k, rs):
    body = mod + '\nfn main()\n'
    for r in rs:
        body += '    emit lay_ok(%s, %s, %s, %s)\n' % (lit(r[1]), lit(r[2]), lit(r[3]), lit(r[4]))
        body += '    emit lay(%s, %s, %s)\n' % (lit(r[1]), lit(r[2]), lit(r[3]))
    f = '/tmp/l50/run_%s_%d.rce' % (cat, k); open(f, 'w').write(body)
    p = subprocess.run([RCE, 'run', f], capture_output=True, text=True, timeout=600)
    outs = [l[5:] for l in p.stdout.splitlines() if l.startswith('emit ')]
    res = []
    for j, r in enumerate(rs):
        if 2 * j + 1 < len(outs):
            res.append((r[0], outs[2 * j] == 'true', outs[2 * j + 1], r[4]))
        else:
            res.append((r[0], False, 'ERR ' + (p.stdout + p.stderr)[-300:], r[4]))
    return res
N = 20
chunks = [rows[i:i + N] for i in range(0, len(rows), N)]
with cf.ThreadPoolExecutor(int(os.environ.get('P', '6'))) as ex:
    allr = [x for rs in ex.map(lambda a: chunk(*a), enumerate(chunks)) for x in rs]
ok = sum(1 for r in allr if r[1])
for name, good, got, exp in allr:
    if not good:
        print('FAIL', name)
        if verbose:
            print('  exp', exp[:400]); print('  got', got[:400])
print('%s: %d/%d' % (cat, ok, len(allr)))
