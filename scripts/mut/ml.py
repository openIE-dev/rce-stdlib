# Mutation test for lib/ml (ad, tensor, prob): apply each mutant to a copy
# of its module, test it as scripts/check.sh does (with the modules named
# on its "# uses:" line) against the listed checks files in order, and
# require at least one `rce test` to fail.
# Usage: RCE=path/to/rce python3 scripts/mut/ml.py
import glob, os, shutil, subprocess, sys, tempfile
RCE = os.environ.get('RCE', 'rce')
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.insert(0, os.path.join(root, 'scripts'))
import uses
T = 'tests/ml/'
AD = [T + 'ad_checks.rce', T + 'ad_checks_0.rce', T + 'ad_checks_1.rce']
HS = [T + 'ad_checks_hess_0.rce', T + 'ad_checks_hess_1.rce']
TE = [T + 'tensor_checks.rce', T + 'tensor_checks_0.rce']
TP = [T + 'tensor_checks_place.rce', T + 'tensor_checks_probe_1.rce', T + 'tensor_checks_probe.rce']
PR = [T + 'prob_checks.rce'] + sorted(glob.glob(os.path.join(root, T + 'prob_checks_*.rce')))
M = [
    # ml.ad: elementary functions
    ('ad', '    let y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi)', '    let y = 1.0 - ((lo - (r * c) / (2.0 + c)) - hi)', AD),
    ('ad', '    if m >= SQRT2\n', '    if m > 1.5\n', AD),
    ('ad', '        while n >= 2 bound 20', '        while n >= 3 bound 20', AD),
    ('ad', '    if u == 1.0\n        return x\n    return ln(u) * (x / (u - 1.0))', '    if u == 1.0\n        return x\n    return ln(u)', AD),
    ('ad', '    let t = e / (e + 2.0)', '    let t = e / (e + 1.0)', AD),
    ('ad', '    if x > 36.0\n        return x + exp(0.0 - x)', '    if x > 3.0\n        return x + exp(0.0 - x)', AD),
    ('ad', '    let j0 = if s >= 2 then (s - 2) / 24 else 0', '    let j0 = if s >= 2 then (s - 2) / 24 + 1 else 0', AD),
    ('ad', '    if fr[0] >= 8388608', '    if fr[0] >= 16777216', AD),
    ('ad', '        n = (4 - n) & 3', '        n = n & 3', AD),
    ('ad', '    let fl = (bh - (fh - ah)) + ch', '    let fl = ch', AD),
    ('ad', '    return w - (hz - (z * r - x * y))', '    return w - (hz - z * r)', AD),
    ('ad', '        qx = ldexp((ldexp(q, 20 - e)).floor(), e - 20)', '        qx = q', AD),
    ('ad', '                x = (2.0 * x - 1.0) / (2.0 + x)', '                x = (2.0 * x - 1.0) / (2.0 - x)', AD),
    # ml.ad: sweeps and nested AD
    ('ad', '        return (1.0 / z, 0.0 - y / z)', '        return (1.0 / z, y / z)', AD),
    ('ad', '        return (1.0 - y * y, 0.0)', '        return (1.0 - y, 0.0)', AD),
    ('ad', '        return (stepv(x - z, 1), stepv(z - x, 1))', '        return (stepv(x - z, 0), stepv(z - x, 0))', AD),
    ('ad', '        return ((z * y) / x, y * ln(x))', '        return ((z * y) / x, y)', AD),
    ('ad', '                s = s + pb * d[t.b[u] as usize]', '                s = s - pb * d[t.b[u] as usize]', AD),
    ('ad', '        let q = div(t, u, b)\n        let pb = neg(t, q)', '        let q = div(t, u, b)\n        let pb = q', HS),
    ('ad', '        let om = sub(t, c1, u)\n        let pa = mul(t, u, om)', '        let om = sub(t, c1, u)\n        let pa = om', HS),
    ('ad', '        let sn = tsin(t, a)\n        let pa = neg(t, sn)', '        let sn = tsin(t, a)\n        let pa = sn', HS),
    ('ad', '        let s = add(t, g[ju], c)\n        g[ju] = s', '        g[ju] = c', HS),
    ('ad', '        if gi >= 0 and t.a[u] >= 0 and t.op[u] != STEP', '        if gi >= 0 and t.a[u] >= 0', HS),
    # ml.tensor
    ('tensor', '    while h * 2 < n bound 64', '    while h * 2 <= n bound 64', TE),
    ('tensor', '    let mid = lo + n / 2', '    let mid = lo + (n + 1) / 2', TE),
    ('tensor', '                    d.push(if x.data[q] > 0.0 then gy.data[q] else 0.0)', '                    d.push(if x.data[q] >= 0.0 then gy.data[q] else 0.0)', TE),
    ('tensor', '                    let c = ((r * r) * r / (dd as f64)) * sdot', '                    let c = ((r * r) / (dd as f64)) * sdot', TE),
    ('tensor', '                let gb = tmul(&gy, &n.val[xa as usize])', '                let gb = copy(gy)', TE),
    ('tensor', '    return f.tostr() == "-0"', '    return false', TE),
    ('tensor', '    if e < -1022\n        return sg | (ad.ldexp(a, 1074) as u64)', '    if e < -1023\n        return sg | (ad.ldexp(a, 1074) as u64)', TE),
    ('tensor', '            if t < bt or (t == bt and best != "cpu" and c.fabric < best)', '            if t <= bt', TP),
    ('tensor', '        if e.fabric == fabric and e.class == class and e.probe == want and e.reference == want', '        if e.fabric == fabric and e.class == class and e.probe == e.reference', TP),
    ('tensor', '        if c.class == class and c.throughput > 0.0 and admitted(&c.fabric, &c.class, exacts)', '        if c.class == class and admitted(&c.fabric, &c.class, exacts)', TP),
    ('tensor', '        return "6485f9d5ab55f621dd6712c7b687012023c9288264cdc0e53a6c839b43f4fa4f"', '        return "6485f9d5ab55f621dd6712c7b687012023c9288264cdc0e53a6c839b43f4fa4e"', TP),
    ('tensor', '                j += 64', '                j += 32', TP),
    # ml.prob
    ('prob', '    let w = 1.0 / (n + 10.0)', '    let w = 1.0 / (n + 1.0)', PR),
    ('prob', '        p[i as usize] = s / mt.chol[(i * d + i) as usize]', '        p[i as usize] = s', PR),
    ('prob', '                l[(i * d + j) as usize] = s / l[(j * d + j) as usize]', '                l[(i * d + j) as usize] = s', PR),
    ('prob', '            s = s + mt.minv[(i * d + j) as usize] * p[j as usize]', '            s = s + mt.minv[(i * d + i) as usize] * p[j as usize]', PR),
    ('prob', '    let b = det.next_u32(r) >> 6', '    let b = det.next_u32(r) >> 5', PR),
    ('prob', '    let vhat = ((hh - 1.0) / hh) * w + b / hh', '    let vhat = w + b / hh', PR),
]
def run_pkg(mod, src, chk, d):
    f = os.path.join(root, 'lib/ml/%s.rce' % mod)
    deps = uses.deps(f)
    pk = os.path.join(d, 'pkg'); shutil.rmtree(pk, ignore_errors=True); os.makedirs(pk)
    for u in deps: shutil.copy(os.path.join(root, 'lib', u + '.rce'), pk)
    o = os.path.join(pk, mod + '.rce')
    open(o, 'w').write(src + '\n' + open(os.path.join(root, chk)).read())
    return subprocess.run([RCE, 'test', pk if deps else o], capture_output=True).returncode != 0
killed = 0
with tempfile.TemporaryDirectory() as d:
    for i, (mod, a, b, chks) in enumerate(M):
        src = open(os.path.join(root, 'lib/ml/%s.rce' % mod)).read()
        assert src.count(a) == 1, a
        dead = any(run_pkg(mod, src.replace(a, b), c, d) for c in chks)
        killed += dead
        print('%s %2d %s: %s' % ('killed' if dead else 'SURVIVED', i, mod, a.strip().split('\n')[0]), flush=True)
print('%d of %d mutants killed' % (killed, len(M)))
sys.exit(0 if killed == len(M) else 1)
