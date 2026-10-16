"""Gates that run a library through `rce emit` (spec 10.5).

  emitgate.py run LIB EMITFILE
      LIB with EMITFILE appended (a `fn main()` that emits text lines) is
      flattened with its dependencies, emitted to Rust, built with rustc
      and run with the fuel of the file's '# fuel: N' line. Every emitted
      line must start with "ok" and their number must equal the file's
      'expect: N'. For work beyond the 50,000,000 comptime fuel of a check
      (e.g. MCMC against reference posteriors).
  emitgate.py targets LIB CHECKS [TARGET...]
      LIB with CHECKS appended is emitted to each TARGET (python, js, rust;
      default python js) and run; every observation must be true and the
      observation lines must be identical across targets (the checks
      compare floats bit for bit, so this is cross-target bit identity).
Set RCE to the rce binary (default: rce on PATH)."""
import os, re, subprocess, sys, tempfile
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RCE = os.environ.get('RCE', 'rce')

def flat(lib, extra, tmp):
    src = open(lib).read() + '\n' + open(extra).read()
    m = os.path.join(tmp, 'main.rce'); open(m, 'w').write(src)
    out = subprocess.run([sys.executable, os.path.join(root, 'scripts', 'flatten.py'), m], capture_output=True, text=True, check=True).stdout
    # flatten.py resolves '# uses:' relative to the repository's lib/
    f = os.path.join(tmp, 'flat.rce'); open(f, 'w').write(out); return f

def emit(f, lang, tmp):
    ext = {'python': 'py', 'js': 'js', 'rust': 'rs'}[lang]
    r = subprocess.run([RCE, 'emit', lang, f], capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.startswith('not supported'):
        sys.exit('emitgate: emit %s failed: %s' % (lang, (r.stdout + r.stderr)[:300]))
    p = os.path.join(tmp, 'prog.' + ext); open(p, 'w').write(r.stdout)
    if lang == 'python': return [sys.executable, p]
    if lang == 'js': return ['node', p]
    b = os.path.join(tmp, 'prog')
    subprocess.run(['rustc', '-O', '--edition', '2024', '-o', b, p], check=True, capture_output=True)
    return [b]

def run(lib, ef):
    txt = open(ef).read()
    fuel = re.search(r'^# fuel: (\d+)', txt, re.M).group(1)
    expect = int(re.search(r'expect: (\d+)', txt).group(1))
    with tempfile.TemporaryDirectory() as tmp:
        cmd = emit(flat(lib, ef, tmp), 'rust', tmp)
        r = subprocess.run(cmd + ['--fuel=' + fuel], capture_output=True, text=True)
    lines = [l[len('emit '):].strip('"') for l in r.stdout.split('\n') if l.startswith('emit ')]
    for l in lines: print(l)
    ok = sum(1 for l in lines if l.startswith('ok'))
    print('emitgate run %s: %d ok of %d expected' % (ef, ok, expect))
    if r.returncode != 0 or ok != expect or len(lines) != expect:
        print(r.stderr[-2000:]); sys.exit(1)

def targets(lib, cf, langs):
    outs = {}
    with tempfile.TemporaryDirectory() as tmp:
        f = flat(lib, cf, tmp)
        for lang in langs:
            r = subprocess.run(emit(f, lang, tmp), capture_output=True, text=True)
            obs = [l for l in r.stdout.split('\n') if l.startswith('check ')]
            bad = [l for l in obs if not l.endswith('= true')]
            if r.returncode != 0 or bad or not obs:
                sys.exit('emitgate targets %s on %s: %d observations, %d not true, status %d %s' % (cf, lang, len(obs), len(bad), r.returncode, r.stderr[-500:]))
            outs[lang] = obs
    first = outs[langs[0]]
    for lang in langs[1:]:
        if outs[lang] != first: sys.exit('emitgate targets %s: %s differs from %s' % (cf, lang, langs[0]))
    print('emitgate targets %s: %d observations true and identical on %s' % (cf, len(first), ', '.join(langs)))

if __name__ == '__main__':
    if sys.argv[1] == 'run': run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'targets': targets(sys.argv[2], sys.argv[3], sys.argv[4:] or ['python', 'js'])
    else: sys.exit(__doc__)
