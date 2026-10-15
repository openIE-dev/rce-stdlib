# Flatten a foundation module and the modules it names on its
# "# uses:" line into one file, for `rce run`, which takes a single file
# (needed where each event must get its own fuel, e.g. Ed25519 per event).
# Dependency bodies lose their `module` lines; the main module loses its
# `use` lines and the `alias.` qualifiers of those modules.
# Usage: python3 scripts/flatten.py lib/energy/energy.rce [extra.rce...] > out.rce
import re, sys, os
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
main = open(sys.argv[1]).read()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uses
try:
    deps = uses.deps(sys.argv[1])
except uses.UsesError as e:
    sys.exit('flatten.py: %s' % e)
out = []
lines = main.split('\n')
head, body = lines[0], lines[1:]
out.append(head)
out += [l for l in body if l.startswith('#!')]
# Top-level names that two of the flattened modules both define (ml.ad
# wraps math.exp as its own exp, say) are renamed in the dependency that
# defines them, to <alias>_<name>, and every "alias.name" use follows.
# Dependencies lose their `use` lines and qualifiers as the main module does.
DEF = re.compile(r'^(?:pub )?(?:fn|const|struct|enum|type)\s+([A-Za-z_]\w*)', re.M)
mods = []
for d in deps:
    src = open(os.path.join(root, 'lib', d + '.rce')).read().split('\n')
    # a dependency may start with '# rce-version: V' before its module line
    # (rand.det does, for Cap); the version line moves to the top
    mi = next(i for i, l in enumerate(src) if l.startswith('module '))
    for l in src[:mi]:
        if l.startswith('# rce-version:') and not out[0].startswith('# rce-version:'):
            out.insert(0, l)
    name = src[mi].split()[1]
    mods.append((name.split('.')[-1], [l for l in src[mi + 1:] if not l.startswith('#!')]))
aliases = [a for a, _ in mods]
defs = {a: set(DEF.findall('\n'.join(ls))) for a, ls in mods}
defs[None] = set(DEF.findall(main))
rename = {}
for a, _ in mods:
    for n in defs[a]:
        if any(n in defs[o] for o in defs if o != a):
            rename[(a, n)] = '%s_%s' % (a, n)
# A local (let/var/for) in one module that has the name of another
# module's top-level definition would shadow it once flattened (W-NAME-2),
# so such locals get a trailing underscore in their own module.
LOC = re.compile(r'\b(?:let|var|for)\s+([A-Za-z_]\w*)')
texts = {a: '\n'.join(ls) for a, ls in mods}
texts[None] = main
alltop = set().union(*defs.values())
PAR = re.compile(r'^(?:pub )?fn [^(]*\((.*)\)', re.M)
def locs(t):
    ns = set(LOC.findall(t))
    for ps in PAR.findall(t):
        ns |= set(re.findall(r'([A-Za-z_]\w*)\s*:', ps))
    return ns
shadow = {o: set(n for n in locs(t) if n in alltop and n not in defs[o]) for o, t in texts.items()}
def fix(l, own):
    for n in shadow[own]:
        # not a field (x.n) or a field name or named argument (n: v)
        if re.match(r'(?:pub )?fn ', l):
            head, _, rest = l.partition(')')
            head = re.sub(r'(?<![\w.])%s(?=\s*:)' % re.escape(n), n + '_', head)
            l = head + _ + re.sub(r'(?<![\w.])%s\b(?!\s*:(?!:))' % re.escape(n), n + '_', rest)
            continue
        # a typed let/var names it before ':'; elsewhere 'n:' is a field
        l = re.sub(r'^(\s*(?:let|var)\s+)%s(?=\s*:)' % re.escape(n), r'\g<1>' + n + '_', l)
        l = re.sub(r'(?<![\w.])%s\b(?!\s*:(?!:))' % re.escape(n), n + '_', l)
    for (a, n), r in rename.items():
        l = re.sub(r'(?<![\w.])%s\.%s\b' % (re.escape(a), re.escape(n)), r, l)
        if a == own:
            l = re.sub(r'(?<![\w.])%s\b' % re.escape(n), r, l)
    for a in aliases:
        l = re.sub(r'(?<![\w.])%s\.(?=[A-Za-z_])' % re.escape(a), '', l)
    return l
for a, ls in mods:
    out += [fix(l, a) for l in ls if not re.match(r'^use ', l)]
for l in body:
    if l.startswith('#!') or re.match(r'^use ', l):
        continue
    out.append(fix(l, None))
for extra in sys.argv[2:]:
    out.append('')
    out.append(open(extra).read())
print('\n'.join(out))
