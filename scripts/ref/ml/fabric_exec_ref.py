"""Reference digests of the engine's HC-13 FabricExec probe.

A line-for-line port of probe_vector, fixed_tree_sum, LaneOp::eval,
bits_digest and the case lists of Host::probe_fabric from the engine
(openIE-dev/rce spec/checker/src/host.rs at b497ce3). Prints, per class,
the reference digest a fabric's probe must equal, plus the finite cases
ml.tensor recomputes in RCE (RCE traps on non-finite values, so the
special lanes are checked here only).
"""
import hashlib, math, struct, sys

MINP = 2.2250738585072014e-308

def probe_vector():
    a = [0.0, -0.0, MINP / 4.0, 1e308, -1e308, math.inf, math.nan, 1.0, 3.0, 1e16, -1e16, 0.1]
    b = [-0.0, 0.0, MINP / 8.0, 1e308, 1e-308, 1.0, 2.0, 3.0, 7.0, 1.0, 1.0, 0.2]
    a += [MINP, MINP / 4.0, MINP * 0.75]
    b += [0.5, 1.0, -MINP / 2.0]
    for i in range(1000):
        a.append(float((i * 7919) % 1009) * (-1e8 if i % 3 == 0 else 1.0 / 3.0))
        b.append(float((i * 104729) % 1013) + 0.5)
    return a, b

def fixed_tree_sum(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]
    h = max((1 << (n - 1).bit_length()) // 2, 1)
    return fixed_tree_sum(xs[:h]) + fixed_tree_sum(xs[h:])

def ieee(f, *args):
    try:
        return f(*args)
    except ZeroDivisionError:
        a, b = args
        if a != a or a == 0.0:
            return math.nan
        neg = (math.copysign(1, a) < 0) != (math.copysign(1, b) < 0)
        return -math.inf if neg else math.inf
    except ValueError:
        return math.nan
    except OverflowError:
        return math.inf

def lane(op, a, b):
    if op == 'add': return a + b
    if op == 'sub': return a - b
    if op == 'mul': return a * b
    if op == 'div': return ieee(lambda x, y: x / y, a, b)
    if op == 'neg': return -a
    if op == 'sqrt': return math.nan if a < 0 or a != a else (math.inf if a == math.inf else math.sqrt(a))
    raise ValueError(op)

def bits(x):
    return 0x7ff8000000000000 if x != x else struct.unpack('<Q', struct.pack('<d', x))[0]

def digest(xs):
    return hashlib.sha256(''.join(f'{bits(x):016x}' for x in xs).encode()).hexdigest()

def sum_cases():
    a, _ = probe_vector()
    fin = [x for x in a if math.isfinite(x)]
    cases = [fin[:k] for k in range(65)] + [fin]
    cases += [[1.0, x, -1.0] for x in a if (not math.isfinite(x)) or x == 0.0]
    return cases

def reference(cls):
    a, b = probe_vector()
    if cls == 'sum.f64':
        return [fixed_tree_sum(c) for c in sum_cases()]
    op = cls.split('.')[-1]
    return [lane(op, a[i], b[i]) for i in range(len(a))]

CLASSES = ['sum.f64'] + [f'lanes.f64.{o}' for o in ['add', 'sub', 'mul', 'div', 'neg', 'sqrt']]

if __name__ == '__main__':
    for c in CLASSES:
        print(c, digest(reference(c)))
