# ml: differentiation, tensors and probabilistic inference in pure RCE

Status: proposal, implemented in `lib/ml/ad.rce` (`ml.ad`),
`lib/ml/tensor.rce` (`ml.tensor`) and `lib/ml/prob.rce` (`ml.prob`).

## Abstract

`ml` is a machine learning layer written entirely in RCE. `ml.ad`
computes derivatives of programs to any order on one recorded operation
list: reverse mode, forward mode, Hessians and Hessian-vector products,
and further nesting. `ml.tensor` gives tensors, neural network layers
and optimizers whose every reduction has a fixed evaluation tree, and a
placement hook that admits a GPU only through the engine's FabricExec
probe. `ml.prob` runs seeded Hamiltonian Monte Carlo, the No-U-Turn
sampler with diagonal or dense metrics, and variational inference. Every
result is a function of its inputs and seed alone, with the same bits on
every target, because the only floating point operations used are the
correctly rounded binary64 operations of RCE and a fixed order of them.
Results are checked bit for bit against independent models, against
PyTorch and JAX, against finite differences, and against posteriordb
reference posteriors.

## In plain language

1. The same program gives the same answer, to the last bit, on every
   computer RCE runs on. Training a model or drawing samples twice with
   the same seed gives identical numbers, so a result can be replayed and
   audited.
2. A graphics card is used only after it has proved, on a published test,
   that it computes exactly what the reference computes. Until then the
   ordinary processor does the work, and the answer is the same either
   way.
3. Learned models never decide anything that matters for safety. They
   run outside the trusted part of the system, and nothing about security
   depends on what a model outputs.

## 1. Requirements

R1 (determinism). A result is a function of (program, inputs, seed). It
is bit-identical on every target of `rce emit` (Python, JavaScript, Rust)
and in the interpreter.

R2 (IEEE 754, no contraction). Only binary64 (and binary32 computed per
6.3 f32) addition, subtraction, multiplication, division and square
root are used, each correctly rounded to nearest even; `a * b + c` is
always two roundings. A non-finite intermediate traps (6.3 F4).

R3 (trusted loop). No neural or language model component is part of the
trusted computing base. `ml` computes values; it never grants authority.
Placement onto an accelerator is decided from logged evidence (R5), not
from a model or a heuristic.

R4 (verification). Every function is checked against an independent
reference: bit for bit where the reference follows the same operation
order, within a stated tolerance otherwise.

R5 (placement). A kernel leaves the CPU reference only for a fabric whose
logged probe digest equals the reference digest for that kernel class
(HC-12, HC-13).

## 2. Derivations

### 2.1 Where the elementary functions come from

RCE has no built-in transcendental functions (6.3 F9 [R1]): platform
libraries differ in the last bit, which would violate R1. `exp`, `ln`
(as `math.log`), `sin` and `cos` are the correctly rounded binary64
functions of this repository's `math` module (`lib/math/math.rce`,
written in RCE with + - * / and sqrt only, after CORE-MATH), which
`ml.ad` names on its `# uses:` line: the result is the double nearest the
real value, for every finite argument, on every target. Payne and Hanek
reduction for large trigonometric arguments now lives there.
`ml.ad` builds the rest from them in a fixed order: `expm1` (Taylor
series below 0.35, else `exp(x) - 1`), `log1p` (Goldberg's correction),
`tanh`, `sigmoid`, `softplus` and `pow` (`exp(y ln x)`). These are not
correctly rounded but are the same everywhere, which is what R1 needs.
Measured against mpmath at 300 bits on 20,000 random arguments each:
expm1 2.0, log1p 2.4, tanh 2.6, sigmoid 1.8, softplus 2.3 and atan 0.64
ulp; `pow` loses about |y ln x| ulp (31.7 ulp for x in [0.01, 100],
|y| <= 5) because the rounding of ln x is scaled by y.

The correctly rounded functions cost far more interpreter fuel than the
short kernels they replace, so the comptime checks are split into many
files (`scripts/gen/ml_reshard.py` halves a file until each part fits).

`atan` is fdlibm 5.3 `s_atan.c` [F1]: reduction by the addition formula
at 1/2, 1, 3/2 and infinity, then an odd polynomial in two halves.

### 2.2 Automatic differentiation

A program is recorded as a Wengert list (`Tape`): node i has an
operation, up to two argument nodes, an integer parameter and its value.
For node u = f(a, b) let pa, pb be its partial derivatives, computed from
recorded values.

Reverse mode. With adjoints g (g_out = 1), visiting nodes from out down
to 0: g_a += g_u pa, g_b += g_u pb. One sweep gives the gradient with
respect to all inputs. The order of additions is the node order, so the
gradient is a fixed sequence of roundings.

Forward mode. With tangents d over the inputs, visiting nodes upwards:
d_u = pa d_a + pb d_b.

Nested differentiation. `grad_tape` records the reverse sweep itself on
the tape: for each node it appends nodes computing pa and pb (from the
same operations, with the same roundings as the values `partials`
returns) and the products and sums of the adjoint recurrence. The
gradient entries become ordinary nodes, so every mode applies to them
again:

- reverse over reverse: the Hessian, row i = gradient of gradient node i;
- forward over reverse: H v in one forward sweep of the gradient program;
- further nesting: third and higher derivatives.

Each level is an explicit region of the tape with its own nodes; nothing
is captured by closures, so the perturbation confusion of
closure-based nested forward mode cannot arise. Two design points follow
from correctness rather than speed. First, the recorded sweep keeps
adjoints that are zero at the point (the plain sweep skips them):
skipping would lose second derivatives at critical points. Second,
piecewise operations (relu, abs, max, min) record their derivative as a
`STEP` node whose own derivative is zero, with the tie conventions of
PyTorch and JAX (relu'(0) = abs'(0) = 0; max and min split the gradient
evenly at a tie).

Operations: add, sub, mul, div, neg, exp, ln, sqrt, tanh, sigmoid, sin,
cos, integer power, relu, softplus, log1p, expm1, reciprocal, abs, max,
min, pow (x > 0), atan, and step.

### 2.3 Tensors and reduction trees

Floating point addition is not associative, so a sum is defined by its
tree. `ml.tensor` names the tree of every reduction: `LEFT` (a left fold
from the first element; the default proposed as N-9), `PAIRWISE` (split
at n/2), and `T18`, the engine's tree (split at the largest power of two
below n, `fixed_tree_sum` in the host). The shape and the tree, never a
schedule or a thread count, fix the bits. Products are rounded before
they are added.

Layers are recorded on a tensor tape (`Net`) with hand-written
vector-Jacobian products: linear (x W^T + b), bias, relu, tanh, sigmoid,
softplus, elementwise sum (residual) and product (gate), RMS
normalisation with weight, mean squared error and softmax cross entropy.
Optimizers follow PyTorch's update formulas: SGD with momentum and
weight decay, Adam.

Signed zero. 6.3 F6 [R1] makes -0.0 a distinct state while `==` cannot tell
it from 0.0. `signbit` reads the sign through binary32 text, which F6
fixes as "-0" on every target; `f64_bits` and `f32_bits` return raw bit
patterns including the sign, and `same_bits` compares raw bits.

### 2.4 Placement through FabricExec

The engine's placement executor (host contract HC-12, HC-13 [R2];
`FabricExec`, `probe_fabric` and `place_from` in the host [R3]) admits a
fabric for an operation class only after a logged `host.Exact` entry
whose probe digest equals the CPU reference digest; the probe vector
contains signed zeros, subnormal results, infinities, NaN and every sum
length up to 64. The cost model chooses the least overhead + work /
throughput among the CPU (16, 1) and the admitted, calibrated fabrics,
ties to the CPU and then to the lexically smaller fabric id.

`ml.tensor.place` is the same function of the same entries, for the
classes `sum.f64` and `lanes.f64.{add, sub, mul, div, neg, sqrt}`, and
additionally requires the logged reference to equal the digest `ml`
itself expects. The binary64 reference digests were produced by the
engine's own `probe_fabric` (an instrumented run at openIE-dev/rce
b497ce3) and reproduced independently by
`scripts/ref/ml/fabric_exec_ref.py`; since the probe vector contains
non-finite values, which trap in RCE, the checks recompute the finite
part of every probe in RCE bit for bit.

Ferric (github.com/dcharlot-physicalai-bmi/ferric) computes in binary32
only, so it cannot pass the binary64 probes and is never placed for
them: this is the intended outcome of HC-13, not a gap in the hook. Its
f32 kernels are admitted under the classes `f32.mm` and
`f32.rmsnorm_tree` by the HC-13 rule for per-device verified operations:
`ml.tensor` reproduces Ferric's probe kernels bit for bit in RCE (f32
per 6.3, Newton reciprocal and inverse square root, the strided 64-lane
tree) and computes the digests Ferric publishes (`2a5bf3731421828e`,
`ddeeb58c50f5ae9f`).

### 2.5 Probabilistic inference

A model is a log density on unconstrained R^d written against the tape;
inference reads its gradient by the reverse sweep. All randomness comes
from `rand.det` (ChaCha20) under a 32-byte seed, one stream per chain.
Uniforms take 53 random bits; normals use the Box-Muller transform with
`ml.ad` `ln`, `sqrt`, `sin` and `cos`, so draws are the same on every
target.

Samplers: static HMC with a leapfrog integrator; NUTS with multinomial
sampling over the trajectory and the generalised no-U-turn criterion;
step size by dual averaging (gamma 0.05, t0 10, kappa 0.75) with a
heuristic initial step size. Metrics: diagonal, or dense. In warmup the
draws of [w/4, 3w/4) give the variance (diagonal) or the covariance
(dense, Welford's update of the co-moments), shrunk toward 1e-3 I as
Stan does: Sigma = n/(n+5) S + 1e-3 * 5/(n+5) I. For a dense metric
M^-1 = Sigma = L L^T (Cholesky, sums as left folds); the kinetic energy
is p^T Sigma p / 2, the velocity Sigma p, and a momentum p ~ N(0, M) is
the solution of L^T p = z for z ~ N(0, I), since then Cov(p) = L^-T L^-1
= Sigma^-1 = M. Variational inference: mean-field Gaussian ADVI with
reparameterised gradients and Adam. Diagnostics: split R-hat and an
effective sample size by Geyer's initial monotone sequence.

## 3. Verification

| Module | Checks | References |
|---|---|---|
| `ml.ad` | 2,271 | Python model of the tape (bit for bit: values, gradients, JVPs, Hessians, HVPs); PyTorch 2.9 float64 autograd after `gradcheck`, cross-checked with JAX 0.6 (gradients within 6.4e-16, Hessians within 5.3e-16 relative); central differences; mpmath for the elementary functions |
| `ml.tensor` | 132 | Python model of each tree (bit for bit); PyTorch 2.9 float64 (layers, losses, gradients, 6 SGD and Adam steps); central differences; the engine's probe digests; Ferric's published digests |
| `ml.prob` | 63 + 41 | PyTorch densities and gradients of five models; NumPy R-hat, ESS, Cholesky; Python model of the draws (bit for bit); reversibility; replay; posteriordb [P1] reference posteriors for Rate_1, eight schools (noncentered), kidiq, a Bayesian linear regression, and an analytic correlated Gaussian, under NUTS with diagonal and dense metrics and under ADVI |

Cross-target identity: every checks file is emitted to Python,
JavaScript and Rust and must give identical observation lines
(`scripts/emitgate.py targets`). Mutation testing: `scripts/mut/ml.py`.

## 4. Open questions

- Correct rounding for `expm1`, `log1p`, `tanh` and `atan`: when the
  `math` module gains them, `ml.ad` should switch (bits change once).
- A wgpu backend that offers binary64 lanes would enter through the
  `sum.f64` and `lanes.f64.*` classes unchanged.
- Emitter limits (writes back in conditional positions, a loop variable
  shadowing an outer one in the Python target) shape some code; see
  the [ml-ad] entries of the progress log.

## References

Only sources read in full are cited.

- [F1] fdlibm 5.3, `s_atan.c`, Sun Microsystems 1993 (netlib.org/fdlibm).
- [R1] RCE specification, the 6.3 paragraphs F6 (signed zero) and F9
  (transcendentals),
  openIE-dev/rce `spec/RCE-SPEC.md` at b497ce3.
- [R2] RCE book, chapter 16 (host contract), the HC-12 paragraphs on
  `cpu.threads` and platform fabrics and the HC-13 bit-exactness gate,
  openIE-dev/rce `spec/book/16-host-contract.md` at b497ce3.
- [R3] openIE-dev/rce `spec/checker/src/host.rs` at b497ce3: `place_from`,
  `fixed_tree_sum`, `fixed_tree`, `LaneOp`, `FabricExec`, `probe_vector`,
  `bits_digest`, `Host::probe`, `Host::probe_fabric`, `place_sum`,
  `place_lanes`; and `spec/checker/tests/host_place.rs`.
- [P1] stan-dev/posteriordb at 91027f9: the Stan model, data and
  reference posterior draw files of the four posteriors named in
  section 3 (the correlated Gaussian has an analytic reference).

Methods named above by their usual names (Wengert lists, HMC, NUTS, dual averaging, ADVI, split
R-hat, Geyer's sequence, Box-Muller, Welford's update, Stan's metric
regularisation) are described here from their definitions as
implemented; the papers that introduced them are not cited because they
were not read in full for this work.
