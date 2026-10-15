"""Generate tests/ml/prob_checks.rce and tests/ml/prob_emit.rce for
lib/ml/prob.rce. Reference posteriors: stan-dev/posteriordb at 91027f9
(posterior_database/reference_posteriors/draws, 10 chains x 1000 draws
from Stan). Run with /workspace/venv310/bin/python; PDB points at a
posteriordb checkout (default /workspace/posteriordb-ro)."""
import io, json, math, os, random, statistics as S, sys, zipfile
import numpy as np, torch
torch.set_default_dtype(torch.float64)
ROOT=os.path.join(os.path.dirname(__file__),'..','..')
sys.path.insert(0,os.path.join(ROOT,'scripts','ref','ml')); import mlref as M
PDB=os.environ.get('PDB','/workspace/posteriordb-ro/posterior_database')
def lit(x):
    r=repr(float(x)); return r if ('.' in r or 'e' in r) else r+'.0'
def L(xs): return '['+', '.join(lit(x) for x in xs)+']'
def zjson(p): 
    with zipfile.ZipFile(p) as z: return json.loads(z.read(z.namelist()[0]))
def data(name): return zjson(f'{PDB}/data/data/{name}.json.zip')
def ref(name):
    d=zjson(f'{PDB}/reference_posteriors/draws/draws/{name}.json.zip'); out=[]
    for k in d[0]:
        xs=[v for ch in d for v in ch[k]]; out.append((k,S.fmean(xs),S.stdev(xs)))
    return out
r1=data('Rate_1_data'); es=data('eight_schools'); kd=data('kidiq'); bl=data('sblri')
D_RATE=[float(r1['n']),float(r1['k'])]
D_EIGHT=[float(es['J'])]+[float(v) for v in es['y']]+[float(v) for v in es['sigma']]
# torch densities, same unnormalised forms as scripts/gen/ml_prob_models.rce
sp=torch.nn.functional.softplus
def t_rate(x): return -(D_RATE[1]+1)*sp(-x[0])-(D_RATE[0]-D_RATE[1]+1)*sp(x[0])
def t_eight(x):
    J=8; y=torch.tensor(D_EIGHT[1:9]); s=torch.tensor(D_EIGHT[9:17]); tt=x[:J]; mu=x[J]; v=x[J+1]; tau=torch.exp(v)
    th=tt*tau+mu
    return v-0.5*(tt**2).sum()-0.5*(((th-y)/s)**2).sum()-0.5*(mu/5)**2-torch.log1p((tau/5)**2)
ky=torch.tensor([float(v) for v in kd['kid_score']]); kx=torch.tensor([float(v) for v in kd['mom_iq']])
def t_kid(x):
    b1,b2,v=x[0],x[1],x[2]; sg=torch.exp(v); r=ky-b1-b2*kx
    return -0.5*(r**2).sum()/sg**2+(1-len(ky))*v-torch.log1p((sg/2.5)**2)
BX=torch.tensor(bl['X']); BY=torch.tensor(bl['y'])
def t_blr(x):
    D=bl['D']; b=x[:D]; v=x[D]; sg=torch.exp(v); r=BY-BX@b
    return -0.5*(r**2).sum()/sg**2+(1-bl['N'])*v-0.5*(sg/10)**2-0.5*((b/10)**2).sum()
rnd=random.Random(4711)
out=[]
KID_DATA=f'suff2(&{L(kd["kid_score"])}, &{L(kd["mom_iq"])})'
flatX=[v for row in bl['X'] for v in row]
# data builders in RCE (computed there from the raw posteriordb data)
datafns=f'''
fn data_rate() -> List[f64] = {L(D_RATE)}
fn data_eight() -> List[f64] = {L(D_EIGHT)}
fn data_kid() -> List[f64]
    let y = {L(kd["kid_score"])}
    let x = {L(kd["mom_iq"])}
    return suff2(&y, &x)
fn data_blr() -> List[f64]
    let xm = {L(flatX)}
    let y = {L(bl["y"])}
    return blr_stats({bl["N"]}, {bl["D"]}, &xm, &y)
fn data_of(k: i64) -> List[f64]
    if k == 0
        return data_rate()
    if k == 1
        return data_eight()
    if k == 2
        return data_kid()
    if k == 3
        return data_blr()
    return data_corr()
fn data_corr() -> List[f64] = [0.99, 1.0, 10.0]
'''
def t_corr(x):
    r,s0,s1=0.99,1.0,10.0; u=x[0]/s0; v=x[1]/s1
    return (-0.5/(1-r*r))*((u*u-(2*r)*(u*v))+v*v)
TF=[t_rate,t_eight,t_kid,t_blr,t_corr]; DIMS=[1,10,3,bl['D']+1,2]
CENT=[[0.0],[0.0]*8+[4.0,1.0],[26.0,0.6,2.9],[1.0]*bl['D']+[0.0],[0.0,0.0]]
SCL=[[1.0],[1.0]*8+[3.0,0.8],[6.0,0.06,0.05],[0.002]*bl['D']+[0.1],[1.0,10.0]]
for k in range(5):
    for i in range(6):
        q=[c+s*rnd.uniform(-2,2) for c,s in zip(CENT[k],SCL[k])]
        x=torch.tensor(q,requires_grad=True); v=TF[k](x); v.backward()
        out.append(f'check density_ref_k({k}, {L(q)}, {lit(v.item())}, {L(x.grad.tolist())})')
# draws
for b,st in [(1,0),(1,3),(200,0)]:
    seed=[(b+7*i)%256 for i in range(32)]
    r=M.Rng(seed,st); us=[r.uniform() for _ in range(40)]
    out.append(f'check uniform_bits({b}, {st}, {L(us)})')
    r=M.Rng(seed,st); ns=[r.normal() for _ in range(40)]
    out.append(f'check normal_bits({b}, {st}, {L(ns)})')
# leapfrog reversibility
for i in range(3):
    q=[c+s*rnd.uniform(-1,1) for c,s in zip(CENT[1],SCL[1])]; p=[rnd.gauss(0,1) for _ in q]
    out.append(f'check reversible(data_eight(), {L(q)}, {L(p)}, {lit(0.05*(i+1))}, {10*(i+1)})')
# rhat / ess vs numpy (same formulas)
def np_rhat(xs,c):
    n=len(xs)//c; h=n//2; parts=[]
    for j in range(2*c):
        base=(j//2)*n+(j%2)*h+(j%2)*(n-2*h); parts.append(np.array(xs[base:base+h]))
    # mirror RCE indexing: part j starts at j*h + (j//2)*(n-2h)
    parts=[np.array(xs[j*h+(j//2)*(n-2*h): j*h+(j//2)*(n-2*h)+h]) for j in range(2*c)]
    ms=np.array([p.mean() for p in parts]); vs=np.array([p.var(ddof=1) for p in parts])
    W=vs.mean(); B=h*ms.var(ddof=1); vh=(h-1)/h*W+B/h; return math.sqrt(vh/W)
def np_ess(xs,c):
    n=len(xs)//c; ch=[np.array(xs[j*n:(j+1)*n]) for j in range(c)]
    mus=[a.mean() for a in ch]; W=np.mean([a.var(ddof=1) for a in ch]); vplus=(n-1)/n*W+np.var(mus,ddof=1)
    def ac(a,mu,k): return ((a[:n-k]-mu)*(a[k:]-mu)).sum()/n
    tau=-1.0; prev=1e300; k=0
    while k+1<n:
        a0=sum(ac(a,m,k) for a,m in zip(ch,mus)); a1=sum(ac(a,m,k+1) for a,m in zip(ch,mus))
        r0=1-(W-a0/c*(n/(n-1)))/vplus; r1=1-(W-a1/c*(n/(n-1)))/vplus; pk=r0+r1
        if pk<=0: break
        pk=min(pk,prev); prev=pk; tau+=2*pk; k+=2
    tot=c*n; cap=tot*math.log10(tot); e=tot/tau; return min(e,cap)
for i,(c,n,phi) in enumerate([(2,60,0.0),(4,50,0.6),(3,41,0.9)]):
    xs=[]
    for j in range(c):
        x=rnd.gauss(0,1)
        for _ in range(n): x=phi*x+rnd.gauss(0,1)*math.sqrt(1-phi*phi); xs.append(x+0.1*j)
    out.append(f'check rhat_ref({L(xs)}, {c}, {lit(np_rhat(xs,c))})')
    out.append(f'check ess_ref({L(xs)}, {c}, {lit(np_ess(xs,c))})')
for sm in (0,1):
    for mt in (0,1): out += [f'check replay({sm}, {mt})', f'check reseed({sm}, {mt})']
# dense metric pieces
for i,d in enumerate((2,3,5)):
    A=np.array([[rnd.gauss(0,1) for _ in range(d)] for _ in range(d)]); A=A@A.T+0.5*np.eye(d)
    Lc=np.linalg.cholesky(A); p=[rnd.gauss(0,1) for _ in range(d)]
    out.append(f'check chol_ref({L(A.flatten())}, {d}, {L(Lc.flatten())})')
    out.append(f'check kinetic_ref({L(A.flatten())}, {d}, {L(p)}, {lit(0.5*float(np.array(p)@A@np.array(p)))})')
    out.append(f'check draw_ref({L(A.flatten())}, {d}, {i+3})')
out.append('check reversible_dense([1.0, 2.7, 2.7, 9.0], [0.4, -1.1], [0.3, 0.8], 0.05, 20)')
models=open(os.path.join(ROOT,'scripts/gen/ml_prob_models.rce')).read()
head=open(os.path.join(ROOT,'scripts/gen/ml_prob_head.rce')).read()
extra='''
fn density_ref_k(k: i64, q: List[f64], want: f64, wg: List[f64]) -> bool = density_ref(k, data_of(k), q, want, wg)
'''
cut=next(i for i,l in enumerate(out) if l.startswith('check replay'))
rest=out[cut:]; out=out[:cut]
open(os.path.join(ROOT,'tests/ml/prob_checks.rce'),'w').write(models+datafns+head+extra+'\n'+'\n'.join(out)+'\n')
for i in range(0,len(rest),2):
    open(os.path.join(ROOT,f'tests/ml/prob_checks_{i//2}.rce'),'w').write(models+datafns+head+extra+'\n'+'\n'.join(rest[i:i+2])+'\n')
# ---- heavy: inference against posteriordb reference posteriors
R=[ref('Rate_1_data-Rate_1_model'),ref('eight_schools-eight_schools_noncentered'),ref('kidiq-kidscore_momiq'),ref('sblri-blr')]
# eight schools reference order: theta[1..8], mu, tau (as eight_out)
emit=[models,datafns,'''
fn fmt(x: f64) -> text = x.tostr()

fn outs(k: i64, q: &[f64]) -> List[f64]
    if k == 0
        return rate1_out(q)
    if k == 1
        return eight_out(q)
    if k == 2
        return kid_out(q)
    if k == 3
        return blr_out(q)
    return corr2_out(q)

fn run(k: i64, m: &Model, q0: &[f64], seed: &[i64], st: &Settings) -> Fit
    if k == 0
        return sample(m, rate1, q0, seed, st, 1)
    if k == 1
        return sample(m, eight, q0, seed, st, 1)
    if k == 2
        return sample(m, kid, q0, seed, st, 1)
    if k == 3
        return sample(m, blr, q0, seed, st, 1)
    return sample(m, corr2, q0, seed, st, 1)

fn fit_advi(k: i64, m: &Model, q0: &[f64], seed: &[i64]) -> Advi
    if k == 0
        return advi(m, rate1, q0, seed, 2000, 4, 0.05)
    if k == 2
        return advi(m, kid, q0, seed, 2000, 4, 0.05)
    return advi(m, blr, q0, seed, 2000, 4, 0.05)

# Compare each output column with the reference mean and sd: mean within
# 4 Monte Carlo standard errors (ours from ESS, the reference's from its
# 10000 draws), sd within 4 / sqrt(2 ESS) + 0.03 relative, R-hat < 1.01.
fn judge(k: i64, name: text, seed: &[i64], st: &Settings, q0: List[f64], names: List[text], rm: List[f64], rs: List[f64]) -> List[text]
    let m = Model(dim: q0.len() as i64, data: data_of(k))
    let f = run(k, &m, &q0, seed, st)
    var cols: List[List[f64]] = []
    for _ in 0..rm.len()
        cols.push([])
    for d in f.draws
        let o = outs(k, &d)
        for j in 0..o.len()
            cols[j].push(o[j])
    var lines: List[text] = []
    let ndiv = f.divergent
    for j in 0..rm.len()
        let c = &cols[j]
        let mu = mean_of(c)
        let sd = sd_of(c)
        let e = ess(c, st.chains)
        let rh = split_rhat(c, st.chains)
        let se = ((sd * sd) / e + (rs[j] * rs[j]) / 10000.0).sqrt()
        let okm = (mu - rm[j]).abs() <= 4.0 * se
        let oks = (sd / rs[j] - 1.0).abs() <= 4.0 / (2.0 * e).sqrt() + 0.03
        let okr = rh < 1.01
        let tag = if okm and oks and okr then "ok" else "FAIL"
        lines.push(tag + " nuts " + copy(name) + " " + copy(names[j]) + " mean " + fmt(mu) + " ref " + fmt(rm[j]) + " sd " + fmt(sd) + " ref " + fmt(rs[j]) + " ess " + fmt(e) + " rhat " + fmt(rh))
    let okd = ndiv * 100 <= (f.draws.len() as i64)
    lines.push((if okd then "ok" else "FAIL") + " nuts " + copy(name) + " divergent " + ndiv.tostr() + " of " + f.draws.len().tostr() + " eps " + fmt(f.eps[0]) + " accept " + fmt(f.accept[0]))
    return lines

# ADVI: the variational mean (mapped to the output scale) within 0.3
# reference sd of the reference mean.
fn judge_advi(k: i64, name: text, seed: &[i64], q0: List[f64], names: List[text], rm: List[f64], rs: List[f64]) -> List[text]
    let m = Model(dim: q0.len() as i64, data: data_of(k))
    let a = fit_advi(k, &m, &q0, seed)
    let o = outs(k, &a.mu)
    var lines: List[text] = []
    for j in 0..rm.len()
        let ok = (o[j] - rm[j]).abs() <= 0.3 * rs[j]
        lines.push((if ok then "ok" else "FAIL") + " advi " + copy(name) + " " + copy(names[j]) + " mean " + fmt(o[j]) + " ref " + fmt(rm[j]))
    return lines

fn seed_of(b: i64) -> List[i64]
    var s: List[i64] = []
    for i in 0..32
        s.push((b + 7 * i) % 256)
    return s

fn main()
    let sd = seed_of(20261015)
''']
specs=[(0,'Rate_1',0.8,[0.0],0),(1,'eight_schools_noncentered',0.95,[0.0]*8+[0.0,0.0],0),(2,'kidscore_momiq',0.8,[0.0,0.0,0.0],0),(3,'blr',0.8,[0.0]*bl['D']+[0.0],0),(2,'kidscore_momiq',0.8,[0.0,0.0,0.0],1),(4,'corr2',0.8,[0.0,0.0],1)]
R.append([('x0',0.0,1.0),('x1',0.0,10.0)])
nlines=0
for si,(k,name,delta,q0,mtr) in enumerate(specs):
    names='['+', '.join('"%s"'%n for n,_,_ in R[k])+']'
    rm=[m for _,m,_ in R[k]]; rs=[s for _,_,s in R[k]]
    tag=name+(' dense' if mtr else '')
    emit.append(f'    let st{si} = Settings(warmup: 500, draws: 500, chains: 4, steps: 8, delta: {lit(delta)}, max_depth: 10, metric: {mtr})')
    emit.append(f'    for l in judge({k}, "{tag}", &sd, &st{si}, {L(q0)}, {names}, {L(rm)}, {L(rs)})\n        emit l')
    nlines+=len(rm)+1
    if k not in (1,4) and mtr==0:
        emit.append(f'    for l in judge_advi({k}, "{name}", &sd, {L(q0)}, {names}, {L(rm)}, {L(rs)})\n        emit l')
        nlines+=len(rm)
hdr=f'# Inference against posteriordb reference posteriors (emitted Rust; see\n# scripts/emitgate.py). expect: {nlines}\n# fuel: 1000000000000\n'
open(os.path.join(ROOT,'tests/ml/prob_emit.rce'),'w').write(hdr+'\n'.join(emit)+'\n')
print('comptime checks',len(out),'emit lines',nlines)
