"""Generate tests/ml/tensor_checks*.rce for lib/ml/tensor.rce against
PyTorch 2.9 float64 (run with /workspace/venv310/bin/python)."""
import os, random, struct, sys, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ref', 'ml'))
import fabric_exec_ref as FX
torch.set_default_dtype(torch.float64)
ROOT=os.path.join(os.path.dirname(__file__),'..','..')
def lit(x):
    r=repr(float(x)); return r if ('.' in r or 'e' in r) else r+'.0'
def L(xs): return '['+', '.join(lit(x) for x in xs)+']'
def LI(xs): return '['+', '.join(str(int(x)) for x in xs)+']'
def left(xs):
    s=xs[0]
    for x in xs[1:]: s=s+x
    return s
def pw(xs):
    if len(xs)==1: return xs[0]
    m=len(xs)//2; return pw(xs[:m])+pw(xs[m:])
def red(xs,t): return 0.0 if not xs else (pw(xs) if t==1 else (FX.fixed_tree_sum(xs) if t==2 else left(xs)))
def U(xs): return '['+', '.join(str(FX.bits(x)) for x in xs)+']'
rnd=random.Random(15102026)
out=[]
# reductions
for t in (0,1,2):
    for n in (1,2,3,7,8,31,64,100):
        xs=[rnd.uniform(-1,1)*10**rnd.randint(-3,3) for _ in range(n)]
        out.append(f'check reduce_bits({L(xs)}, {t}, {lit(red(xs,t))})')
cancel=[1e16,1.0,-1e16,1.0]
out.append(f'check reduce_bits({L(cancel)}, 0, {lit(left(cancel))})')
out.append(f'check reduce_bits({L(cancel)}, 1, {lit(pw(cancel))})')
out.append(f'check trees_differ({L(cancel)})')
c5=[1e16,1.0,1.0,-1e16,1.0]
for t in (1,2): out.append(f'check reduce_bits({L(c5)}, {t}, {lit(red(c5,t))})')
out.append(f'check tree_split({L(c5)})')
# matmul
for i in range(12):
    m,k,n=rnd.randint(1,5),rnd.randint(1,9),rnd.randint(1,5); t=i%2
    a=[rnd.uniform(-2,2) for _ in range(m*k)]; b=[rnd.uniform(-2,2) for _ in range(k*n)]
    want=[red([a[r*k+q]*b[q*n+c] for q in range(k)],t) for r in range(m) for c in range(n)]
    tr=(torch.tensor(a).reshape(m,k)@torch.tensor(b).reshape(k,n)).flatten().tolist()
    out.append(f'check matmul_bits({m}, {k}, {n}, {L(a)}, {L(b)}, {t}, {L(want)}, {L(tr)})')
# log_softmax
for i in range(6):
    B,C=rnd.randint(1,4),rnd.randint(2,6); z=[rnd.uniform(-30,30) for _ in range(B*C)]
    w=torch.log_softmax(torch.tensor(z).reshape(B,C),dim=1).flatten().tolist()
    out.append(f'check lsm_ref({B}, {C}, {L(z)}, {L(w)})')
# nets
ACT={3:torch.relu,4:torch.tanh,5:torch.sigmoid}
def tnet(sh,a,lossk,xs,ys,p):
    B,i,h,o=sh
    ps=[]; q=0
    for shape in [(h,i),(h,),(o,h),(o,)]:
        n=1
        for d in shape: n*=d
        ps.append(torch.tensor(p[q:q+n]).reshape(shape).requires_grad_(True)); q+=n
    return ps
def forward(ps,a,lossk,x,y):
    hh=ACT[a](x@ps[0].T+ps[1]); o=hh@ps[2].T+ps[3]
    return torch.nn.functional.cross_entropy(o,y) if lossk==0 else torch.nn.functional.mse_loss(o,y)
nets=[]
for idx in range(10):
    B,i,h,o=rnd.randint(2,6),rnd.randint(2,4),rnd.randint(2,6),rnd.randint(2,4)
    a=[3,4,5][idx%3]; lossk=idx%2
    xs=[rnd.uniform(-2,2) for _ in range(B*i)]
    ys=[float(rnd.randrange(o)) for _ in range(B)] if lossk==0 else [rnd.uniform(-1,1) for _ in range(B*o)]
    p=[rnd.uniform(-1,1) for _ in range(h*i+h+o*h+o)]
    sh=[B,i,h,o]; nets.append((sh,a,lossk,xs,ys,p))
    x=torch.tensor(xs).reshape(B,i); y=torch.tensor([int(v) for v in ys]) if lossk==0 else torch.tensor(ys).reshape(B,o)
    ps=tnet(sh,a,lossk,xs,ys,p); l=forward(ps,a,lossk,x,y); l.backward()
    g=[v for t in ps for v in t.grad.flatten().tolist()]
    out.append(f'check net_ref({LI(sh)}, {a}, {lossk}, {L(xs)}, {L(ys)}, {L(p)}, {lit(l.item())}, {L(g)})')
    out.append(f'check net_fd({LI(sh)}, {a}, {lossk}, {L(xs)}, {L(ys)}, {L(p)}, {rnd.randrange(len(p))})')
# the wider op set (RMSNorm, softplus, gating product, residual sum)
F=torch.nn.functional
for idx in range(6):
    B,i,h,o=rnd.randint(1,4),rnd.randint(1,4),rnd.randint(2,5),rnd.randint(1,3)
    xs=[rnd.uniform(-2,2) for _ in range(B*i)]; ys=[rnd.uniform(-1,1) for _ in range(B*o)]
    sizes=[h*i,h,h,h*h,h,o*h,o]; p=[rnd.uniform(-1,1) for _ in range(sum(sizes))]
    shapes=[(h,i),(h,),(h,),(h,h),(h,),(o,h),(o,)]
    ps=[]; q=0
    for shp,nn in zip(shapes,sizes):
        ps.append(torch.tensor(p[q:q+nn]).reshape(shp).requires_grad_(True)); q+=nn
    W1,b1,g,Wg,bg,W2,b2=ps
    x=torch.tensor(xs).reshape(B,i); y=torch.tensor(ys).reshape(B,o)
    h1=x@W1.T+b1; r=F.rms_norm(h1,(h,),g,1e-5); m=F.softplus(r)*torch.sigmoid(r@Wg.T+bg); res=m+h1
    l=F.mse_loss(res@W2.T+b2,y); l.backward()
    gr=[v for t in ps for v in t.grad.flatten().tolist()]
    out.append(f'check net2_ref({LI([B,i,h,o])}, {L(xs)}, {L(ys)}, {L(p)}, {lit(l.item())}, {L(gr)})')
    out.append(f'check net2_fd({LI([B,i,h,o])}, {L(xs)}, {L(ys)}, {L(p)}, {rnd.randrange(len(p))})')
# training
for idx,(sh,a,lossk,xs,ys,p) in enumerate(nets[:8]):
    B,i,h,o=sh; opt=idx%2; lr=0.1 if opt==0 else 0.01; mom=0.9 if opt==0 else 0.0; wd=[0.0,1e-3][(idx//2)%2]; steps=12
    x=torch.tensor(xs).reshape(B,i); y=torch.tensor([int(v) for v in ys]) if lossk==0 else torch.tensor(ys).reshape(B,o)
    ps=tnet(sh,a,lossk,xs,ys,p)
    O=torch.optim.SGD(ps,lr=lr,momentum=mom,weight_decay=wd) if opt==0 else torch.optim.Adam(ps,lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=wd)
    for _ in range(steps):
        O.zero_grad(); forward(ps,a,lossk,x,y).backward(); O.step()
    w=[v for t in ps for v in t.detach().flatten().tolist()]
    out.append(f'check train_ref({LI(sh)}, {a}, {lossk}, {L(xs)}, {L(ys)}, {L(p)}, {opt}, {lit(lr)}, {lit(mom)}, {lit(wd)}, {steps}, {L(w)})')
sh,a,lossk,xs,ys,p=nets[0]
out.append(f'check train_twice({LI(sh)}, {L(xs)}, {L(ys)}, {L(p)})')
# f32 bits
vals=[0.1,-0.1,1.0,-2.5,3.4028234663852886e38,1.1754943508222875e-38,1.401298464324817e-45,-7.006492321624085e-45,6.0e-39,123456.789,-1e-30,0.333333]
bits=[struct.unpack('<I',struct.pack('<f',v))[0] for v in vals]
out.append(f'check bits_ok({L(vals)}, {LI(bits)})')
out.append('check wmul(18446744073709551615, 18446744073709551615) == 1')
out.append('check wmul(11400714819323198485, 1) == 11400714819323198485')
out.append('check wmul(1099511628211, 14695981039346656037) == %d' % ((1099511628211*14695981039346656037)%2**64))
# raw f64 bits, signed zero included
b64=[1.5*2.0**-1023,0.0,-0.0,1.0,-1.0,0.1,-2.5e-310,5e-324,-5e-324,2.2250738585072014e-308,1.7976931348623157e308,-1e-300,123456.789]
out.append(f'check bits64_ok({L(b64)}, {U(b64)})')
out.append('check f64_bits(-0.0) == 9223372036854775808 and f64_bits(0.0) == 0')
out.append('check relu_zero_grad()')
out.append('check not same_bits(-0.0, 0.0) and same_bits(0.0, 0.0) and -0.0 == 0.0')
out.append('check signbit(0.0 * -1.0) and not signbit(0.0) and signbit(-5.0e-324)')
out.append('check f32_bits(-0.0 as f32) == 2147483648 and f32_bits(0.0 as f32) == 0')
a,b=FX.probe_vector()
fin=[x for x in a if math.isfinite(x)]
lens=list(range(65))+[len(fin)]
probe=[
 'check reference_digest("f32.rmsnorm_tree") == "ddeeb58c50f5ae9f"',
 'check reference_digest("f32.mm") == "2a5bf3731421828e"',
 'check reference_digest("softmax") == ""',
]
for c in FX.CLASSES: probe.append(f'check reference_digest("{c}") == "{FX.digest(FX.reference(c))}"')
probe.append(f'check probe_sums({L(fin)}, {LI(lens)}, {U([FX.fixed_tree_sum(fin[:k]) for k in lens])})')
for oi,op in enumerate(['add','sub','mul','div','neg','sqrt']):
    A=[];B=[];W=[]
    for i in range(len(a)):
        x,y=a[i],b[i]
        if not (math.isfinite(x) and math.isfinite(y)): continue
        r=FX.lane(op,x,y)
        if not math.isfinite(r) or (op=='sqrt' and x<0) or (op=='div' and y==0): continue
        A.append(x);B.append(y);W.append(r)
    probe.append(f'check probe_lanes({oi}, {L(A)}, {L(B)}, {U(W)})')
MM='"2a5bf3731421828e"'; RM='"ddeeb58c50f5ae9f"'; S='"'+FX.digest(FX.reference('sum.f64'))+'"'
probe+=[
 'check place_of("f32.mm", 768.0, [], []) == "cpu"',
 f'check place_of("f32.mm", 768.0, [cal("ferric-metal", "f32.mm", 2.0, 50.0)], [ex("ferric-metal", "f32.mm", {MM})]) == "ferric-metal"',
 f'check place_of("f32.mm", 768.0, [cal("ferric-metal", "f32.mm", 2.0, 50.0)], []) == "cpu"',
 f'check place_of("f32.mm", 768.0, [], [ex("ferric-metal", "f32.mm", {MM})]) == "cpu"',
 'check place_of("f32.mm", 768.0, [cal("ferric-metal", "f32.mm", 2.0, 50.0)], [ex("ferric-metal", "f32.mm", "2a5bf3731421828f")]) == "cpu"',
 f'check place_of("f32.mm", 768.0, [cal("ferric-metal", "f32.mm", 2.0, 50.0)], [ex("ferric-metal", "f32.rmsnorm_tree", {MM})]) == "cpu"',
 f'check place_of("f32.mm", 4.0, [cal("ferric-metal", "f32.mm", 2000.0, 50.0)], [ex("ferric-metal", "f32.mm", {MM})]) == "cpu"',
 f'check place_of("f32.mm", 768.0, [cal("ferric-metal", "f32.mm", 2.0, 0.0)], [ex("ferric-metal", "f32.mm", {MM})]) == "cpu"',
 'check place_of("softmax", 768.0, [cal("ferric-metal", "softmax", 2.0, 50.0)], [ex("ferric-metal", "softmax", "")]) == "cpu"',
 f'check place_of("f32.rmsnorm_tree", 768.0, [cal("ferric-vulkan", "f32.rmsnorm_tree", 2.0, 50.0), cal("ferric-metal", "f32.rmsnorm_tree", 2.0, 50.0)], [ex("ferric-vulkan", "f32.rmsnorm_tree", {RM}), ex("ferric-metal", "f32.rmsnorm_tree", {RM})]) == "ferric-metal"',
 f'check place_of("f32.rmsnorm_tree", 768.0, [cal("ferric-vulkan", "f32.rmsnorm_tree", 1.0, 50.0), cal("ferric-metal", "f32.rmsnorm_tree", 2.0, 50.0)], [ex("ferric-vulkan", "f32.rmsnorm_tree", {RM}), ex("ferric-metal", "f32.rmsnorm_tree", {RM})]) == "ferric-vulkan"',
 f'check place_of("sum.f64", 1.0e6, [cal("ferric-metal", "sum.f64", 2.0, 50.0)], [Exact(fabric: "ferric-metal", class: "sum.f64", probe: "0000", reference: {S})]) == "cpu"',
 f'check place_of("sum.f64", 1.0e6, [cal("gpu.good", "sum.f64", 2.0, 50.0)], [ex("gpu.good", "sum.f64", {S})]) == "gpu.good"',
 f'check place_of("sum.f64", 1.0e6, [cal("gpu.good", "sum.f64", 2.0, 50.0)], [ex("gpu.good", "sum.f64", "{"0"*64}")]) == "cpu"',
 'check place_of("sum.f64", 16.0, [cal("gpu.tie", "sum.f64", 0.0, 0.5)], [ex("gpu.tie", "sum.f64", ' + S + ')]) == "cpu"',
 'check lanes_class(2) == "lanes.f64.mul"',
]
head=open(os.path.join(ROOT,'scripts/gen/ml_tensor_head.rce')).read()
# shard: cheap checks in _checks, nets/training spread over shards, probe digests alone
cheap=[l for l in out if not l.startswith(('check net_','check train_','check net2_'))]
heavy=[l for l in out if l.startswith(('check net_','check train_','check net2_'))]
dig=[l for l in probe if 'reference_digest' in l]; fin_=[l for l in probe if 'probe_' in l]; pl=[l for l in probe if l not in dig and l not in fin_]
files={'tests/ml/tensor_checks.rce':cheap,'tests/ml/tensor_checks_probe.rce':dig,'tests/ml/tensor_checks_probe_1.rce':fin_,'tests/ml/tensor_checks_place.rce':pl}
for s in range(3): files[f'tests/ml/tensor_checks_{s}.rce']=heavy[s::3]
for f,ls in files.items(): open(os.path.join(ROOT,f),'w').write(head+'\n'+'\n'.join(ls)+'\n')
print('checks',len(out)+len(probe))
