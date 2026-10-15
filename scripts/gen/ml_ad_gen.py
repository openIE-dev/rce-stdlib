"""Generate tests/ml/ad_checks*.rce for lib/ml/ad.rce.

Each test program is a straight-line SSA list (PROGS). From it this script
emits (a) an RCE builder on the ml.ad Tape, (b) an independent Python model
of the tape (scripts/ref/ml/mlref.py, same rounding order) for bit-exact
values, gradients and JVPs, and (c) PyTorch 2.9 float64 autograd gradients
(after torch.autograd.gradcheck passes on the same program) cross-checked
against JAX 0.6 float64 jax.grad. Run with /workspace/venv310/bin/python.
Writes vectors/ml/ad_ref.tsv (program, point, inputs, torch grad, jax grad).
"""
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ref', 'ml'))
import mlref as M
import torch
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

def P(*ops): return list(ops)
# ('in',) ('c', v) (op, i[, j]) ('powi', i, n)
PROGS = [
  ('xy_sin', 2, P(('in',),('in',),('mul',0,1),('sin',0),('add',2,3))),
  ('exp_over', 2, P(('in',),('in',),('c',1.0),('powi',1,2),('add',2,3),('exp',0),('div',5,4))),
  ('tanh_log1p', 3, P(('in',),('in',),('in',),('mul',0,1),('sub',3,2),('tanh',4),('powi',0,2),('log1p',6),('add',5,7))),
  ('sqrt_softplus', 3, P(('in',),('in',),('in',),('softplus',0),('powi',1,2),('add',3,4),('sqrt',5),('sigmoid',2),('mul',6,7))),
  ('cos_gauss_relu', 3, P(('in',),('in',),('in',),('cos',0),('powi',1,2),('c',-0.5),('mul',4,5),('exp',6),('mul',3,7),('sub',2,0),('relu',9),('sub',8,10),('exp',1),('c',1.0),('add',12,13),('ln',14),('add',11,15))),
  ('neg_div_chain', 2, P(('in',),('in',),('neg',0),('exp',2),('c',2.0),('add',3,4),('div',1,5),('tanh',6),('sin',7),('exp',8),('powi',9,-3))),
]
def rosen(n):
    ops=[('in',)]*n; c1=len(ops); ops.append(('c',1.0)); c100=len(ops); ops.append(('c',100.0)); terms=[]
    for i in range(n-1):
        a=len(ops); ops.append(('powi',i,2)); b=len(ops); ops.append(('sub',i+1,a)); c=len(ops); ops.append(('powi',b,2))
        d=len(ops); ops.append(('mul',c100,c)); e=len(ops); ops.append(('sub',c1,i)); f=len(ops); ops.append(('powi',e,2))
        g=len(ops); ops.append(('add',d,f)); terms.append(g)
    acc=terms[0]
    for t in terms[1:]:
        ops.append(('add',acc,t)); acc=len(ops)-1
    return ops
PROGS.append(('rosenbrock4', 4, rosen(4)))
def mlp():
    # inputs: x0 x1, w00 w01 w10 w11 w20 w21, b0 b1 b2
    ops=[('in',)]*11; hs=[]
    for j in range(3):
        a=len(ops); ops.append(('mul',2+2*j,0)); b=len(ops); ops.append(('mul',3+2*j,1))
        c=len(ops); ops.append(('add',a,b)); d=len(ops); ops.append(('add',c,8+j)); e=len(ops); ops.append(('tanh',d))
        f=len(ops); ops.append(('exp',e)); hs.append(f)
    s=len(ops); ops.append(('add',hs[0],hs[1])); s2=len(ops); ops.append(('add',s,hs[2])); ops.append(('ln',s2))
    return ops
PROGS.append(('mlp_logsumexp', 11, mlp()))
def deep():
    ops=[('in',)]; cur=0; fs=['tanh','exp','sin','sigmoid','cos','softplus','log1p']
    for i in range(21):
        ops.append((fs[i%7],cur)); cur=len(ops)-1
    return ops
PROGS.append(('deep21', 1, deep()))
# the wider op set: expm1, recip, abs, max, min, pow, atan
PROGS.append(('atan_pow', 3, P(('in',),('in',),('in',),('atan',0),('exp',1),('pow',4,2),('mul',3,5),('expm1',2),('recip',4),('mul',7,8),('add',6,9))))
PROGS.append(('abs_max_min', 3, P(('in',),('in',),('in',),('abs',0),('max',3,1),('mul',0,1),('expm1',2),('min',5,6),('sub',4,7),('powi',8,2),('atan',8),('add',9,10))))

UN={'neg':lambda x:0.0-x,'exp':M.exp,'ln':M.ln,'sqrt':math.sqrt,'tanh':M.tanh,'sigmoid':M.sigmoid,'sin':M.sin,'cos':M.cos,
    'relu':lambda x: x if x>0.0 else 0.0,'softplus':M.softplus,'log1p':M.log1p}
BIN={'add':lambda a,b:a+b,'sub':lambda a,b:a-b,'mul':lambda a,b:a*b,'div':lambda a,b:a/b,'max':0,'min':0,'pow':0}

def pytape(ops, xs):
    t=M.Tape(); q=0
    for o in ops:
        if o[0]=='in': t.input(xs[q]); q+=1
        elif o[0]=='c': t.const(o[1])
        elif o[0]=='powi': t.powi(o[1],o[2])
        elif o[0] in BIN: t.bin(o[0],o[1],o[2])
        else: t.un(o[0],o[1])
    return t

def model_grad(ops, xs):
    t=pytape(ops,xs); n=len(t.val)-1
    return t.val[n],t.grad(n)
def model_jvp(ops, xs, dr):
    t=pytape(ops,xs); return t.jvp(len(t.val)-1,dr)

def tfun(ops, lib):
    T = lib=='torch'
    E = (torch if T else jnp)
    U={'neg':lambda x:-x,'exp':E.exp,'ln':E.log,'sqrt':E.sqrt,'tanh':E.tanh,'sin':E.sin,'cos':E.cos,'log1p':E.log1p,
       'sigmoid':(torch.sigmoid if T else jax.nn.sigmoid),'relu':(torch.relu if T else jax.nn.relu),
       'softplus':(lambda x: torch.log1p(torch.exp(x))) if T else (lambda x: jnp.logaddexp(x,0.0)),
       'expm1':E.expm1,'recip':lambda x:1.0/x,'abs':E.abs,'atan':(torch.atan if T else jnp.arctan)}
    B2={'add':lambda a,b:a+b,'sub':lambda a,b:a-b,'mul':lambda a,b:a*b,'div':lambda a,b:a/b,
        'max':(torch.maximum if T else jnp.maximum),'min':(torch.minimum if T else jnp.minimum),'pow':lambda a,b:a**b}
    def f(*xs):
        v=[]; q=0
        for o in ops:
            if o[0]=='in': v.append(xs[q]); q+=1
            elif o[0]=='c': v.append(o[1] if not T else torch.tensor(o[1],dtype=torch.float64))
            elif o[0]=='powi': v.append(v[o[1]]**o[2])
            elif o[0] in BIN: v.append(B2[o[0]](v[o[1]],v[o[2]]))
            else: v.append(U[o[0]](v[o[1]]))
        return v[-1]
    return f

def lit(x):
    r=repr(float(x))
    return r if ('.' in r or 'e' in r) else r+'.0'
def L(xs): return '['+', '.join(lit(x) for x in xs)+']'

def rce_builder(k, name, ops):
    out=[f'# {name}', f'fn build_{k}(t: &mut Tape, x: &[f64]) -> i64']
    q=0
    fn={'add':'add','sub':'sub','mul':'mul','div':'div','neg':'neg','exp':'texp','ln':'tln','sqrt':'tsqrt','tanh':'ttanh','sigmoid':'tsigmoid',
        'sin':'tsin','cos':'tcos','relu':'trelu','softplus':'tsoftplus','log1p':'tlog1p','expm1':'texpm1','recip':'trecip',
        'abs':'tabs','max':'tmax','min':'tmin','pow':'tpow','atan':'tatan'}
    for i,o in enumerate(ops):
        if o[0]=='in': e=f'input(t, x[{q}])'; q+=1
        elif o[0]=='c': e=f'constant(t, {lit(o[1])})'
        elif o[0]=='powi': e=f'tpowi(t, n{o[1]}, {o[2]})'
        elif o[0] in BIN: e=f'{fn[o[0]]}(t, n{o[1]}, n{o[2]})'
        else: e=f'{fn[o[0]]}(t, n{o[1]})'
        out.append(f'    let n{i} = {e}')
    out.append(f'    return n{len(ops)-1}')
    return '\n'.join(out)

def main(root):
    rnd=random.Random(20261015)
    head=open(os.path.join(root,'scripts/gen/ml_ad_head.rce')).read()
    builders='\n\n'.join(rce_builder(k,n,o) for k,(n,_,o) in enumerate(PROGS))
    disp=['fn build(k: i64, t: &mut Tape, x: &[f64]) -> i64']
    for k in range(len(PROGS)):
        disp.append(f'    if k == {k}\n        return build_{k}(t, x)')
    disp.append('    return build_0(t, x)')
    body=head+'\n'+builders+'\n\n'+'\n'.join(disp)+'\n'
    tsv=['prog\tpoint\tinputs\ttorch_grad\tjax_grad\tmax_rel_torch_jax']
    shards={}; hshards={}; hworst=[0.0]
    worst=0.0
    for k,(name,n,ops) in enumerate(PROGS):
        tf=tfun(ops,'torch'); jf=tfun(ops,'jax')
        npts=24
        for p in range(npts):
            lo,hi=(-1.5,1.5) if name!='rosenbrock4' else (-2.0,2.0)
            xs=[rnd.uniform(lo,hi) for _ in range(n)]
            dr=[rnd.uniform(-1,1) for _ in range(n)]
            ts=[torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in xs]
            if p<4:
                assert torch.autograd.gradcheck(tf,tuple(ts),eps=1e-6,atol=1e-7,rtol=1e-6), name
            y=tf(*ts); y.backward(); tg=[t.grad.item() for t in ts]
            jg=[float(g) for g in jax.grad(jf,argnums=tuple(range(n)))(*[jnp.float64(x) for x in xs])]
            rel=max(abs(a-b)/max(1.0,abs(b)) for a,b in zip(tg,jg)); worst=max(worst,rel)
            assert rel<1e-12,(name,p,rel)
            mv,mg=model_grad(ops,xs); mj=model_jvp(ops,xs,dr)
            assert abs(mv-y.item())<=1e-13*max(1,abs(mv)),(name,mv,y.item())
            tsv.append(f'{name}\t{p}\t{L(xs)}\t{L(tg)}\t{L(jg)}\t{rel:.3e}')
            sh=shards.setdefault(k%4,[])
            sh.append(f'check value_bits({k}, {L(xs)}, {lit(mv)})')
            sh.append(f'check grad_bits({k}, {L(xs)}, {L(mg)})')
            sh.append(f'check jvp_bits({k}, {L(xs)}, {L(dr)}, {lit(mj)})')
            sh.append(f'check grad_ref({k}, {L(xs)}, {L(tg)})')
            sh.append(f'check grad_fd({k}, {L(xs)})')
            sh.append(f'check jvp_dot({k}, {L(xs)}, {L(dr)})')
            if p<6:
                xt=torch.tensor(xs,dtype=torch.float64)
                TH=torch.autograd.functional.hessian(lambda v: tf(*v.unbind()), xt).tolist()
                JH=[[float(a) for a in r] for r in jax.hessian(lambda v: jf(*[v[i] for i in range(n)]))(jnp.array(xs,dtype=jnp.float64))]
                hrel=max(abs(TH[i][j]-JH[i][j])/max(1.0,abs(JH[i][j])) for i in range(n) for j in range(n)); hworst[0]=max(hworst[0],hrel)
                assert hrel<1e-11,(name,hrel)
                pt=pytape(ops,xs); out_=len(pt.val)-1
                MH=pt.hessian(out_); MV=pt.hvp(out_,dr)
                flatH=[TH[i][j] for i in range(n) for j in range(n)]
                tv=[sum(TH[i][j]*dr[j] for j in range(n)) for i in range(n)]
                hs=hshards.setdefault(k%2,[])
                hs.append(f'check hess_bits({k}, {L(xs)}, {L([v for r in MH for v in r])})')
                hs.append(f'check hvp_bits({k}, {L(xs)}, {L(dr)}, {L(MV)})')
                hs.append(f'check hess_ref({k}, {L(xs)}, {L(flatH)}, {L(dr)}, {L(tv)})')
                hs.append(f'check hess_fd({k}, {L(xs)})')
                hs.append(f'check grad_tape_bits({k}, {L(xs)})')
                if p<2 and name in ('xy_sin','atan_pow','tanh_log1p','abs_max_min'):
                    def f00(v):
                        g=torch.autograd.grad(tf(*v.unbind()),v,create_graph=True)[0]
                        return torch.autograd.grad(g[0],v,create_graph=True)[0][0]
                    v=xt.clone().requires_grad_(True); T3=torch.autograd.grad(f00(v),v)[0].tolist()
                    J3=[float(a) for a in jax.grad(lambda v: jax.hessian(lambda u: jf(*[u[i] for i in range(n)]))(v)[0][0])(jnp.array(xs,dtype=jnp.float64))]
                    assert max(abs(a-b)/max(1,abs(b)) for a,b in zip(T3,J3))<1e-11
                    hs.append(f'check third_ref({k}, {L(xs)}, {L(T3)})')
    os.makedirs(os.path.join(root,'vectors/ml'),exist_ok=True)
    open(os.path.join(root,'vectors/ml/ad_ref.tsv'),'w').write('\n'.join(tsv)+'\n')
    # elementary functions: bit-exact against the model, plus the max ulp vs libm noted
    el=[]
    fx={'exp':(M.exp,-745.0,709.0),'ln':(M.ln,None,None),'expm1':(M.expm1,-3.0,3.0),'log1p':(M.log1p,-0.999,20.0),'tanh':(M.tanh,-25.0,25.0),
        'sigmoid':(M.sigmoid,-40.0,40.0),'softplus':(M.softplus,-50.0,50.0),'sin':(M.sin,-1e5,1e5),'cos':(M.cos,-1e5,1e5),'atan':(M.atan,-1e3,1e3)}
    for nm,(f,lo,hi) in fx.items():
        xs=[]
        for i in range(120):
            if nm=='ln': x=math.exp(rnd.uniform(-700,700)) if i%3 else rnd.uniform(0.5,2.0)
            elif i%3==0: x=rnd.uniform(-1,1)*min(1.0,hi)
            elif i%3==1: x=rnd.uniform(-1,1)*1e-6
            else: x=rnd.uniform(lo,hi)
            xs.append(x)
        if nm=='ln': xs+= [5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 1.0000000000000002, 0.9999999999999999]
        if nm=='exp': xs+= [-745.0, 709.78, 1e-300, -1e-300, 0.5*math.log(2)]
        if nm=='atan': xs+=[1e30,-1e30,0.4375,1.1875,0.6875,2.4375,1e-10]
        ys=[f(x) for x in xs]
        el.append(f'# {nm}\ncheck elem_bits({list(fx).index(nm)}, {L(xs)}, {L(ys)})')
    # sin and cos at large arguments (Payne and Hanek, used for every |x| >= pi/4): bits against
    # the model, and within 1 ulp of mpmath at 2000 bits (asserted here)
    import mpmath; mpmath.mp.prec=2500
    big=[rnd.uniform(1,2)*2.0**rnd.randint(21,1023)*rnd.choice([-1.0,1.0]) for _ in range(60)]
    big+=[6381956970095103.0*2.0**797,-6381956970095103.0*2.0**797,5.319372648326541e+255,1.7976931348623157e308,-1.7976931348623157e308,1647099.0,-1647099.0,2.0**60,1e22,-1e22,1e300]
    bigw=0.0
    for nm,f,mf in (('sin',M.sin,mpmath.sin),('cos',M.cos,mpmath.cos)):
        ys=[f(x) for x in big]
        for x,y in zip(big,ys):
            ref=mf(mpmath.mpf(x)); u=mpmath.mpf(2)**(M.ilogb(abs(float(ref)))-52); e=float(abs(mpmath.mpf(y)-ref)/u); bigw=max(bigw,e)
            assert e<1.0,(nm,x,e)
        el.append(f'# {nm}, large |x| (Payne and Hanek)\ncheck elem_bits({list(fx).index(nm)}, {L(big)}, {L(ys)})')
    print('sin/cos large-argument max ulp',bigw)
    ad_ext = ['check dual_matches_tape()', 'check trelu_zero()', 'check ties()', 'check pow2(-1074) == 5e-324', 'check pow2(1023) == 8.98846567431158e+307',
              'check ilogb(5e-324) == -1074', 'check ilogb(1.7976931348623157e+308) == 1023', 'check ilogb(1.0) == 0', 'check ilogb(0.75) == -1']
    files={}
    files['tests/ml/ad_checks.rce']=body+'\n'+'\n'.join(el+ad_ext)+'\n'
    for s,lines in sorted(hshards.items()):
        files[f'tests/ml/ad_checks_hess_{s}.rce']=body+'\n'+'\n'.join(lines)+'\n'
    for s,lines in sorted(shards.items()):
        files[f'tests/ml/ad_checks_{s}.rce']=body+'\n'+'\n'.join(lines)+'\n'
    for p,c in files.items(): open(os.path.join(root,p),'w').write(c)
    print('hess torch-jax max rel',hworst[0],'hess checks',sum(len(v) for v in hshards.values()));print('programs',len(PROGS),'points',sum(len(v) for v in shards.values())//6,'checks',sum(len(v) for v in shards.values())+len(el)+len(ad_ext),'torch-jax max rel',worst)
main(os.path.join(os.path.dirname(__file__),'..','..'))
