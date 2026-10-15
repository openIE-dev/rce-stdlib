"""Independent Python model of lib/ml/ad.rce elementary functions and
tape (same operation order; CPython floats are IEEE 754 binary64 with no
contraction). Used by scripts/gen/ml_gen.py to emit bit-exact vectors."""
import math
LN2_HI=0.6931471803691238; LN2_LO=1.9082149292705877e-10; INV_LN2=1.4426950408889634

def pow2(k): return math.ldexp(1.0,k)
def ldexp(x,k):
    if k>1000: return x*pow2(1000)*pow2(k-1000)
    if k<-1000: return x*pow2(-1000)*pow2(k+1000)
    return x*pow2(k)
def ilogb(x): return math.frexp(abs(x))[1]-1
import mpmath
mpmath.mp.prec=400
def _cr(v):
    """The binary64 nearest to the mpf v, ties to even (subnormals included)."""
    if v==0: return 0.0
    a=abs(v); e=int(mpmath.floor(mpmath.log(a,2)))
    q=max(e,-1022)-52
    m=a*mpmath.mpf(2)**(-q); n=int(mpmath.floor(m)); fr=m-n
    if fr>0.5 or (fr==0.5 and n%2==1): n+=1
    r=math.ldexp(float(n),q)
    return r if v>0 else -r
# exp, ln, sin, cos: correctly rounded (as lib/math/math.rce), from
# mpmath at 400 bits; no tested argument is near a hard case at that
# precision (CORE-MATH's worst cases need fewer than 160 bits).
def exp(x): return _cr(mpmath.exp(mpmath.mpf(x)))
def ln(x):
    assert x>0
    return _cr(mpmath.log(mpmath.mpf(x)))
def _trig(f,x):
    # the reduction needs about log2|x| more bits than the result
    with mpmath.workprec(400+max(0,math.frexp(x)[1])):
        return _cr(f(mpmath.mpf(x)))
def sin(x): return _trig(mpmath.sin,x)
def cos(x): return _trig(mpmath.cos,x)
def expm1(x):
    if abs(x)<0.35:
        p=1.0
        for n in range(18,1,-1): p=1.0+x*p/float(n)
        return x*p
    return exp(x)-1.0
def log1p(x):
    u=1.0+x
    if u==1.0: return x
    return ln(u)*(x/(u-1.0))
def tanh(x):
    a=abs(x)
    if a>22.0: return -1.0 if x<0 else 1.0
    e=expm1(2.0*a); t=e/(e+2.0)
    return 0.0-t if x<0 else t
def sigmoid(x):
    if x>=0.0: return 1.0/(1.0+exp(0.0-x))
    e=exp(x); return e/(1.0+e)
def softplus(x):
    if x>36.0: return x+exp(0.0-x)
    if x< -36.0: return exp(x)
    if x>0.0: return x+log1p(exp(0.0-x))
    return log1p(exp(x))
def powi(x,n):
    r=1.0; m=abs(n); b=x
    while m>0:
        if m%2==1: r=r*b
        m//=2
        if m>0: b=b*b
    return 1.0/r if n<0 else r

# ---- model of rand.det streams (as scripts/gen/rand_gen.py) and ml.prob draws
M32=0xffffffff
def _rotl(x,n): return ((x<<n)|(x>>(32-n)))&M32
def _qr(v,a,b,c,d):
    v[a]=(v[a]+v[b])&M32; v[d]=_rotl(v[d]^v[a],16); v[c]=(v[c]+v[d])&M32; v[b]=_rotl(v[b]^v[c],12)
    v[a]=(v[a]+v[b])&M32; v[d]=_rotl(v[d]^v[a],8); v[c]=(v[c]+v[d])&M32; v[b]=_rotl(v[b]^v[c],7)
def _block(kw,ctr,nw):
    s=[1634760805,857760878,2036477234,1797285236]+list(kw)+[ctr&M32]+list(nw); x=list(s)
    for _ in range(10):
        _qr(x,0,4,8,12);_qr(x,1,5,9,13);_qr(x,2,6,10,14);_qr(x,3,7,11,15)
        _qr(x,0,5,10,15);_qr(x,1,6,11,12);_qr(x,2,7,8,13);_qr(x,3,4,9,14)
    return [(x[i]+s[i])&M32 for i in range(16)]
class Rng:
    def __init__(s,seed,stream):
        s.key=[int.from_bytes(bytes(seed[i:i+4]),'little') for i in range(0,32,4)]; s.stream=stream; s.block=0; s.buf=[]; s.pos=16
    def u32(s):
        if s.pos>=16:
            s.buf=_block(s.key,s.block&M32,[s.stream&M32,s.stream>>32,s.block>>32]); s.block+=1; s.pos=0
        v=s.buf[s.pos]; s.pos+=1; return v
    def uniform(s):
        a=s.u32()>>5; b=s.u32()>>6
        return (float(a)*67108864.0+float(b)+0.5)*pow2(-53)
    def normal(s):
        u1=s.uniform(); u2=s.uniform()
        return math.sqrt(-2.0*ln(u1))*cos(6.283185307179586*u2)

# ---- atan, pow (fdlibm s_atan.c order, as lib/ml/ad.rce)
AT=[0.3333333333333293,-0.19999999999876483,0.14285714272503466,-0.11111110405462356,0.09090887133436507,-0.0769187620504483,
    0.06661073137387531,-0.058335701337905735,0.049768779946159324,-0.036531572744216916,0.016285820115365782]
ATHI=[0.4636476090008061,0.7853981633974483,0.982793723247329,1.5707963267948966]
ATLO=[2.2698777452961687e-17,3.061616997868383e-17,1.3903311031230998e-17,6.123233995736766e-17]
def atan(x0):
    a=abs(x0)
    if a>=7.378697629483821e+19:
        r=ATHI[3]+ATLO[3]; return 0.0-r if x0<0 else r
    x=x0; i=-1
    if a<0.4375:
        if a<1.862645149230957e-09: return x0
    else:
        x=a
        if a<1.1875:
            if a<0.6875: i=0; x=(2.0*x-1.0)/(2.0+x)
            else: i=1; x=(x-1.0)/(x+1.0)
        elif a<2.4375: i=2; x=(x-1.5)/(1.0+1.5*x)
        else: i=3; x=-1.0/x
    z=x*x; w=z*z
    s1=z*(AT[0]+w*(AT[2]+w*(AT[4]+w*(AT[6]+w*(AT[8]+w*AT[10])))))
    s2=w*(AT[1]+w*(AT[3]+w*(AT[5]+w*(AT[7]+w*AT[9]))))
    if i<0: return x-x*(s1+s2)
    r=ATHI[i]-((x*(s1+s2)-ATLO[i])-x)
    return 0.0-r if x0<0 else r
def pow(x,y): return exp(y*ln(x))

# ---- a model of the ml.ad Tape, node for node, for nested AD
def stepv(x,tie):
    if x>0.0: return 1.0
    if x<0.0: return 0.0
    return 0.5 if tie==1 else 0.0
class Tape:
    UN={'neg':lambda x:0.0-x,'exp':exp,'ln':ln,'sqrt':math.sqrt,'tanh':tanh,'sigmoid':sigmoid,'sin':sin,'cos':cos,
        'relu':lambda x: x if x>0.0 else 0.0,'softplus':softplus,'log1p':log1p,'expm1':expm1,'recip':lambda x:1.0/x,
        'abs':abs,'atan':atan}
    BIN={'add':lambda a,b:a+b,'sub':lambda a,b:a-b,'mul':lambda a,b:a*b,'div':lambda a,b:a/b,
         'max':lambda a,b:a if a>=b else b,'min':lambda a,b:a if a<=b else b,'pow':pow}
    def __init__(s): s.op=[];s.a=[];s.b=[];s.k=[];s.val=[];s.inputs=[]
    def push(s,o,a,b,k,v):
        s.op.append(o);s.a.append(a);s.b.append(b);s.k.append(k);s.val.append(v); return len(s.val)-1
    def input(s,x):
        i=s.push('in',-1,-1,0,x); s.inputs.append(i); return i
    def const(s,x): return s.push('c',-1,-1,0,x)
    def un(s,o,i): return s.push(o,i,-1,0,Tape.UN[o](s.val[i]))
    def bin(s,o,i,j): return s.push(o,i,j,0,Tape.BIN[o](s.val[i],s.val[j]))
    def powi(s,i,n): return s.push('powi',i,-1,n,powi(s.val[i],n))
    def step(s,i,tie): return s.push('step',i,-1,tie,stepv(s.val[i],tie))
    def partials(s,i):
        o=s.op[i]; y=s.val[i]; x=s.val[s.a[i]] if s.a[i]>=0 else 0.0; z=s.val[s.b[i]] if s.b[i]>=0 else 0.0
        if o=='add': return 1.0,1.0
        if o=='sub': return 1.0,-1.0
        if o=='mul': return z,x
        if o=='div': return 1.0/z,0.0-y/z
        if o=='neg': return -1.0,0.0
        if o=='exp': return y,0.0
        if o=='ln': return 1.0/x,0.0
        if o=='sqrt': return 0.5/y,0.0
        if o=='tanh': return 1.0-y*y,0.0
        if o=='sigmoid': return y*(1.0-y),0.0
        if o=='sin': return cos(x),0.0
        if o=='cos': return 0.0-sin(x),0.0
        if o=='powi': return float(s.k[i])*powi(x,s.k[i]-1),0.0
        if o=='relu': return (1.0 if x>0.0 else 0.0),0.0
        if o=='softplus': return sigmoid(x),0.0
        if o=='log1p': return 1.0/(1.0+x),0.0
        if o=='expm1': return y+1.0,0.0
        if o=='recip': return 0.0-y*y,0.0
        if o=='abs': return stepv(x,0)-stepv(0.0-x,0),0.0
        if o=='max': return stepv(x-z,1),stepv(z-x,1)
        if o=='min': return stepv(z-x,1),stepv(x-z,1)
        if o=='pow': return (z*y)/x,y*ln(x)
        if o=='atan': return 1.0/(1.0+x*x),0.0
        return 0.0,0.0
    def adjoints(s,out):
        g=[0.0]*len(s.val); g[out]=1.0
        for u in range(out,-1,-1):
            gi=g[u]
            if gi!=0.0 and s.a[u]>=0 and s.op[u]!='step':
                pa,pb=s.partials(u); g[s.a[u]]=g[s.a[u]]+gi*pa
                if s.b[u]>=0: g[s.b[u]]=g[s.b[u]]+gi*pb
        return g
    def grad(s,out):
        g=s.adjoints(out); return [g[i] for i in s.inputs]
    def jvp(s,out,d0):
        d=[0.0]*len(s.val)
        for q,i in enumerate(s.inputs): d[i]=d0[q]
        for u in range(len(s.val)):
            if s.a[u]>=0:
                pa,pb=s.partials(u); v=pa*d[s.a[u]]
                if s.b[u]>=0: v=v+pb*d[s.b[u]]
                d[u]=v
        return d[out]
    def partial_nodes(s,u,c1,cm1,ch):
        o=s.op[u]; a=s.a[u]; b=s.b[u]
        if o=='add': return c1,c1
        if o=='sub': return c1,cm1
        if o=='mul': return b,a
        if o=='div':
            pa=s.bin('div',c1,b); q=s.bin('div',u,b); return pa,s.un('neg',q)
        if o=='neg': return cm1,-1
        if o=='exp': return u,-1
        if o=='ln': return s.bin('div',c1,a),-1
        if o=='sqrt': return s.bin('div',ch,u),-1
        if o=='tanh': sq=s.bin('mul',u,u); return s.bin('sub',c1,sq),-1
        if o=='sigmoid': om=s.bin('sub',c1,u); return s.bin('mul',u,om),-1
        if o=='sin': return s.un('cos',a),-1
        if o=='cos': sn=s.un('sin',a); return s.un('neg',sn),-1
        if o=='powi':
            kk=s.k[u]; ck=s.const(float(kk)); pw=s.powi(a,kk-1); return s.bin('mul',ck,pw),-1
        if o=='relu': return s.step(a,0),-1
        if o=='softplus': return s.un('sigmoid',a),-1
        if o=='log1p': den=s.bin('add',c1,a); return s.bin('div',c1,den),-1
        if o=='expm1': return s.bin('add',u,c1),-1
        if o=='recip': sq=s.bin('mul',u,u); return s.un('neg',sq),-1
        if o=='abs':
            sp=s.step(a,0); na=s.un('neg',a); sm=s.step(na,0); return s.bin('sub',sp,sm),-1
        if o=='max':
            d1=s.bin('sub',a,b); d2=s.bin('sub',b,a); return s.step(d1,1),s.step(d2,1)
        if o=='min':
            d1=s.bin('sub',b,a); d2=s.bin('sub',a,b); return s.step(d1,1),s.step(d2,1)
        if o=='pow':
            zy=s.bin('mul',b,u); pa=s.bin('div',zy,a); la=s.un('ln',a); return pa,s.bin('mul',u,la)
        if o=='atan':
            sq=s.bin('mul',a,a); den=s.bin('add',c1,sq); return s.bin('div',c1,den),-1
        return -1,-1
    def copy(s):
        t=Tape(); t.op=s.op[:];t.a=s.a[:];t.b=s.b[:];t.k=s.k[:];t.val=s.val[:];t.inputs=s.inputs[:]; return t
    def grad_tape(s,out):
        t=s.copy(); n=len(t.val); g=[-1]*n
        c1=t.const(1.0); cm1=t.const(-1.0); ch=t.const(0.5); g[out]=c1
        def acc(j,c):
            if g[j]<0: g[j]=c
            else: g[j]=t.bin('add',g[j],c)
        for u in range(out,-1,-1):
            gi=g[u]
            if gi>=0 and t.a[u]>=0 and t.op[u]!='step':
                pa,pb=t.partial_nodes(u,c1,cm1,ch); ja=t.a[u]; jb=t.b[u]
                ca=t.bin('mul',gi,pa); acc(ja,ca)
                if jb>=0:
                    cb=t.bin('mul',gi,pb); acc(jb,cb)
        z=t.const(0.0)
        return t,[g[k] if g[k]>=0 else z for k in t.inputs]
    def hessian(s,out):
        t,gs=s.grad_tape(out); return [t.grad(gi) for gi in gs]
    def hvp(s,out,v):
        t,gs=s.grad_tape(out); return [t.jvp(gi,v) for gi in gs]
