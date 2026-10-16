"""Independent Python model of lib/ml/ad.rce elementary functions and
tape (same operation order; CPython floats are IEEE 754 binary64 with no
contraction). Used by scripts/gen/ml_gen.py to emit bit-exact vectors."""
import math
LN2_HI=0.6931471803691238; LN2_LO=1.9082149292705877e-10; INV_LN2=1.4426950408889634
P1=0.16666666666666602; P2=-0.0027777777777015593; P3=6.613756321437934e-05; P4=-1.6533902205465252e-06; P5=4.1381367970572385e-08
LG1=0.6666666666666735; LG2=0.3999999999940942; LG3=0.2857142874366239; LG4=0.22222198432149784; LG5=0.1818357216161805; LG6=0.15313837699209373; LG7=0.14798198605116586
SQRT2=1.4142135623730951
S1=-0.16666666666666632; S2=0.00833333333332249; S3=-0.0001984126982985795; S4=2.7557313707070068e-06; S5=-2.5050760253406863e-08; S6=1.58969099521155e-10
C1=0.0416666666666666; C2=-0.001388888888887411; C3=2.480158728947673e-05; C4=-2.7557314351390663e-07; C5=2.087572321298175e-09; C6=-1.1359647557788195e-11
PIO2_1=1.5707963267341256; PIO2_2=6.077100506303966e-11; PIO2_2T=2.0222662487959506e-21; INV_PIO2=0.6366197723675814

def pow2(k): return math.ldexp(1.0,k)
def ldexp(x,k):
    if k>1000: return x*pow2(1000)*pow2(k-1000)
    if k<-1000: return x*pow2(-1000)*pow2(k+1000)
    return x*pow2(k)
def ilogb(x): return math.frexp(abs(x))[1]-1
def trunc(x): return int(x)
def exp(x):
    if x==0.0: return 1.0
    if x< -745.1332191019412: return 0.0
    half=-0.5 if x<0 else 0.5
    k=trunc(x*INV_LN2+half); kf=float(k)
    hi=x-kf*LN2_HI; lo=kf*LN2_LO; r=hi-lo; t=r*r
    c=r-t*(P1+t*(P2+t*(P3+t*(P4+t*P5))))
    y=1.0-((lo-(r*c)/(2.0-c))-hi)
    return ldexp(y,k)
def ln(x):
    assert x>0
    if x==1.0: return 0.0
    k=ilogb(x); m=ldexp(x,-k)
    if m>=SQRT2: m=m*0.5; k+=1
    f=m-1.0; dk=float(k); s=f/(2.0+f); z=s*s; w=z*z
    t1=w*(LG2+w*(LG4+w*LG6)); t2=z*(LG1+w*(LG3+w*(LG5+w*LG7))); rr=t2+t1
    hfsq=0.5*f*f
    return dk*LN2_HI-((hfsq-(s*(hfsq+rr)+dk*LN2_LO))-f)
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
def ksin(x):
    z=x*x; v=z*x; r=S2+z*(S3+z*(S4+z*(S5+z*S6))); return x+v*(S1+z*r)
def kcos(x): return kcos2(x,0.0)
def reduce(x):
    assert abs(x)<1647099.0
    half=-0.5 if x<0 else 0.5
    n=trunc(x*INV_PIO2+half); f=float(n)
    return n,((x-f*PIO2_1)-f*PIO2_2)-f*PIO2_2T

# Payne and Hanek reduction, digit for digit as lib/ml/ad.rce ph_reduce.
TWO_OVER_PI=[10680707, 7228996, 1387004, 2578385, 16069853, 12639074, 9804092, 4427841, 16666979, 11263675, 12935607, 2387514, 4345298, 14681673, 3074569, 13734428, 16653803, 1880361, 10960616, 8533493, 3062596, 8710556, 7349940, 6258241, 3772886, 3769171, 3798172, 8675211, 12450088, 3874808, 9961438, 366607, 15675153, 9132554, 7151469, 3571407, 2607881, 12013382, 4155038, 6285869, 7677882, 13102053, 15825725, 473591, 9065106, 15363067, 6271263, 9264392, 5636912, 4652155, 7056368, 13614112, 10155062, 1944035, 9527646, 15080200, 6658437, 6231200, 6832269, 16767104, 5075751, 3212806, 1398474, 7579849, 6349435, 12618859]
PIO2_H=1.5707963267948966; PIO2_L=6.123233995736766e-17; SPLIT=134217729.0
def two_prod(a,b):
    p=a*b; t=SPLIT*a; ah=t-(t-a); al=a-ah; t=SPLIT*b; bh=t-(t-b); bl=b-bh
    return p,((ah*bh-p)+ah*bl+al*bh)+al*bl
def ksin2(x,y):
    z=x*x; v=z*x; r=S2+z*(S3+z*(S4+z*(S5+z*S6)))
    return x-((z*(0.5*y-v*r)-y)-v*S1)
def kcos2(x,y):
    # fdlibm 5.3 k_cos.c
    z=x*x; r=z*(C1+z*(C2+z*(C3+z*(C4+z*(C5+z*C6))))); a=abs(x)
    if a<0.2999999523162842: return 1.0-(0.5*z-(z*r-x*y))
    qx=0.28125
    if a<0.7812504768371582:
        q=0.25*a; e=ilogb(q); qx=ldexp(float(math.floor(ldexp(q,20-e))),e-20)
    hz=0.5*z-qx; w=1.0-qx
    return w-(hz-(z*r-x*y))
def ph_reduce(x):
    a=abs(x); e=ilogb(a); m=int(ldexp(a,52-e))
    ms=[m>>48,(m>>24)&0xffffff,m&0xffffff]; s=e-52
    j0=(s-2)//24 if s>=2 else 0
    D=[0]*13
    for j in range(10):
        for k in range(3): D[j+k+1]+=TWO_OVER_PI[j0+j]*ms[k]
    for i in range(12,0,-1):
        D[i-1]+=D[i]>>24; D[i]&=0xffffff
    F=24*(j0+10)-s; r=(24-F%24)%24
    if r:
        D=[0]+D
        for i in range(len(D)-1): D[i]=((D[i]<<r)&0xffffff)|(D[i+1]>>(24-r))
        D[-1]=(D[-1]<<r)&0xffffff
    nf=(F+r)//24; st=len(D)-nf; fr=D[st:]; n=D[st-1]&3; neg=False
    if fr[0]>=0x800000:
        n=(n+1)&3; neg=True; b=0
        for i in range(len(fr)-1,-1,-1):
            v=0-fr[i]-b; b=1 if v<0 else 0; fr[i]=v+(b<<24)
    z=0
    while fr[z]==0: z+=1
    fr=fr+[0]*5
    A=ldexp(float(fr[z]*16777216+fr[z+1]),-24*(z+2)); B=ldexp(float(fr[z+2]*16777216+fr[z+3]),-24*(z+4)); Cc=ldexp(float(fr[z+4]),-24*(z+5))
    fh=A+B; fl=(B-(fh-A))+Cc
    p,pe=two_prod(fh,PIO2_H); pe=pe+(fh*PIO2_L+fl*PIO2_H)
    y0=p+pe; y1=pe-(y0-p)
    if neg: y0=0.0-y0; y1=0.0-y1
    if x<0: n=(4-n)&3; y0=0.0-y0; y1=0.0-y1
    return n,y0,y1
def sin(x):
    if abs(x)<0.7853981633974483: return ksin(x)
    if True:
        n,y0,y1=ph_reduce(x); return [ksin2(y0,y1),kcos2(y0,y1),0.0-ksin2(y0,y1),0.0-kcos2(y0,y1)][n]
    n,y=reduce(x); q=n%4
    return [ksin(y),kcos(y),0.0-ksin(y),0.0-kcos(y)][q]
def cos(x):
    if abs(x)<0.7853981633974483: return kcos(x)
    if True:
        n,y0,y1=ph_reduce(x); return [kcos2(y0,y1),0.0-ksin2(y0,y1),0.0-kcos2(y0,y1),ksin2(y0,y1)][n]
    n,y=reduce(x); q=n%4
    return [kcos(y),0.0-ksin(y),0.0-kcos(y),ksin(y)][q]
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
