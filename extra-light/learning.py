"""NumPy adaptation of Andrej Karpathy's microgpt (one layer, RMSNorm, ReLU).

Source: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
This is a game-policy adaptation, not the unchanged scalar-autograd script.
Only the last token has a loss. With ONE layer, earlier queries/MLPs cannot
affect that token, so computing only its query is exactly equivalent and O(T).
All trainable weights, including embeddings and Q/K/V, receive gradients.
"""
import numpy as np


def norm(x):
    inv = 1 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-5)
    return x * inv, inv


def norm_back(grad, normalized, inv):
    return (grad - normalized*np.mean(grad*normalized, axis=-1, keepdims=True))*inv


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class MicroGPT:
    def __init__(self, context, vocab=130, actions=2083, width=16, heads=4, seed=42, dtype=np.float32):
        assert width % heads == 0
        self.config = dict(context=context, vocab=vocab, actions=actions, width=width, heads=heads, seed=seed)
        self.d, self.h = width, heads
        rng = np.random.default_rng(seed)
        shapes = dict(wte=(vocab,width), wpe=(context,width), wq=(width,width), wk=(width,width),
                      wv=(width,width), wo=(width,width), fc1=(4*width,width), fc2=(width,4*width),
                      lm_head=(actions,width))
        self.p = {k: rng.normal(0,.08,s).astype(dtype) for k,s in shapes.items()}

    def forward(self, tokens, legal, cache=False):
        p=self.p; b,t=tokens.shape; h=self.h; d=self.d; hd=d//h
        assert t == self.config['context']
        e=p['wte'][tokens]+p['wpe'][None,:,:]
        x,xi=norm(e); z,zi=norm(x)
        q=(z[:,-1]@p['wq'].T).reshape(b,h,hd)
        k=(z@p['wk'].T).reshape(b,t,h,hd).transpose(0,2,1,3)
        v=(z@p['wv'].T).reshape(b,t,h,hd).transpose(0,2,1,3)
        att=softmax(np.einsum('bhd,bhtd->bht',q,k)/hd**.5)
        a=np.einsum('bht,bhtd->bhd',att,v).reshape(b,d)
        r=x[:,-1]+a@p['wo'].T
        n,ni=norm(r); pre=n@p['fc1'].T; hidden=np.maximum(pre,0)
        y=r+hidden@p['fc2'].T
        selected=p['lm_head'][np.maximum(legal,0)]
        logits=np.einsum('bd,bld->bl',y,selected)
        logits=np.where(legal>=0,logits,-1e9)
        if cache:
            return logits,(tokens,legal,x,xi,z,zi,q,k,v,att,a,r,n,ni,pre,hidden,y,selected)
        return logits

    def loss(self, tokens, legal, targets, backward=False):
        logits,c=self.forward(tokens,legal,cache=True)
        probs=softmax(logits); b=len(tokens)
        indices=np.argmax(legal==targets[:,None],axis=1)
        assert np.all(legal[np.arange(b),indices]==targets), 'Illegal target'
        value=float(-np.log(np.maximum(probs[np.arange(b),indices],1e-30)).mean())
        if not backward:return value
        (tokens,legal,x,xi,z,zi,q,k,v,att,a,r,n,ni,pre,hidden,y,selected)=c
        p=self.p; g={key:np.zeros_like(w) for key,w in p.items()}; h=self.h; d=self.d; hd=d//h; t=tokens.shape[1]
        dl=probs.copy(); dl[np.arange(b),indices]-=1; dl/=b
        np.add.at(g['lm_head'],np.maximum(legal,0).ravel(),(dl[:,:,None]*y[:,None,:]).reshape(-1,d))
        dy=np.einsum('bl,bld->bd',dl,selected)
        g['fc2']=dy.T@hidden
        dh=(dy@p['fc2'])*(pre>0)
        g['fc1']=dh.T@n
        dr=dy+norm_back(dh@p['fc1'],n,ni)
        g['wo']=dr.T@a
        da=(dr@p['wo']).reshape(b,h,hd)
        dv=att[:,:,:,None]*da[:,:,None,:]
        datt=np.einsum('bhd,bhtd->bht',da,v)
        ds=att*(datt-(datt*att).sum(axis=-1,keepdims=True))/hd**.5
        dq=np.einsum('bht,bhtd->bhd',ds,k).reshape(b,d)
        dk=ds[:,:,:,None]*q[:,:,None,:]
        dk=dk.transpose(0,2,1,3).reshape(b*t,d)
        dv=dv.transpose(0,2,1,3).reshape(b*t,d)
        zflat=z.reshape(b*t,d)
        g['wq']=dq.T@z[:,-1]; g['wk']=dk.T@zflat; g['wv']=dv.T@zflat
        dz=(dk@p['wk']+dv@p['wv']).reshape(b,t,d)
        dz[:,-1]+=dq@p['wq']
        dx=norm_back(dz,z,zi); dx[:,-1]+=dr
        de=norm_back(dx,x,xi)
        g['wpe']=de.sum(axis=0)
        np.add.at(g['wte'],tokens.ravel(),de.reshape(-1,d))
        return value,g

    def choose(self,tokens,legal):
        return legal[np.arange(len(tokens)),self.forward(tokens,legal).argmax(axis=1)]

    def save(self,path):
        import json
        np.savez_compressed(path,config=json.dumps(self.config),**self.p)

    @classmethod
    def load(cls,path):
        import json
        with np.load(path,allow_pickle=False) as data:
            model=cls(**json.loads(str(data['config'])))
            model.p={k:data[k].copy() for k in model.p}
        return model


class Adam:
    def __init__(self,model):
        self.m={k:np.zeros_like(v) for k,v in model.p.items()}
        self.v={k:np.zeros_like(v) for k,v in model.p.items()}
        self.step=0

    def update(self,model,grads,lr):
        self.step+=1
        length=np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        scale=min(1.,5./max(length,1e-12))
        for key,p in model.p.items():
            g=grads[key]*scale
            self.m[key]=.85*self.m[key]+.15*g
            self.v[key]=.99*self.v[key]+.01*g*g
            p-=lr*(self.m[key]/(1-.85**self.step))/(np.sqrt(self.v[key]/(1-.99**self.step))+1e-8)
