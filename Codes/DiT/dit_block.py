from torch import nn
import torch
import math

class DiTBlock(nn.Module):
    def __init__(self, emb_size,nhead):
        super().__init__()

        self.emb_size=emb_size
        self.nhead=nhead
        self.head_dim=emb_size//nhead

        #conditioning
        self.gamma1=nn.Linear(emb_size,emb_size)
        self.beta1=nn.Linear(emb_size,emb_size)
        self.alpha1=nn.Linear(emb_size,emb_size)
        self.gamma2=nn.Linear(emb_size,emb_size)
        self.beta2=nn.Linear(emb_size,emb_size)
        self.alpha2=nn.Linear(emb_size,emb_size)

        #layer norm
        self.ln1=nn.LayerNorm(emb_size)
        self.ln2=nn.LayerNorm(emb_size)

        #MHSA
        # B,seq_len,nhead*emb_size
        self.wq=nn.Linear(emb_size,nhead*self.head_dim)
        self.wk=nn.Linear(emb_size,nhead*self.head_dim)
        self.wv=nn.Linear(emb_size,nhead*self.head_dim)
        self.lv=nn.Linear(emb_size,emb_size)

        self.ff=nn.Sequential(
            nn.Linear(emb_size,emb_size*4),
            nn.GELU(),
            nn.Linear(emb_size*4,emb_size)
        )

    def forward(self,x,cond):
        # cond: B,1
        gamma1_val=self.gamma1(cond) #B,emb_sz
        beta1_val=self.beta1(cond)
        alpha1_val=self.alpha1(cond)
        gamma2_val=self.gamma2(cond)
        beta2_val=self.beta2(cond)
        alpha2_val=self.alpha2(cond)

        y=self.ln1(x) # B,seq_len,emb_size
        #scale&shift
        y=y*(1+gamma1_val.unsqueeze(1))+beta1_val.unsqueeze(1)
        #y: B,seq_len,emb_sz

        #attention
        q=self.wq(y) #B,seq_len,n*emb_size
        k=self.wk(y)
        v=self.wv(y)
        q=q.view(q.size(0),q.size(1),self.nhead,self.head_dim).permute(0,2,1,3)
        #q: B,nhead,seq_len,dhead
        k=k.view(k.size(0),k.size(1),self.nhead,self.head_dim).permute(0,2,3,1)
        #k: B,nhead,dhead,seq_len
        v=v.view(v.size(0),v.size(1),self.nhead,self.head_dim).permute(0,2,1,3)
        #v: B,nhead,seq_len,dhead
        attn=q@k/math.sqrt(q.size(3))
        attn=torch.softmax(attn,dim=-1)
        y=attn@v
        y=y.permute(0,2,1,3)
        #y: B,seq_len,nhead,dhead
        y=y.reshape(y.size(0),y.size(1),y.size(2)*y.size(3))
        y=self.lv(y)
        #y:B,seq_len,emb_sz

        #scale
        y=y*alpha1_val.unsqueeze(1)
        #residual
        y=x+y

        #layer norm
        z=self.ln2(y)
        #scale&shift
        z=z*(1+gamma2_val.unsqueeze(1))+beta2_val.unsqueeze(1)
        # feed forward
        z=self.ff(z)
        #scale
        z=z*alpha2_val.unsqueeze(1)
        return y+z
    
if __name__=='__main__':
    dit_block=DiTBlock(emb_size=16,nhead=4)

    x=torch.rand((5,49,16))
    cond=torch.rand((5,16))
    outputs=dit_block(x,cond)
    print(outputs.shape)
