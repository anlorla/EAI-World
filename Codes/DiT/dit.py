from torch import nn 
import torch 
from time_emb import TimeEmbedding
from dit_block import DiTBlock
from config import T

class DiT(nn.Module):
    def __init__(self,img_size,patch_size,channel,emb_size,label_num,dit_num,head):
        super().__init__()
        self.patch_size=patch_size
        self.patch_cnt=img_size//self.patch_size
        self.channel=channel

        #patchify
        self.conv=nn.Conv2d(in_channels=channel,out_channels=channel*patch_size**2,kernel_size=patch_size,padding=0,stride=patch_size)
        self.patch_emb=nn.Linear(in_features=channel*patch_size**2,out_features=emb_size)
        self.patch_pos_emb=nn.Parameter(torch.rand(1,self.patch_cnt**2,emb_size))
        #time emb
        self.time_emb=nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size,emb_size)
        )
        #label emb
        self.label_emb=nn.Embedding(num_embeddings=label_num,embedding_dim=emb_size)
        
        #dit blocks
        self.dits=nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size,head))

        self.ln=nn.LayerNorm(emb_size)

        # linear back to patch
        self.linear=nn.Linear(emb_size,channel*patch_size**2)

    def forward(self,x,t,y):
        # x: B,C,H,W
        # t: B,1
        # y: B,1
        y_emb=self.label_emb(y)
        t_emb=self.time_emb(t)
        condition_emb=y_emb+t_emb

        x=self.conv(x) # x: B,C*patch_size**2,patch_cnt,patch_cnt
        x=x.permute(0,2,3,1)
        #tokens' len = patch_cnt^2
        #x: B,patch_cnt^2,C*patch_size**2, channel at the end
        x=x.view(x.size(0),self.patch_cnt*self.patch_cnt,x.size(3))

        x=self.patch_emb(x)
        x+=self.patch_pos_emb
        #x: B,patch_cnt^2,emb_size

        for dit in self.dits:
            x=dit(x,condition_emb)
        x=self.ln(x)
        x=self.linear(x) #x:B,patch_cnt^2,C*patch_size**2

        x=x.view(x.size(0),self.patch_cnt,self.patch_cnt,self.channel,self.patch_size,self.patch_size)
        x=x.permute(0,3,1,2,4,5)
        # x: B,C,cnt,cnt,sz,sz
        x=x.permute(0,1,2,4,3,5)
        # x: B,C,cnt,sz,cnt,sz
        x=x.reshape(x.size(0),self.channel,self.patch_cnt*self.patch_size,self.patch_cnt*self.patch_size)
        # x: B,C,H,W
        return x

if __name__=='__main__':
    dit=DiT(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4)
    x=torch.rand(5,1,28,28)
    t=torch.randint(0,T,(5,)) #0~T-1
    y=torch.randint(0,10,(5,)) #0~9
    outputs=dit(x,t,y)
    print(outputs.shape)


