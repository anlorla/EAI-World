from torch import nn
import torch

class ViT(nn.Module):
    def __init__(self,emb_size=16):
        super().__init__()
        self.patch_size=4 
        self.patch_count=28//self.patch_size 
        self.conv=nn.Conv2d(in_channels=1,out_channels=self.patch_size**2,kernel_size=self.patch_size,padding=0,stride=self.patch_size)
        self.patch_emb=nn.Linear(in_features=self.patch_size**2,out_features=emb_size) # B,49,16
        self.cls_token=nn.Parameter(torch.rand(1,1,emb_size)) #1,16
        self.pos_emb=nn.Parameter(torch.rand(1,self.patch_count**2+1,emb_size)) #1,50,16

        self.transformer_enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emb_size,nhead=2,batch_first=True),num_layers=3)
        self.cls_linear=nn.Linear(in_features=emb_size,out_features=10)

    def forward(self,x):
        x=self.conv(x) #B,1,28,28->B,16,7,
        x=x.view(x.size(0),x.size(1),self.patch_count**2) #B,16,49
        x=x.permute(0,2,1) #B,49,16
        x=self.patch_emb(x)
        cls_token=self.cls_token.expand(x.size(0),1,x.size(2)) 
        x=torch.cat((cls_token,x),dim=1) #B,50,16
        x=self.pos_emb+x

        y=self.transformer_enc(x) 
        return self.cls_linear(y[:,0,:]) #[CLS]输出分类
    
if __name__=='main':
    vit=ViT()
    x=torch.rand(5,1,28,28) # B=5
    y=vit(x)
    print(y.shape)

