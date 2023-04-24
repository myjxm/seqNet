import torch
import torch.nn as nn
import numpy as np

class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(seqNet, self).__init__()
        self.inDims = inDims
        #print("seqNet inDims:" + str(self.inDims)) #4096
        self.outDims = outDims
        #print("seqNet outDims:" + str(self.outDims)) #4096
        self.w = w
        #print("seqNet w:" + str(self.w)) #5
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):

        #print("seqNet inputx: "+ str(np.shape(x)))  #cache torch.Size([24, 10, 4096])  train torch.Size([192, 10, 4096])  seqL=10
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        #print("seqNet segFt:" + str(np.shape(seqFt))) #cache torch.Size([24, 4096, 6]) train torch.Size([192, 4096, 6])
        seqFt = torch.mean(seqFt,-1)
        #print("seqNet segFtaftermean:" + str(np.shape(seqFt))) #torch.Size([24, 4096]) train torch.Size([192, 4096])

        return seqFt
    
class Delta(nn.Module):
    def __init__(self, inDims, seqL):

        super(Delta, self).__init__()
        self.inDims = inDims
        self.weight = (np.ones(seqL,np.float32))/(seqL/2.0)
        self.weight[:seqL//2] *= -1
        self.weight = nn.Parameter(torch.from_numpy(self.weight),requires_grad=False)

    def forward(self, x):

        # make desc dim as C
        x = x.permute(0,2,1) # makes [B,T,C] as [B,C,T]
        delta = torch.matmul(x,self.weight)

        return delta
