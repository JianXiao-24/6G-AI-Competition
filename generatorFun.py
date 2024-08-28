import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
flag_file = 'flag0'
import numpy as np
import torch
import h5py

def layer_norm(x):
    return torch.layer_norm(x,[x.shape[-1]])
def swish(x):
    return x * torch.sigmoid(x)

class Attention(torch.nn.Module):
    def __init__(self,dims,hd,heads=1):
        super(Attention,self).__init__()
        self.fc=torch.nn.Linear(dims,heads*hd*3)
        self.fcfc=torch.nn.Linear(hd*heads,dims)
        self.dims=dims
        self.hd=hd
        self.heads=heads
    def forward(self,x):
        batch_size=x.shape[0]
        length=x.shape[1]
        x=torch.nn.functional.dropout(x,p=0.3,training=self.training)
        qkv=swish(self.fc(x))
        qkv=qkv.reshape([batch_size,length,self.heads,self.hd*3])
        q,k,v=torch.split(qkv,self.hd,-1)
        att=torch.softmax(torch.einsum('bwhd,bmhd->bhwm',q,k),-1)
        new_v=torch.einsum('bhwm,bmhd->bwhd',att,v).reshape([batch_size,length,-1])
        
        m=swish(self.fcfc(new_v))
        return layer_norm(m+x)

class Quan(torch.nn.Module):
    def __init__(self,d,dim_list=[35,32,28,25]):
        super(Quan,self).__init__()
        self.dim_list=dim_list
        self.linears = torch.nn.ModuleList([torch.nn.Linear(d, i) for i in self.dim_list])
    def forward(self, x):
        oup_list=[linear(x[:,i]) for i,linear in enumerate(self.linears)]
        oup=torch.cat(oup_list,1)
        oup=layer_norm(oup.T).T
        oup=torch.sigmoid(oup)
        return oup

class Dequan(torch.nn.Module):
    def __init__(self,d,dim_list=[35,32,28,25]):
        super(Dequan,self).__init__()
        self.dim_list=dim_list
        self.linears = torch.nn.ModuleList([torch.nn.Linear(i,d) for i in self.dim_list])
    def forward(self, x):
        oup=x
        if self.training:
            oup_B=torch.randint(0,2,oup.shape,dtype=torch.float32).cuda()
            i=(torch.rand(oup.shape,dtype=torch.float32).cuda()<0.05).type(torch.float32)
            oup=oup*(1-i)+oup_B*i
        oup=layer_norm(oup)
        s=0
        oup_list=list()
        for dim,linear in zip(self.dim_list,self.linears):
            oup_list.append(linear(oup[:,s:s+dim]))
            s+=dim
        oup=torch.stack(oup_list,1)
        oup=layer_norm(oup)
        return oup
    
class Encoder(torch.nn.Module):
    def __init__(self,hd=256):
        super(Encoder,self).__init__()
        self.PE=torch.nn.Parameter(torch.zeros([32,hd],dtype=torch.float32).cuda())
        torch.nn.init.kaiming_uniform(self.PE)
        self.attention=torch.nn.Sequential(*[Attention(hd,32,2) for i in range(3)])
        self.fc1=torch.nn.Linear(128*3,hd)
        self.quan=Quan(hd)
        self.hd=hd
        self.E=torch.diag(torch.ones(32)).cuda()

    def forward(self, x):
        batch_size=x.shape[0]
        std=torch.sqrt(x[:,0]**2+x[:,1]**2).std(-1)
        _,I=torch.sort(-std,-1)
        I=I[:,:4]
        IE=torch.nn.functional.embedding(I.type(torch.int64),self.E)
        
        x_norm=(x[:,0]**2+x[:,1]**2)**0.5
        x=torch.stack([x[:,0],x[:,1],x_norm],dim=1)
        
        oup=x.transpose(1,2)
        oup=oup.reshape([batch_size,32,-1])
        oup=layer_norm(self.fc1(oup)+self.PE)
        
        oup=self.attention(oup)
        
        oup=torch.einsum('bmd,bwm->bwd',oup,IE)
        
        
        oup=self.quan(oup)
        
        oup=torch.cat([oup,I.type(torch.float32)],-1)
        return oup

class Decoder(torch.nn.Module):
    def __init__(self,hd=256):
        super(Decoder,self).__init__()
        self.PE=torch.nn.Parameter(torch.zeros([32,hd],dtype=torch.float32).cuda())
        torch.nn.init.kaiming_uniform(self.PE)
        self.fc1=torch.nn.Linear(hd,256)
        self.fc2=torch.nn.Linear(32,32)
        self.dequan=Dequan(hd)
        self.attention=torch.nn.Sequential(*[Attention(hd,32,2) for i in range(3)])
        self.hd=hd
        self.E=torch.diag(torch.ones(32)).cuda()

    def forward(self, x):
        batch_size=x.shape[0]
        oup,I=x[:,:-4],x[:,-4:]
        I=I.type(torch.int64)
        IE=torch.nn.functional.embedding(I.type(torch.int64),self.E)
        oup=self.dequan(oup)
        oup=layer_norm(torch.einsum('bmd,bmw->bwd',oup,IE)+self.PE)
        
        oup=self.attention(oup)
        oup=self.fc2(oup.transpose(1,2)).transpose(1,2)
        oup=self.fc1(oup)
        
        oup=oup.reshape([batch_size,32,2,128]).transpose(1,2)

        return oup
    
class Generator1(torch.nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()
    
        self.linear_block = torch.nn.Sequential(
            torch.nn.Linear(LATENT_DIM, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(negative_slope=0.3),
        )
        self.convblock1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 1024, 2, stride=2),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(negative_slope=0.3))
        
        self.convblock2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512, 2, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(negative_slope=0.3))
        
        self.convblock3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 2, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(negative_slope=0.3))
        
        self.convblock4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 2, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(negative_slope=0.3))
        
        self.convblock5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 1, 2, stride=2),
            torch.nn.Tanh())
        
    def forward(self, input):
        output = self.linear_block(input)
        output = output.view(-1, 128, 4, 2)
        output = self.convblock1(output)
        output = self.convblock2(output)
        output = self.convblock3(output)
        output = self.convblock4(output)
        output = self.convblock5(output)
        return output
    
class Generator2(torch.nn.Module):
    def __init__(self,hd=128):
        super(Generator2,self).__init__()
        self.data=torch.nn.Parameter(torch.zeros([4000,2,4,32,32],dtype=torch.float32))
        self.encoder=Encoder(hd)
        self.decoder=Decoder(hd)
        self.lamda=0.5
        self.detach_encoder=False
    def forward(self,ids):
        batch_size=len(ids)
        index=torch.tensor(ids,dtype=torch.int64).cuda().type(torch.int64)
        x=torch.nn.functional.embedding(index,self.data)
        x=(x.transpose(0,-1)/x.std((1,2,3,4))).transpose(0,-1)
        oup=x
        oup=oup.transpose(2,-1).reshape([batch_size,2,32,128])
        oup=self.encoder(oup)
        if self.training:
            if self.detach_encoder:
                oup=oup.round().detach()
        else:
            feature,I=oup[:,:-4].round(),oup[:,-4:]
            feature=(torch.nn.functional.dropout(torch.ones_like(feature),p=0.5)>0).type(torch.float32)
            oup=torch.cat([feature,I],-1)
        oup=self.decoder(oup)
        oup=oup.reshape([batch_size,2,32,32,4]).transpose(2,-1)
        if self.training:
            pass
        else:
            return oup

def generator_1(num_fake_1, file_generator_1,file_real_2):
    use_cuda = torch.cuda.is_available()
    generator_C = torch.load(file_generator_1)
    generator_C = generator_C.cuda().eval()
    # real_C = np.load(file_real_1)
    num_tx = 32
    num_rx = 4
    num_delay = 32
    latent_dim = 128
    size_packet = 500
    with torch.no_grad():
        for idx in range(int(num_fake_1 / size_packet)):
            latent_vectors = torch.randn(size_packet, latent_dim)
            if use_cuda:
                latent_vectors = latent_vectors.cuda()
            fake_data = generator_C(latent_vectors)
            fake_data = fake_data.cpu().numpy()
            fake_data = np.reshape(fake_data, [size_packet, num_rx, num_tx, num_delay, 2])
            fake_data_r = fake_data[:, :, :, :, 0]
            fake_data_i = fake_data[:, :, :, :, 1]
            fake_data_reshape = fake_data_r + fake_data_i * 1j
            if idx == 0:
                data_fake_all = fake_data_reshape
            else:
                data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
    return data_fake_all


def generator_2(num_fake_2, file_generator_2,file_real_2=None):
    model_path=os.path.dirname(os.path.abspath(file_generator_2))+'/generator_2.pth.tar'
    model=Generator2(256).cuda()
    model.load_state_dict(torch.load(model_path)['state_dict'],True)
    _=model.eval()
#     lis=[model(torch.arange(i*100,i*100+100)%4000).detach().cpu().numpy() for i in range(400)]
    lis=[model(torch.arange(i*100,i*100+100)%4000).detach().cpu().numpy() for i in range(360)]+[model.data.cpu().detach().numpy()]
    data=np.concatenate(lis,0)
    data=data[:,0]+data[:,1]*1j
    return data[:num_fake_2]