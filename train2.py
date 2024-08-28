import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import torch
import time
import h5py


# In[2]:


seed=7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# In[3]:


def load_data(path):
    data = h5py.File(path, 'r')
    key=os.path.basename(path).split('.')[0]
    data = np.transpose(data[key])
    data = np.stack([data['real'],data['imag']],1)
    return data


# In[4]:


def K_nearest(h_true_smp, h_fake_smp, rx_num=4, tx_num=32, delay_num=32, flag=2):
    t1 = time.time()
    h_true = np.reshape(h_true_smp, [h_true_smp.shape[0], rx_num * tx_num * delay_num])
    h_fake = np.reshape(h_fake_smp, [h_fake_smp.shape[0], rx_num * tx_num * delay_num])
    h_true_norm = np.linalg.norm(h_true, axis=1)
    h_fake_norm = np.linalg.norm(h_fake, axis=1)
    h_true_norm = h_true_norm[:, np.newaxis]
    h_fake_norm = h_fake_norm[:, np.newaxis]
    h_true_norm_matrix = np.tile(h_true_norm, (1, rx_num * tx_num * delay_num))
    h_fake_norm_matrix = np.tile(h_fake_norm, (1, rx_num * tx_num * delay_num))
    h_true = h_true / h_true_norm_matrix
    h_fake = h_fake / h_fake_norm_matrix

    t2 = time.time()

    r_s = abs(np.dot(h_fake, h_true.conj().T))
    r = r_s * r_s

    t3 = time.time()

    r_max = np.max(r, axis = 1)
    r_idx = np.argmax(r, axis = 1)
    K_sim_abs_mean = np.mean(r_max)

    counts_idx, counts_num = np.unique(r_idx,return_counts=True)
    K_multi = np.zeros((1, h_fake_smp.shape[0]))
    K_multi[:, counts_idx] = counts_num
    K_multi_std = float(np.sqrt(np.var(K_multi, axis=1) * h_fake_smp.shape[0] / (h_fake_smp.shape[0] - 1)))

    t4 = time.time()
    return K_sim_abs_mean, K_multi_std, K_multi_std / K_sim_abs_mean


# In[5]:


def save(path,g1=None,g2=None):
    if not os.path.isdir(path):
        os.mkdir(path)
    if g1!=None:
        modelSave1 = path+'/generator_1.pth.tar'
        torch.save({'state_dict': g1.state_dict(), }, modelSave1)
    
    if g2!=None:
        modelSave2 = path+'/generator_2.pth.tar'
        torch.save({'state_dict': g2.state_dict(), }, modelSave2)
    return 'OK'

def load(path,g1=None,g2=None):
    if g1!=None:
        model_path = path+'/generator_1.pth.tar'
        g1.load_state_dict(torch.load(model_path)['state_dict'],False)
    
    if g2!=None:
        model_path = path+'/generator_2.pth.tar'
        g2.load_state_dict(torch.load(model_path)['state_dict'],False)
    return 'OK'


# In[6]:


def get_loss(y_,y):
    batch_size=y_.shape[0]
    y_=y_.reshape([batch_size,2,-1])
    y=y.reshape([batch_size,2,-1])
    r=(y_[:,0]*y[:,0]+y_[:,1]*y[:,1]).sum(-1)
    i=(y_[:,1]*y[:,0]-y_[:,0]*y[:,1]).sum(-1)
    a=r**2+i**2
    b=torch.square(y_).sum((1,2))*torch.square(y).sum((1,2))
    loss=a/b
    loss=loss.mean()
    return -loss


# In[7]:


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
class Quory(torch.nn.Module):
    def __init__(self,dims,hd,lenght,heads=1):
        super(Quory,self).__init__()
        self.q=torch.nn.Parameter(torch.zeros([lenght,heads,hd],dtype=torch.float32))
        torch.nn.init.kaiming_uniform(self.q)
        self.fc=torch.nn.Linear(dims,heads*hd*2)
        self.fcfc=torch.nn.Linear(hd*heads,dims)
        self.dims=dims
        self.hd=hd
        self.heads=heads
    def forward(self,x):
        batch_size=x.shape[0]
        length=self.q.shape[0]
        kv=self.fc(x)
        kv=kv.reshape([batch_size,-1,self.heads,self.hd*2])
        k,v=torch.split(kv,self.hd,-1)
        att=torch.softmax(torch.einsum('whd,bmhd->bhwm',self.q,k),-1)
        new_v=torch.einsum('bhwm,bmhd->bwhd',att,v).reshape([batch_size,length,-1])
        
        return layer_norm(self.fcfc(new_v))
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


# In[8]:


class Generator(torch.nn.Module):
    def __init__(self,hd=128):
        super(Generator,self).__init__()
        self.data=torch.Tensor(load_data(root_path+'/raw_data/H2_32T4R.mat')).type(torch.float32).cuda()
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
            loss=get_loss(x,oup)+(torch.square(oup-x).mean((1,2,3,4)).sqrt()+1e-7).mean()
            d=(torch.square(oup.unsqueeze(1)-oup).mean((2,3,4,5))+1e-7).sqrt()
            d=(d+torch.diag(torch.ones(d.shape[-1]).cuda()*1000)).min(-1)[0]
            dx=(torch.square(x.unsqueeze(1)-x).mean((2,3,4,5))+1e-7).sqrt()
            dx=(dx+torch.diag(torch.ones(dx.shape[-1]).cuda()*1000)).min(-1)[0]
            d_loss=torch.minimum(d,dx*self.lamda).mean()
            return loss,d_loss
        else:
            return oup


# In[9]:


data=load_data(root_path+'/raw_data/H2_32T4R.mat')
model=Generator(hd=256)
model=model.cuda()
test_data=data[3000:,0]+data[3000:,1]*1j


# In[10]:


optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)


# In[11]:


batch_size=128
step=0
model.lamda=0.8


# In[14]:


while True:    
    optimizer.param_groups[0]['lr']=min(1e-4,1e-4/3000*(step+1))
    if not model.training:model.train()
    loss,d_loss=model(torch.randint(0,4000,[batch_size]))
    optimizer.zero_grad()
    (loss-d_loss*0.1).backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
    optimizer.step()
    if step%1000==0:
        test_loss,_=model(torch.arange(step*batch_size,step*batch_size+batch_size)%1000+3000)
        model.eval()
        n=torch.cat([model(torch.arange(0,100))for i in range(10)],0).cpu().detach().numpy()
        n=n[:,0]+n[:,1]*1j
        ks=K_nearest(test_data,n)
        print(step,loss.detach().cpu().numpy(),test_loss.cpu().detach().numpy(),*ks)
    step+=1
    if step==986107:break


# In[15]:


model.data=torch.nn.Parameter(model.data)
save(root_path+'/user_data',g2=model)
