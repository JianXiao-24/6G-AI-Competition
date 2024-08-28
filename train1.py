import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import torch
import time
import h5py
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
seed=7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# In[2]:


def K_nearest(h_true_smp, h_fake_smp, rx_num, tx_num, delay_num, flag):

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
    r_s = abs(np.dot(h_fake, h_true.conj().T))
    r = r_s * r_s
    r_max = np.max(r, axis = 1)
    r_idx = np.argmax(r, axis = 1)
    K_sim_abs_mean = np.mean(r_max)
    counts_idx, counts_num = np.unique(r_idx,return_counts=True)
    K_multi = np.zeros((1, h_fake_smp.shape[0]))
    K_multi[:, counts_idx] = counts_num
    K_multi_std = float(np.sqrt(np.var(K_multi, axis=1) * h_fake_smp.shape[0] / (h_fake_smp.shape[0] - 1)))
    return K_sim_abs_mean, K_multi_std, K_multi_std / K_sim_abs_mean

def norm_data(x, num_sample, num_rx, num_tx, num_delay):
    x2 = np.reshape(x, [num_sample, num_rx * num_tx * num_delay * 2])
    x_max = np.max(abs(x2), axis=1)
    x_max = x_max[:,np.newaxis]
    x3 = x2 / x_max
    y = np.reshape(x3, [num_sample, 1, num_rx * num_tx , num_delay * 2])
    return y


# In[3]:


NUM_RX = 4
NUM_TX = 32
NUM_DELAY = 32
NUM_SAMPLE_TRAIN = 500
LATENT_DIM = 128
BATCH_SIZE = 4
EPOCH = 500

num_tx = 32
num_rx = 4
num_delay = 32
latent_dim = 128
size_packet = 500
num_fake_1 = 500
NUM_REAL_1 = 500
data_train = h5py.File(root_path+'/raw_data/H1_32T4R.mat', 'r')
data_train = np.transpose(data_train['H1_32T4R'][:])
data_train = data_train[:, :, :, :, np.newaxis]
data_train = np.concatenate([data_train['real'], data_train['imag']], 4)
data_train = np.reshape(data_train, [NUM_SAMPLE_TRAIN, NUM_RX* NUM_TX, NUM_DELAY* 2, 1])
train_channel = norm_data(data_train, NUM_SAMPLE_TRAIN, NUM_RX, NUM_TX, NUM_DELAY)

real_1_test = h5py.File(root_path+'/raw_data/H1_32T4R.mat', 'r')
real_1_test = np.transpose(real_1_test['H1_32T4R'][:])
real_1_test = real_1_test[::int(real_1_test.shape[0] / NUM_REAL_1), :, :, :]
real_1_test = real_1_test['real'] + real_1_test['imag'] * 1j
LATENT_DIM = 128


# In[4]:


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convblock1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, 3, 2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.3),
        )
        self.convblock2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.3),
        )
        self.convblock3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.3),
        )
        self.convblock4 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 3, 2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.3), 
        )
        self.convblock5 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, 3, 2, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.3), 
        )
        self.linear_block = torch.nn.Sequential(
            torch.nn.Linear(256, 1),
        )

    def forward(self, input):
        output = self.convblock1(input)
        output = self.convblock2(output)
        output = self.convblock3(output)
        output = self.convblock4(output)
        output = self.convblock5(output)
        output = output.view(-1, 256) 
        output = self.linear_block(output)
        return output

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


g_model = Generator1()
d_model = Discriminator()

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
if use_cuda:
    d_model = d_model.cuda(gpu)
    g_model = g_model.cuda(gpu)

one = torch.tensor(1, dtype=torch.float)
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

discriminator_optimizer = torch.optim.Adam(d_model.parameters(), lr=2e-4, betas=(0.5, 0.9))
generator_optimizer = torch.optim.Adam(g_model.parameters(), lr=2e-4, betas=(0.5, 0.9))


def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    epochs = EPOCH
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def cal_gradient_penalty(d_model, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 1, 128, 64)
    alpha = alpha.cuda(gpu) if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = d_model(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty
best_score = 0
epoch = 0
for iteration in range(int(EPOCH*(NUM_SAMPLE_TRAIN/BATCH_SIZE))):
    for p in d_model.parameters():
        p.requires_grad = True
    for i in range(3):
        d_model.zero_grad()
        real_channel = train_channel[np.random.choice(train_channel.shape[0], BATCH_SIZE, replace=False)]
        real_channel = torch.from_numpy(real_channel)  
        if use_cuda:
            real_channel = real_channel.cuda(gpu)
        real_channel_v = torch.autograd.Variable(real_channel)
        real_channel_v = real_channel_v.type(torch.cuda.FloatTensor)
        real_logits = d_model(real_channel_v)
        real_logits = real_logits.mean() 
        real_logits.backward(mone)
        random_latent_vectors = torch.randn(BATCH_SIZE, LATENT_DIM)
        if use_cuda:
            random_latent_vectors = random_latent_vectors.cuda(gpu)
        with torch.no_grad():
            noisev = torch.autograd.Variable(random_latent_vectors)
        fake_channel = torch.autograd.Variable(g_model(noisev).data)
        fake_logits = d_model(fake_channel)
        fake_logits = fake_logits.mean() 
        fake_logits.backward(one)
        gradient_penalty = cal_gradient_penalty(d_model, real_channel_v.data, fake_channel.data)
        gradient_penalty.backward()
        d_loss = fake_logits - real_logits + gradient_penalty
        discriminator_optimizer.step()

    for p in d_model.parameters():
        p.requires_grad = False
    g_model.zero_grad()
    random_latent_vectors = torch.randn(BATCH_SIZE, LATENT_DIM)
    if use_cuda:
        random_latent_vectors = random_latent_vectors.cuda(gpu)
    noisev = torch.autograd.Variable(random_latent_vectors)
    fake_channel = g_model(noisev)
    fake_logits = d_model(fake_channel)
    fake_logits = fake_logits.mean()
    fake_logits.backward(mone)
    g_loss = -fake_logits
    generator_optimizer.step()
    if iteration % (int(NUM_SAMPLE_TRAIN/BATCH_SIZE)) == 0:
        epoch = epoch+1
        adjust_learning_rate(generator_optimizer, epoch,2e-4,8e-6)
        adjust_learning_rate(discriminator_optimizer, epoch,2e-4,8e-6)
        with torch.no_grad():
            for idx in range(int(num_fake_1 / size_packet)):
                latent_vectors = torch.randn(size_packet, latent_dim)#(500,128)
                if use_cuda:
                    latent_vectors = latent_vectors.cuda()
                    fake_data = g_model(latent_vectors)
                    fake_data = fake_data.cpu().numpy()
                    fake_data = np.reshape(fake_data, [size_packet, num_rx, num_tx, num_delay, 2])
                    fake_data_r = fake_data[:, :, :, :, 0]
                    fake_data_i = fake_data[:, :, :, :, 1]
                    fake_data_reshape = fake_data_r + fake_data_i * 1j
                    if idx == 0:
                        data_fake_all = fake_data_reshape
                    else:
                        data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
            sim_1, multi_1, multi_div_sim_1 = K_nearest(real_1_test, data_fake_all, NUM_RX, NUM_TX, NUM_DELAY, 1)
        if sim_1 > 0.2 and multi_1 < 20:
            score_2 = (20 - multi_div_sim_1) /20
            print('Score2 = ' + str(float(score_2)))
        print('sim_1 : %.6f, multi_1 : %.6f,  multi_div_sim_1: %.6f' % (sim_1, multi_1, multi_div_sim_1))
        print('Epoch = ' + str(int(iteration / int(NUM_SAMPLE_TRAIN/BATCH_SIZE))) + ', d_loss = ' + str(d_loss.cpu().data.numpy()) + ', g_loss = ' + str(g_loss.cpu().data.numpy()))
        torch.save(g_model, root_path+'/user_data/generator_1.pth.tar')
