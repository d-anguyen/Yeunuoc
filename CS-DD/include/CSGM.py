#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch


def CSGM(G, latentDim, y, x, device, num_iter=1600):
    G.eval()
    G.to(device)
    
    mse_wrt_truth = np.zeros(num_iter)
    
    objective = torch.nn.MSELoss()
    z_init = torch.normal(torch.zeros(1,latentDim)).to(device)

    z = torch.autograd.Variable(z_init, requires_grad = True)
    optimizer = torch.optim.Adam([{'params': z, 'lr': 0.1}])

    print('Running CSGM:')
    for i in range(num_iter):
        optimizer.zero_grad()
        Gz = G(z)
        #AGz = config.A(Gz)
        
        loss = objective(Gz, y)
        loss.backward()
        optimizer.step()
        if(i % 100 == 0):
            print('CSGM step %d/%d, objective = %.5f' %(i, num_iter, loss.item()))
        
        mse_wrt_truth[i] = mse(Gz,x).item()
    return z, mse_wrt_truth



def CSGM2(G, latentDim, y, x, A, device, num_iter=1600):
    G.eval()
    G.to(device)
    
    mse = torch.nn.MSELoss()
    z_init = torch.normal(torch.zeros(1,latentDim)).to(device)
    
    mse_wrt_truth = np.zeros(num_iter)

    z = torch.autograd.Variable(z_init, requires_grad = True)
    optimizer = torch.optim.Adam([{'params': z, 'lr': 0.1}])
    
    print('Running CSGM:')
    for i in range(num_iter):
        optimizer.zero_grad()
        
        Gz = G(z)
        Gz_vec = Gz.reshape(Gz.numel())
        AGz = torch.matmul(A, Gz_vec) # TODO: try nonlinear measurement processes
        
        loss = mse(AGz, y)
        loss.backward()
        optimizer.step()
        
        if(i % 50 == 0):
            print('CSGM step %d/%d, objective = %.5f' %(i, num_iter, loss.item()))
        
        mse_wrt_truth[i] = mse(Gz_vec,x).item()
        
    return z, mse_wrt_truth

