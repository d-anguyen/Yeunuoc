import torch
import numpy as np

def CSGM(G, latentDim, y, device, num_iter=1600):
    G.eval()
    G.to(device)
    
    loss_per_iter = np.zeros(num_iter)
    
    mse = torch.nn.MSELoss()
    z_init = torch.normal(torch.zeros(1,latentDim)).to(device)

    z = torch.autograd.Variable(z_init, requires_grad = True)
    optimizer = torch.optim.Adam([{'params': z, 'lr': 0.1}])

    print('Running CSGM:')
    for i in range(num_iter):
        optimizer.zero_grad()
        
        Gz = G(z)
        
        loss = mse(Gz, y)
        loss.backward()
        optimizer.step()
        
        if(i % 100 == 0):
            print('CSGM step %d/%d, objective loss = %.5f' %(i, num_iter, loss.item()))
        
        loss_per_iter[i] = loss.item()
    return z, loss_per_iter



def CSGM2(G, latentDim, y, A, device, num_iter=1600):
    G.eval()
    G.to(device)
    
    mse = torch.nn.MSELoss()
    z_init = torch.normal(torch.zeros(1,latentDim)).to(device)
    m_image = y.numel()
    
    loss_per_iter = np.zeros(num_iter)

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
            loss_per_iter[i] = loss.item()/np.sqrt(m_image) # normalization A <-> A/sqrt(m)
            print('CSGM step %d/%d, objective = %.5f' %(i, num_iter, loss_per_iter[i]))
        
        
    return z, loss_per_iter

