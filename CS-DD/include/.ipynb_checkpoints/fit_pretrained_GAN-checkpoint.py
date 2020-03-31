{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fit(net,\n",
    "        num_channels,\n",
    "        img_clean_var,\n",
    "        num_iter = 5000,\n",
    "        LR = 0.01,\n",
    "        OPTIMIZER='adam',\n",
    "        optimizer2='SGD',\n",
    "        opt_input = False,\n",
    "        reg_noise_std = 0,\n",
    "        reg_noise_decayevery = 100000,\n",
    "        mask_var = None,\n",
    "        lr_decay_epoch = 0,\n",
    "        net_input = None,\n",
    "        net_input_gen = \"random\",\n",
    "        find_best=False,\n",
    "        weight_decay=0,\n",
    "        Ameas = 1,\n",
    "        model = 1,\n",
    "        LR_LS = 0.02,\n",
    "        code = 'uniform',\n",
    "        num_iters_inner = 100,\n",
    "        decodetype='upsample', #'upsample','transposeconv'\n",
    "        optim = 'gd', #gd or pgd\n",
    "        print_inner = False,\n",
    "        numit_inner = 20,\n",
    "        decay_every = 500,\n",
    "        out_channels=1,\n",
    "       ):\n",
    "\n",
    "    if net_input is not None:\n",
    "        print(\"input provided\")\n",
    "    else:\n",
    "        # feed uniform noise into the network \n",
    "        totalupsample = 2**(len(num_channels)-1)\n",
    "        \n",
    "        #if running as decoder/compressor\n",
    "        if len(img_clean_var.shape)==4:\n",
    "            width = int(img_clean_var.data.shape[2]/(totalupsample))\n",
    "            height = int(img_clean_var.data.shape[3]/(totalupsample))\n",
    "        #if running compressive imaging    \n",
    "        elif len(img_clean_var.shape)==2:\n",
    "            w = np.sqrt(int(Ameas.shape[1]/out_channels))\n",
    "            width = int(w/(totalupsample))\n",
    "            height = int(w/(totalupsample))\n",
    "            \n",
    "        shape = [1,num_channels[0], width, height]  \n",
    "        print(\"shape of latent code B1: \", shape)\n",
    "\n",
    "        print(\"initializing latent code B1...\")\n",
    "        net_input = Variable(torch.zeros(shape))\n",
    "        if code== 'uniform':\n",
    "            net_input.data.uniform_()\n",
    "        elif code== 'gaussian':\n",
    "            net_input.data.normal_()\n",
    "        elif code== 'hadamard':\n",
    "            B = Variable(torch.tensor(hadamard(width*height,dtype=float)))\n",
    "            idx = np.random.choice(width*height,num_channels[0])\n",
    "            net_input.data = B[list(idx),:].view(-1,num_channels[0],width,height)\n",
    "        elif code== 'identity':\n",
    "            B = Variable(torch.tensor(np.identity(width*height,dtype=float)))\n",
    "            idx = np.random.choice(width*height,num_channels[0])\n",
    "            net_input.data = B[list(idx),:].view(-1,num_channels[0],width,height)\n",
    "        elif code=='xavier':\n",
    "            torch.nn.init.xavier_uniform(net_input.data)\n",
    "        \n",
    "        net_input.data *= 1./10\n",
    "        \n",
    "    net_input_saved = net_input.data.clone()\n",
    "    noise = net_input.data.clone()\n",
    "    \n",
    "    # Define variables of optimization\n",
    "    '''if decodetype=='upsample':\n",
    "        p = [x for x in net.decoder.parameters() ] #list of all weigths\n",
    "    elif decodetype=='transposeconv':\n",
    "        p = [x for x in net.convdecoder.parameters() ] #list of all weigths\n",
    "        \n",
    "    if(opt_input == True): # optimizer over the input as well\n",
    "        net_input.requires_grad = True\n",
    "        print('optimizing over latent code Z1')\n",
    "        p += [net_input]\n",
    "    else:\n",
    "        print('not optimizing over latent code Z1')\n",
    "    '''\n",
    "    \n",
    "    p = [net_input]\n",
    "    \n",
    "    \n",
    "\n",
    "    mse_wrt_truth = np.zeros(num_iter)\n",
    "    mse_outer = np.zeros(num_iter)\n",
    "    \n",
    "    if OPTIMIZER == 'SGD':\n",
    "        print(\"optimize decoder with SGD\", LR)\n",
    "        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9,weight_decay=weight_decay)\n",
    "    elif OPTIMIZER == 'adam':\n",
    "        print(\"optimize decoder with adam\", LR)\n",
    "        optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)\n",
    "\n",
    "    mse = torch.nn.MSELoss() \n",
    "    \n",
    "    if find_best:\n",
    "        best_net = copy.deepcopy(net)\n",
    "        best_mse = 1000000.0\n",
    "\n",
    "\n",
    "    if optim=='gd':    \n",
    "        print('optimizing with gradient descent...')\n",
    "        x_in = net(net_input.type(dtype)).data.clone()\n",
    "        for i in range(num_iter):\n",
    "\n",
    "            #################\n",
    "            if lr_decay_epoch is not 0:\n",
    "                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch,factor=0.7)\n",
    "                \n",
    "            #################\n",
    "            def closure():\n",
    "                optimizer.zero_grad()           \n",
    "                outp = net(net_input.type(dtype))\n",
    "                loss = mse(apply_f(outp,Ameas,model), img_clean_var)\n",
    "                loss.backward()\n",
    "                mse_wrt_truth[i] = loss.data.cpu().numpy()\n",
    "                return loss\n",
    "            \n",
    "            loss = optimizer.step(closure) \n",
    "                  \n",
    "            print ('Iteration %05d   Train loss %f ' % (i, loss.detach().cpu().numpy()), '\\r', end='')\n",
    "\n",
    "            if find_best:\n",
    "                # if training loss improves by at least one percent, we found a new best net\n",
    "                if best_mse > 1.01*loss.detach().cpu().numpy():\n",
    "                    best_mse = loss.detach().cpu().numpy()\n",
    "                    best_net = copy.deepcopy(net)\n",
    "\n",
    "        if find_best:\n",
    "            net = best_net\n",
    "    \n",
    "    \n",
    "    return mse_wrt_truth,net_input_saved, net, net_input, x_in"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}