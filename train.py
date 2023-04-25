import numpy as np
import torch

import os

from torch import nn, optim
from Task_generation.Polynomial3D import Polynomial3D
from Task_generation.Polynomial import Polynomial

import learn2learn as l2l

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
from siren_pytorch import SirenNet

from adapters import fast_adapt_poisson, fast_adapt_heat, fast_adapt_integral
from ntk_weights import poisson_weights,heat_weights,integral_weights
import argparse

parser = argparse.ArgumentParser(description='Meta-PINN')
parser.add_argument('--PDE', type=str, help='Integrator, Heat or Poisson')
args = parser.parse_args()

NTK_weights = False
weights = (1,1)
def train_MAML_PINN(maml, adapter, weighter, inner_opt, train_sources,test_sources, boundary, opt, adaptation_steps, num_iterations, meta_batch_size,**residual):
    loss_history_train = []
    loss_history_val = []
    inner_opt_state = inner_opt.state_dict()
    global weights 
    for iteration in range(num_iterations):
        maml.zero_grad()

        meta_train_loss = 0.0
        meta_valid_loss = 0.0
        for task in range(meta_batch_size):
            try:
                adapt_opt = type(inner_opt)(maml.parameters(), lr=inner_opt.defaults['lr'], betas = inner_opt.defaults['betas'])
            except:
                adapt_opt = type(inner_opt)(maml.parameters(), lr=inner_opt.defaults['lr'])
            #adapt_opt.load_state_dict(inner_opt_state)
            # Compute meta-training loss
            task = train_sources.sample_task()
            evaluation_loss, inner_opt_state, _ = adapter(task,
                                    boundary,
                                    maml,
                                    adapt_opt,
                                    adaptation_steps,
                                    weights,
                                    dev,
                                    **residual)
            

            

            evaluation_loss.backward()
            meta_train_loss += evaluation_loss.item()
            #print(evaluation_loss)


            # Compute meta-validation loss
            #adapt_opt.load_state_dict(inner_opt_state)
            try:
                adapt_opt = type(inner_opt)(maml.parameters(), lr=inner_opt.defaults['lr'], betas = inner_opt.defaults['betas'])
            except:
                adapt_opt = type(inner_opt)(maml.parameters(), lr=inner_opt.defaults['lr'])
            task = test_sources.sample_task()
            evaluation_loss,_,_ = adapter(task,
                                    boundary,
                                    maml,
                                    adapt_opt,
                                    adaptation_steps,
                                    weights,
                                    dev,
                                    **residual)
            meta_valid_loss += evaluation_loss.item()


        if iteration%1==0:

            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Loss', meta_train_loss/ meta_batch_size)
            print('Meta Valid Loss', meta_valid_loss / meta_batch_size)

            loss_history_train.append(meta_train_loss/ meta_batch_size)
            loss_history_val.append(meta_valid_loss/ meta_batch_size)

        # Average the accumulated gradients and optimize
        #meta_train_loss.backward()
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

        if NTK_weights == True and (iteration+1)%100==0:
            weights = weighter(train_sources.sample_task(),boundary,maml,dev,**residual)
        weights = (min(weights[0],50),min(weights[1],50))
        #print(weights)

    return loss_history_train,loss_history_val,adapt_opt.state_dict()

pde = args.PDE
#Poisson
if pde=='poisson':

    shots=20
    meta_batch_size=8
    num_tasks=10000
    adapt_lr=0.001
    meta_lr=0.0001
    adaptation_steps=5
    num_iterations=10000

    # Data
    # RHS/source/forcing functions

    train_sources = Polynomial3D(num_samples_per_task=2*shots, degree=5, cube_range =[[0,1],[0,1],[0,0]], num_tasks=num_tasks)
    test_sources = Polynomial3D(num_samples_per_task=2*shots, degree=5, cube_range =[[0,1],[0,1],[0,0]], num_tasks=num_tasks)

    # boundary conditions
    x = torch.linspace(0,1,shots)
    y = torch.linspace(0,1,shots)
    x_0 = torch.zeros(1)
    x_1 = torch.ones(1)
    y_0 = torch.zeros(1)
    y_1 = torch.ones(1)
    g = torch.zeros(shots*4,1).to(dev)

    input_bc_1 = torch.cartesian_prod(x_0,y).to(dev)
    input_bc_2 = torch.cartesian_prod(x_1,y).to(dev)
    input_bc_3 = torch.cartesian_prod(x,y_0).to(dev)
    input_bc_4 = torch.cartesian_prod(x,y_1).to(dev)

    boundary = [torch.cat([input_bc_1,input_bc_2,input_bc_3,input_bc_4]),g]


    # create the model

    model = SirenNet(
        dim_in = 2,                        # input dimension, ex. 2d coor
        dim_hidden = 64,                  # hidden dimension
        dim_out = 1,                       # output dimension, ex. rgb value
        num_layers = 3,                    # number of layers
        final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
        w0_initial = 3.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
    ).to(dev)


    opt = optim.Adam(model.parameters(), meta_lr)
    #inner_optimiser = torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9, 0.999))
    inner_optimiser = torch.optim.SGD(model.parameters(),lr=adapt_lr)
    loss_history_train,loss_history_val,opt_state= train_MAML_PINN(model, fast_adapt_poisson, poisson_weights, inner_optimiser, train_sources,test_sources, boundary, opt, adaptation_steps, num_iterations, meta_batch_size)


elif pde=='heat':
    shots=20
    meta_batch_size=8
    num_tasks=16000
    adapt_lr=0.001
    meta_lr=0.0001
    adaptation_steps=5
    num_iterations=10000

    # Data
    # RHS/source/forcing functions

    train_sources = Polynomial3D(num_samples_per_task=2*shots, degree=5, cube_range =[[0,1],[0,0],[0,0]], num_tasks=num_tasks)
    test_sources = Polynomial3D(num_samples_per_task=2*shots, degree=5, cube_range =[[0,1],[0,0],[0,0]], num_tasks=num_tasks)

    t = torch.linspace(0,1,shots)
    x = torch.linspace(0,1,shots)
    residue = torch.cartesian_prod(x,t).to(dev)
    # u(0,t) = 0 = g1(x,t)
    x_0 = torch.zeros(1)
    g_1 = torch.zeros(1).to(dev)
    input_bc_1 = torch.cartesian_prod(x_0,t).to(dev)
    # u(1,t) = 0 = g2(x,t)
    x_1 = torch.ones(1)
    g_2 = torch.zeros(1).to(dev)
    input_bc_2 = torch.cartesian_prod(x_1,t).to(dev)

    boundary = [torch.cat([input_bc_1,input_bc_2]),torch.zeros((torch.cat([input_bc_1,input_bc_2]).shape[0],1)).to(dev)]


    # create the model
    model = SirenNet(
        dim_in = 2,                        # input dimension, ex. 2d coor
        dim_hidden = 64,                  # hidden dimension
        dim_out = 1,                       # output dimension, ex. rgb value
        num_layers = 3,                    # number of layers
        final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
        w0_initial = 3.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
    ).to(dev)

    opt = optim.Adam(model.parameters(), meta_lr)
    #inner_optimiser = torch.optim.Adam(model.parameters(),lr=adapt_lr,betas=(0, 0.999))
    inner_optimiser = torch.optim.SGD(model.parameters(),lr=adapt_lr)
    loss_history_train,loss_history_val,opt_state = train_MAML_PINN(model, fast_adapt_heat, heat_weights, inner_optimiser, train_sources,test_sources, boundary, opt, adaptation_steps, num_iterations, meta_batch_size, residual=residue)


if pde == 'integral':
    shots=40
    meta_batch_size=8
    num_tasks=16000
    adapt_lr=0.001
    meta_lr=0.001
    adaptation_steps=5
    num_iterations=10000

    # Data
    # RHS/source/forcing functions

    train_sources = Polynomial(num_samples_per_task=2*shots, degree=10, num_tasks=num_tasks)
    test_sources = Polynomial(num_samples_per_task=2*shots, degree=10, num_tasks=num_tasks)

    # Initial conditions f(0) = 0
    x_0 = torch.zeros(1,1).to(dev)
    f_0 = torch.zeros(1,1).to(dev)
    boundary = [x_0,f_0]


    # create the model

    model = SirenNet(
        dim_in = 1,                        # input dimension, ex. 2d coor
        dim_hidden = 64,                  # hidden dimension
        dim_out = 1,                       # output dimension, ex. rgb value
        num_layers = 2,                    # number of layers
        final_activation = nn.Identity(),   # activation of final layer (nn.Identity() for direct output)
        w0_initial = 3.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
    ).to(dev)

    opt = optim.Adam(model.parameters(), meta_lr)
    #inner_optimiser = torch.optim.Adam(model.parameters(),lr=0.001,betas=(0, 0.999))
    inner_optimiser = torch.optim.SGD(model.parameters(),lr=adapt_lr)
    loss_history_train,loss_history_val,opt_state  = train_MAML_PINN(model, fast_adapt_integral, integral_weights, inner_optimiser, train_sources,test_sources, boundary, opt, adaptation_steps, num_iterations, meta_batch_size)

# Save trained model
state = {
            'net': model.state_dict(),
            'weights': weights,
            'opt': opt_state,
            'steps': adaptation_steps,
            'adapt_lr': adapt_lr,
            'loss_history_train': loss_history_train,
            'loss_history_val': loss_history_val
        }

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'models')
if not os.path.isdir(final_directory):
            os.mkdir(final_directory)
torch.save(state, './models/'+pde+'.pth')