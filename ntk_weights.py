import torch
import numpy as np

def list_to_vector(list_):
    return torch.cat([x.reshape(-1) for x in list_])

def jacobian(f, model, device):
    jac = []
    for i in range(len(f)):
        f[i].backward(retain_graph=True)
        deriv = [w.grad.reshape(-1).to(device) if w.grad is not None else torch.tensor([0]).to(device) for w in model.parameters()]
        jac.append(list_to_vector(deriv))
    jac = torch.vstack(jac)
    return jac

def compute_ntk(J1, J2):
    return J1 @ J2.T

def integral_weights(task, boundary,model,device):
    train_inputs, train_targets = torch.from_numpy(task[:][0]).float().to(device), torch.from_numpy(task[:][1]).float().to(device)
    x_support, f_support = train_inputs[::2], train_targets[::2]
    x_query, f_query = train_inputs[1::2], train_targets[1::2]
    x_boundary, g_boundary = boundary[0], boundary[1]
    x_support,x_query = (x_support-0.5)/0.29,(x_query-0.5)/0.29

    x = x_support.clone()
    x.requires_grad = True
    ''' 
    Initial condition fitting
    Calculating the loss that expresses how well our solution respects the initial conditions
    '''
    u_boundary = model(x_boundary)

    ''' 
    ODE fitting
    Calculating the loss that expresses how well our solution respects the ODE
    u' + u = f
    '''
    u = model(x) #u
    u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0] #u'

    '''NTK weights'''
    J_u = jacobian(u_boundary,model,device)
    J_r = jacobian(u_x,model,device)
    K_u = compute_ntk(J_u,J_u)
    K_r = compute_ntk(J_r, J_r)

    trace_K = torch.trace(K_u) + torch.trace(K_r)

    lb = trace_K / torch.trace(K_u)
    lr = trace_K / torch.trace(K_r)
    return lb.item(),lr.item()


def heat_weights(task, boundary,model,device,residual):
    k=0.1
    train_inputs, train_targets = torch.from_numpy(task[:][0]).float().to(device), torch.from_numpy(task[:][1]).float().to(device)
    x_support, f_support = train_inputs[::2,:2], train_targets[::2]
    x_query, f_query = train_inputs[1::2,:2], train_targets[1::2]
    x_boundary, g_boundary = boundary[0], boundary[1]
    f_support = f_support*torch.sin(x_support[:,0]*np.pi)
    f_support = f_support.reshape(-1,1)
    f_query = f_query*torch.sin(x_query[:,0]*np.pi)
    f_query = f_query.reshape(-1,1)

    xt = residual.clone()
    xt.requires_grad = True
    ''' 
    Boundary condition fitting
    Calculating the loss that expresses how well our solution respects the boundary conditions
    '''
    u_boundary = model(x_boundary)

    # Boundary condition as defined by the specific task
    u_boundary_task = model(x_support)
    
    ''' 
    PDE fitting
    Calculating the loss that expresses how well our solution respects the ODE
    ku_xx = u_t 
    '''
    u = model(xt) #u
    du = torch.autograd.grad(u.sum(),xt,create_graph=True)[0]
    u_x = du[:,0].reshape(-1,1) #u_x
    du_x = torch.autograd.grad(u_x.sum(),xt,create_graph=True)[0] 
    u_xx = du_x[:,0].reshape(-1,1) #u_xx
    u_t  = du[:,1].reshape(-1,1) #u_t

    '''NTK weights'''
    J_u = jacobian(torch.cat((u_boundary, u_boundary_task), dim=0),model,device)
    J_r = jacobian(k*u_xx-u_t,model,device)
    K_u = compute_ntk(J_u,J_u)
    K_r = compute_ntk(J_r, J_r)

    trace_K = torch.trace(K_u) + torch.trace(K_r)

    lb = trace_K / torch.trace(K_u)
    lr = trace_K / torch.trace(K_r)
    return lb.item(),lr.item()

def poisson_weights(task, boundary,model,device):

    train_inputs, train_targets = torch.from_numpy(task[:][0]).float().to(device), torch.from_numpy(task[:][1]).float().to(device)
    x_support, f_support = train_inputs[::2,:2], train_targets[::2].reshape(-1,1)
    x_query, f_query = train_inputs[1::2,:2], train_targets[1::2].reshape(-1,1)
    x_boundary, g_boundary = boundary[0], boundary[1]


    x = x_support.clone()
    x.requires_grad = True
    ''' 
    Boundary condition fitting
    Calculating the loss that expresses how well our solution respects the boundary conditions
    '''
    u_boundary = model(x_boundary)

    ''' 
    PDE fitting
    Calculating the loss that expresses how well our solution respects the ODE
    u_xx + u_yy = f(x,y)  
    '''
    u = model(x) #u
    du = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    u_x = du[:,0].reshape(-1,1) #u_x
    du_x = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0] 
    u_xx = du_x[:,0].reshape(-1,1) #u_xx
    u_y  = du[:,1].reshape(-1,1) #u_t
    du_y = torch.autograd.grad(u_y.sum(),x,create_graph=True)[0] 
    u_yy = du_y[:,1].reshape(-1,1) #u_xx

    '''NTK weights'''
    J_u = jacobian(u_boundary,model,device)
    J_r = jacobian(u_xx+u_yy,model,device)
    K_u = compute_ntk(J_u,J_u)
    K_r = compute_ntk(J_r, J_r)

    trace_K = torch.trace(K_u) + torch.trace(K_r)

    lb = trace_K / torch.trace(K_u)
    lr = trace_K / torch.trace(K_r)
    return lb.item(),lr.item()
