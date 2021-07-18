import numpy as np
import torch
import time
import copy


class SpsMirror(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=1.0,
                 mu=None,
                 gamma=2.0,
                 eta_max=None,
                 step_size_method='smooth_iter',
                 fstar=0,
                 eps=1e-8,
                 pnorm=2,
                 project_method=None,
                 ):

        params = list(params)
        super().__init__(params, {})
        self.eps = eps
        self.params = params
       
        self.eta_max = eta_max
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.step_size_method = step_size_method
        self.state['step'] = 0
        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch
        self.project_method = project_method
        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.fstar = fstar
        self.pnorm = pnorm
        

        # step 1
        if pnorm is None:
            qnorm = 2 * np.log(params[0].shape[1]) 
            self.pnorm = pnorm = qnorm / (qnorm - 1.)

        assert self.pnorm <= 2
        assert self.pnorm > 1
        # step 2
        self.qnorm = pnorm / (pnorm - 1.)
        if isinstance(self.step_size_method, float):
           assert c is None 

        self.c = c
        self.mu =  (self.pnorm - 1)

        if self.project_method == 'clip':
            for p in self.params:
                p.data.clamp_(0)
        elif self.project_method == 'L1':
            for p in self.params:
                p.data = torch.rand(p.shape).cuda()


    def step(self, closure):
        # --------------------------------------------
        # step 3: Compute Loss
        
        loss = closure()

        # --------------------------------------------
        # Compute Step Size
        

        grad_current = get_grad_list(self.params)
        if self.project_method == 'L1':
            grad_norm = compute_norm(grad_current, p='inf')
        else:
            grad_norm = compute_norm(grad_current, p=self.qnorm)

        # Acquire step size
        if grad_norm < 1e-8:
            step_size = 0.

        else:
            # Acquire step size
            if self.step_size_method in ['constant', 'smooth_iter']:
                step_size = float((self.mu / self.c) * (loss - self.fstar) / ((grad_norm)**2 + self.eps))

            # Adapt the step size
            if isinstance(self.step_size_method, float):
                step_size = self.step_size_method

            elif self.step_size_method in ['constant']:
                if loss < self.fstar:
                    step_size = 0.
                else:
                    if self.eta_max is not None:
                        step_size = min(self.eta_max, step_size)

            elif self.step_size_method in ['smooth_iter']:
                # smoothly adjust the step size
                coeff = self.gamma**(1./self.n_batches_per_epoch)
                step_size = min(coeff * self.state['step_size'], step_size)
            else:
                raise ValueError('step_size_method: %s not supported' % self.step_size_method)
            
        # --------------------------------------------
        # Update Parameters
        if step_size != 0:

            if self.project_method == 'L1':
                exp_update(self.params, step_size, grad_current, 
                            pnorm=self.pnorm, qnorm=self.qnorm)
            else:
                qnorm_update(self.params, step_size, grad_current, 
                    pnorm=self.pnorm, qnorm=self.qnorm)

        # --------------------------------------------
        # Project Parameters
        if self.project_method is None:
            pass
        elif self.project_method == 'clip':
            for p in self.params:
                p.data.clamp_(0)
        elif self.project_method == 'L1':
            pass
        else:
            raise ValueError('project_method: %s not supported' % self.project_method)

        # --------------------------------------------
        # Update state
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        self.state['step_size'] = step_size
        self.state['grad_norm'] = float(grad_norm)
        self.state['step'] += 1

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('Got NaNs')

        return float(loss)

# utils
# ------------------------------
def compute_norm(any_list, p=2):
    # step 4: TODO verify that this is good
    norm = 0.
    if isinstance(p, str) and p == 'inf':
        for g in any_list:
            if g is None or (isinstance(g, float) and g == 0.):
                continue
            norm = max(norm, torch.abs(g.max()))
    else:
        for g in any_list:
            if g is None or (isinstance(g, float) and g == 0.):
                continue
            
            norm += torch.sum(torch.abs(g)**p) 
        norm = torch.pow(norm, 1./p)
    # assert abs(float(torch.norm(any_list[0], p=p)) - float(norm)) < 1e-4
    return norm


def get_grad_list(params):
    grad_list = []
    for p in params:
        g = p.grad
        if g is None:
            g = 0.
        else:
            g = p.grad.data
                   
        grad_list += [g]        
                   
    return grad_list

# def qnorm_update(params, step_size, grad_current, pnorm=2):
#     for p, g in zip(params, grad_current):
#         if p.grad is None:
#             continue
#         d_p = p.grad.data
#         update = torch.sign(p.data) * pnorm * (torch.abs(p.data)**(pnorm-1))   - step_size * d_p
#         p.data = torch.sign(update) * (torch.abs(update/(pnorm))**(1./(pnorm - 1))) 

def exp_update(params, step_size, grad_current, pnorm=2, qnorm=2):
    update_sum = 0.
    for p, g in zip(params, grad_current):
        if p.grad is None:
            continue
        update = p.data * torch.exp(-step_size*g)
        update_sum += update.sum()
        p.data = update
        
    for p, g in zip(params, grad_current):
        if p.grad is None:
            continue
        p.data = p.data / update_sum
    if torch.isnan(p.data).sum() > 0:
        raise ValueError('NaNs detected')
    assert(p.data.min() >= 0)
    

def qnorm_update(params, step_size, grad_current, pnorm=2, qnorm=2):
    # stage 1
    param_norm = compute_norm([p.data for p in params], p=pnorm)
    # torch.norm(p.data, p=pnorm)
    update_list = []
    for p, g in zip(params, grad_current):
        if p.grad is None:
            continue
        d_p = p.grad.data

        # 
        update = torch.sign(p.data) *  (torch.abs(p.data)**(pnorm-1))  
        update = update * param_norm **(2-pnorm) - step_size * d_p
        update_list += [update]

    update_norm = compute_norm(update_list, p=qnorm)

    # stage 2
    for p, u in zip(params, update_list):
        p.data = torch.sign(u) * (torch.abs(u)**(qnorm-1)) * update_norm**(2-qnorm) 


def qnorm_update_constrained(params, step_size, grad_current, q=0.5):
    raise ValueError('Not implemented')