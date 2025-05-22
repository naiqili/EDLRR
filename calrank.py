import math
from numpy.random import randn
import torch
import numpy as np
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def parameters_gamma_nuclear_norm(gamma=1.0, lam=1.0):
    parameters=[]
    parameters.append(0.0)
    for i in range(1,10):
        parameters.append(math.pow(-1,i+1)*(1 + gamma)*lam/math.pow(gamma,i))
    return parameters

def parameters_LNN(lam=1.0):
    parameters = []
    parameters.append(0.0)
    for i in range(1,11):
        parameters.append(lam*math.pow(-1,i+1)*(1/i))
    return parameters

def parameters_Geman(gamma=1.0, lam=1.0):
    parameters=[]
    parameters.append(0.0)
    for i in range(1,10):
        parameters.append(lam*math.pow(-1,i+1)/math.pow(gamma,i))
    return parameters

def parameters_Laplace(gamma=1.0, lam=1.0):
    parameters=[]
    parameters.append(0.0)
    for i in range(1,11):
        parameters.append(lam*math.pow(-1,i+1)/(math.pow(gamma,i)*math.factorial(i)))
    return parameters

def parameters_ETP(gamma=1.0, lam=1.0):
    parameters=[]
    parameters.append(0.0)
    for i in range(1,11):
        parameters.append(lam*math.pow(-1,i+1)/(math.pow(gamma,i)*math.factorial(i)*(1-math.exp(-1/gamma))))
    return parameters

def cal_gamma_nuclear_norm(s,gamma=1.0, lam=1.0):
    l = 0.0
    for i in range(s.shape[0]):
        x = s[i]
        l += lam*(1+gamma)*x/(gamma+x)
    return l

def cal_LNN(s, lam=1.0):
    l = 0.0
    for i in range(s.shape[0]):
        x = s[i]
        l += math.log(s[i]+1)
    return l

def cal_Geman(s, gamma=1.0, lam=1.0):
    l = 0.0
    for i in range(s.shape[0]):
        x = s[i]
        l += lam*x/(x+gamma)
    return l

def cal_Laplace(s, gamma=1.0, lam=1.0):
    l = 0.0
    for i in range(s.shape[0]):
        x = s[i]
        l += lam*(1-math.exp(-x/gamma))
    return l

def cal_ETP(s, gamma=1.0, lam=1.0):
    l = 0.0
    for i in range(s.shape[0]):
        x = s[i]
        l += lam*(1-math.exp(-x/gamma))/(1-math.exp(-1/gamma))
    return l

def matrix_sqrt(S,k=1e6,n=5000,gpu=try_gpu()):
    A = S @ S.T
    _u,s,_v=np.linalg.svd(A.cpu().detach().numpy())
    k=1.1*s[0]
    d,_ = A.shape
    I = torch.eye(d,requires_grad=False,device=gpu)
    R = (I - A/k).float()
    Ri=torch.eye(d,requires_grad=False,device=gpu)
    Q=0
    t1=1
    t2=1.0
    c=1
    for i in range(1,n+1):
        Ri=Ri@R
        t1=(0.5+1-i)
        t2=i
        c*=(t1/t2)
        Q += torch.tensor(abs(c), requires_grad=False, device=gpu)*Ri

    res=float(np.sqrt(k))*(I-Q)
    return res

def cal_rank_nuclear(S, a=1e-1, num_inv=5, num_g=100, gpu=try_gpu(), n=100):
    #https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    ST = S.T
    _, d = ST.shape
    Ai = torch.inverse(S@ST + a*torch.eye(d,requires_grad=False,device=gpu)) @ S
    for t in range(num_inv):
        Aii = 2*Ai - Ai@ST@Ai
    # print('Aii-Ai:', np.mean(((Aii-Ai)).cpu().detach().numpy()**2))
    Ai = Aii
    Sinv = Ai
    Sinv = Sinv.T
    # print('S @ Sinv - I:', np.mean((S @ Sinv - torch.eye(d,requires_grad=False,device=gpu)).cpu().detach().numpy()**2))
    Q = matrix_sqrt(S,n=n,gpu=gpu)
    d,_ = S.shape
    g = np.float32(randn(d, num_g))
    g = torch.tensor(g, requires_grad=False, device=gpu)
    v1 = (S @ Sinv) @ g
    v2 = Q @ g
    l = v1.reshape(-1)@v2.reshape(-1)
    return l/num_g

def cal_rank_f(S, parameters=[], a=1e-1, num_inv=5, num_g=100, gpu=try_gpu(), n=100):
    if parameters==[]:
        print("empty parameter list")
        return
    ST = S.T
    _, d = ST.shape
    Ai = torch.inverse(S@ST + a*torch.eye(d,requires_grad=False,device=gpu)) @ S
    for t in range(num_inv):
        Aii = 2*Ai - Ai@ST@Ai
    Ai = Aii
    Sinv = Ai
    Sinv = Sinv.T
    Q = matrix_sqrt(S,n=n,gpu=gpu)
    d,_ = S.shape
    g = np.float32(randn(d, num_g))
    g = torch.tensor(g, requires_grad=False, device=gpu)
    v1 = (S @ Sinv) @ g
    total_l = d * parameters[0]
    cur_l = Q
    for i in range(1,len(parameters)):
        parameter = parameters[i]
        v2 = cur_l @ g
        l = v1.reshape(-1)@v2.reshape(-1)
        total_l = total_l + parameter * l/num_g
        cur_l = cur_l @ Q
    return total_l


def cal_rank_nuclear_woinverse(S, a=1e-1, num_inv=5, num_g=100, gpu=try_gpu(), n=100):
    #https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    _, d = S.shape
    Ai = torch.inverse(S.T@S + a*torch.eye(d,requires_grad=False,device=gpu)) @ S.T
    for t in range(num_inv):
        Aii = 2*Ai - Ai@S@Ai
    Ai = Aii
    Sinv = Ai
    Q = matrix_sqrt(S,n=n,gpu=gpu)
    d,_ = S.shape
    g = np.float32(randn(d, num_g))
    g = torch.tensor(g, requires_grad=False, device=gpu)
    v1 = (S @ Sinv) @ g
    v2 = Q @ g
    l = v1.reshape(-1)@v2.reshape(-1)
    return l/num_g

def cal_rank_f_woinverse(S, parameters=[], a=1e-1, num_inv=5, num_g=100, gpu=try_gpu(), n=100):
    if parameters==[]:
        print("empty parameter list")
        return
    _, d = S.shape
    Ai = torch.inverse(S.T@S + a*torch.eye(d,requires_grad=False,device=gpu)) @ S.T
    for t in range(num_inv):
        Aii = 2*Ai - Ai@S@Ai
    Ai = Aii
    Sinv = Ai
    Q = matrix_sqrt(S,n=n,gpu=gpu)
    SS = S @ S.T
    d,_ = S.shape
    g = np.float32(randn(d, num_g))
    g = torch.tensor(g, requires_grad=False, device=gpu)
    v1 = (S @ Sinv) @ g
    total_l = SS.shape[0] * parameters[0]
    cur_l = Q
    for i in range(1,len(parameters)):
        parameter = parameters[i]
        v2 = cur_l @ g
        l = v1.reshape(-1)@v2.reshape(-1)
        total_l = total_l + parameter * l/num_g
        cur_l = cur_l @ Q
    return total_l