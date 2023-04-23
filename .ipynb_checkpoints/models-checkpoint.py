import torch
import torch.nn as nn
from project_func import proj_to_sphere

class AA(nn.Module):
    def __init__(self, X, K):
        super(AA, self).__init__()
        self.X = X # data 
        self.N = X.shape[0] # number of samples
        self.X_tilde = X # transformed version of data (currently no transformation)
        self.S = nn.parameter.Parameter(torch.randn(self.N, K, requires_grad=True)) # Archetypal mixing matrix
        self.C = nn.parameter.Parameter(torch.randn(K, self.N, requires_grad=True)) # Archetypal generator matrix

    def forward(self):
        s1 = torch.softmax(self.S, dim=1) # non-negative and rows sum to 1
        c1 = torch.softmax(self.C, dim=1) # non-negative and rows sum to 1
        
        # reconstructed data matrix
        X_hat = torch.matmul(s1,torch.matmul(c1,self.X_tilde))
        return X_hat
    
    def get_archetypes(self):
        c1 = torch.softmax(self.C, dim=1) # non-negative and rows sum to 1
        A = torch.matmul(c1,self.X_tilde)
        return A
    
    def get_S(self):
        s1 = torch.softmax(self.S, dim=1)
        return s1
    
    
    
class DAA(nn.Module):
    def __init__(self, X, K):
        super(DAA, self).__init__()
        self.X = X # data 
        self.N = X.shape[0] # number of samples
        self.X_tilde = proj_to_sphere(X) # transformed version of data 
        self.S = nn.parameter.Parameter(torch.randn(self.N, K, requires_grad=True)) # Archetypal mixing matrix
        self.C = nn.parameter.Parameter(torch.randn(K, self.N, requires_grad=True)) # Archetypal generator matrix

    def forward(self):
        s1 = torch.softmax(self.S, dim=1) # non-negative and rows sum to 1
        c1 = torch.softmax(self.C, dim=1) # non-negative and rows sum to 1
        
        # reconstructed data matrix
        X_hat = torch.matmul(s1,torch.matmul(c1,self.X_tilde))
        # project the reconstructed x to the sphere
        X_hat_proj = proj_to_sphere(X_hat) 
        return X_hat_proj
    
    def get_archetypes(self):
        c1 = torch.softmax(self.C, dim=1) # non-negative and rows sum to 1
        A = torch.matmul(c1,self.X_tilde)
        A_proj = proj_to_sphere(A) 
        return A_proj

    def get_S(self):
        s1 = torch.softmax(self.S, dim=1)
        return s1
    
    
class NMF(nn.Module):
    def __init__(self, X, K):
        super(NMF, self).__init__()
        self.X = X # data 
        self.N = X.shape[0] # number of samples
        self.M = X.shape[1] # dimension
        # K <= min(m,n)
        self.X_tilde = X # transformed version of data (currently no transformation)
        self.W = nn.parameter.Parameter(torch.randn(self.M, K, requires_grad=True)) # Feature Matrix
        self.H = nn.parameter.Parameter(torch.randn(K, self.N, requires_grad=True)) # coefficient matrix (Weights associated with W)
        self.softplus = nn.Softplus()
        
    def forward(self):
        w1 = self.softplus(self.W) # non-negative and rows sum to 1
        h1 = self.softplus(self.H) # non-negative and rows sum to 1
        
        # reconstructed data matrix
        X_hat = torch.matmul(w1,h1)
        return X_hat.T
    
    def get_H(self):
        h1 = self.softplus(self.H)
        return h1
    
    def get_W(self):
        w1 = self.softplus(self.W)
        return w1


def my_COSINEloss(output, target, weights=None):
    if weights is None:
        weights = torch.ones(len(output))
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # cosine similarity
    sim = cos(output, target)
    # cosine distance
    dist = (1-(weights*sim)).mean()
    #loss = dist/output.shape[0]
    loss = dist
    return loss

    
def my_LSloss(output, target, weights=None):
    if weights is None:
        weights = torch.ones(len(output))
    diff = target-output
    w_diff = (weights*diff.T).T
    dist = torch.linalg.norm(w_diff,ord='fro')**2
    loss = dist/output.shape[0]
    return loss
# dist/n