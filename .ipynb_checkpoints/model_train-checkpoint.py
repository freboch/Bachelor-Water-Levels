import torch

from models import AA, DAA, NMF, my_COSINEloss, my_LSloss



def train_model(model,X,k,lr, n_epoch,W=None, tol=1e-5):
    if 'DAA' in model:
        model_inst = DAA(X,k)  
    elif 'AA' in model:
        model_inst = AA(X,k)
    elif 'NMF' in model:
        model_inst = NMF(X,k)
    
    optimizer = torch.optim.Adam(model_inst.parameters(),lr=lr)
    
    loss_all = torch.zeros(n_epoch).float()
    loss_all[loss_all == 0] = float('nan')
    
    # model training
    epoch = 0
    end_epoch = None
    while epoch <n_epoch and end_epoch is None:
        # DAA
        X_ = model_inst() 
        
        if 'DAA' in model:
            loss = my_COSINEloss(X_, X, W) # loss function
        elif 'AA' in model or 'NMF' in model:
            loss = my_LSloss(X_, X, W)
            
        model_inst.zero_grad() # need to clear the old gradients
        loss.backward()
        optimizer.step()
        # save all loss values
        loss_all[epoch] = loss.detach()
        if epoch > 0 and torch.abs(loss_all[epoch]-loss_all[epoch-1])<tol:
            # the change in loss is below threshold
            end_epoch = epoch
        epoch +=1
        
    if end_epoch is None:
        end_epoch = epoch-1
    if 'AA' in model: # DAA and AA
        final_archetypes = model_inst.get_archetypes()
        final_S = model_inst.get_S()
        return loss_all, final_archetypes, final_S, end_epoch
    elif 'NMF' in model:
        final_H = model_inst.get_H()
        final_W = model_inst.get_W()
        return loss_all, final_H, final_W, end_epoch