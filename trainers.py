#---------------------------------------------------------------------------
# model trainers
# TODO: re-structure with classes maybe?
#---------------------------------------------------------------------------

import torch 
import torchio as tio
from tqdm import tqdm
from time import time 
import math 
import sys 

#---------------------------------------------------------------------------
# model trainers for segmentation

def seg_train(
    dataloader, 
    scaler,
    model, 
    loss_fn,
    metrics, 
    optimizer, 
    callbacks, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):

    model.train()


    for batch, (X, y) in enumerate(dataloader):
        
        callbacks.on_batch_begin(batch)
        X, y = X.cuda(), y.cuda()

        # Compute prediction error
        with torch.cuda.amp.autocast():
            pred = model(X)
            del X
            loss = loss_fn(pred, y)
            with torch.no_grad():
                if use_deep_supervision:
                    for m in metrics: m(pred[-1].detach(),y.detach())
                else: 
                    for m in metrics: m(pred.detach(),y.detach())

        # Backpropagation
        optimizer.zero_grad() # set gradient to zero, why is that needed?
        scaler.scale(loss).backward() # compute gradient

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)

        scaler.step(optimizer) # apply gradient
        scaler.update()

        loss.detach()
        del pred, y 

        callbacks.on_batch_end(batch)
        
    torch.cuda.empty_cache()

def seg_validate(
    dataloader,
    model,
    loss_fn,
    metrics):
    for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval() # set the module in evaluation mode (only useful for dropout or batchnorm like layers)
    with torch.no_grad(): # set all the requires_grad flags to zeros
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                pred=model(X).detach()
                del X
                loss_fn(pred, y)
                loss_fn.update()
                for m in metrics:
                    m(pred, y)
                    m.update()
                del pred, y
    torch.cuda.empty_cache()
    template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
    for m in metrics: template += ", " + str(m)
    print(template)

#---------------------------------------------------------------------------
