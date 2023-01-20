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
    
    torch.cuda.synchronize()
    t_start_epoch = time()
    print("[time] start epoch")

    for batch, (X, y) in enumerate(dataloader):
        
        callbacks.on_batch_begin(batch)
        X, y = X.cuda(), y.cuda()

        # print("batch:",batch, "memory reserved:", torch.cuda.memory_reserved(0), "allocated memory", torch.cuda.memory_allocated(0), "free memory:", torch.cuda.memory_reserved(0)-torch.cuda.memory_allocated(0), "percentage of free memory:", torch.cuda.memory_allocated(0)/torch.cuda.max_memory_allocated(0))

        torch.cuda.synchronize()
        t_data_loading = time()
        # print("[time] data loading:", t_data_loading-t_start_epoch)
        if t_data_loading-t_start_epoch > 1:
            print("SLOW!", batch, "[time] data loading:", t_data_loading-t_start_epoch)

        # Compute prediction error
        with torch.cuda.amp.autocast(scaler is not None):
            pred = model(X); del X
            loss = loss_fn(pred, y)
        with torch.no_grad():
            if use_deep_supervision:
                for m in metrics: m(pred[-1],y)
            else: 
                for m in metrics: m(pred,y)

            # torch.cuda.synchronize()
            # t_model_pred = time()
            # print("[time] model prediction:", t_model_pred-t_data_loading)

        # Backpropagation
        optimizer.zero_grad() # set gradient to zero, why is that needed?

        if scaler is not None:
            scaler.scale(loss).backward() # compute gradient

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)

            # torch.cuda.synchronize()
            # t_backward = time()
            # print("[time] backward computation:", t_backward-t_model_pred)

            scaler.step(optimizer) # apply gradient
            scaler.update()
        else: 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()

        # torch.cuda.synchronize()
        # t_optim_update = time()
        # print("[time] optimizer update:", t_optim_update-t_backward)

        # loss.detach()
        del loss, pred, y 

        callbacks.on_batch_end(batch)

        torch.cuda.synchronize()
        t_start_epoch = time()
        
    torch.cuda.empty_cache()

def seg_validate(
    dataloader,
    model,
    loss_fn,
    metrics,
    use_fp16):
    for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval() # set the module in evaluation mode (only useful for dropout or batchnorm like layers)
    with torch.no_grad(): # set all the requires_grad flags to zeros
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast(use_fp16):
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
# model trainers for segmentation with patches 

def seg_patch_validate(dataloader, model, loss_fn, metrics):
    print("Start validation...")
    for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval() # set the module in evaluation mode (only useful for dropout or batchnorm like layers)
    with torch.no_grad(): # set all the requires_grad flags to zeros
        for it in tqdm(dataloader):
            # pred_aggr = tio.inference.GridAggregator(it, overlap_mode='average')
            # y_aggr = tio.inference.GridAggregator(it, overlap_mode='average')
            patch_loader = torch.utils.data.DataLoader(it, batch_size=dataloader.batch_size)
            for patch in patch_loader:
                X = patch['img'][tio.DATA]
                y = patch['msk'][tio.DATA]
                X, y = X.cuda(), y.cuda()
                pred=model(X).detach()

                # pred_aggr.add_batch(pred, patch[tio.LOCATION])
                # y_aggr.add_batch(y, patch[tio.LOCATION])
            
            # pred = pred_aggr.get_output_tensor()
            # y = y_aggr.get_output_tensor()
                loss_fn(pred, y)
                loss_fn.update()
                for m in metrics:
                    m(pred, y)
                    m.update()

    template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
    for m in metrics: template += ", " + str(m)
    print(template)

def seg_patch_train(
    dataloader, 
    model, 
    loss_fn,
    metrics, 
    optimizer, 
    callbacks, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):

    model.train()
    for batch, queue in enumerate(dataloader):
        patch_loader = torch.utils.data.DataLoader(queue, 
            batch_size  =dataloader.batch_size,  
            drop_last   =False, 
            shuffle     =False, 
            num_workers =0, 
            pin_memory  =True)
        for patch, (X, y) in enumerate(patch_loader):
            callbacks.on_batch_begin(batch * len(patch_loader) + patch)

            X, y = X.cuda(), y.cuda()

            # Compute prediction error
            pred = model(X)
            if use_deep_supervision:
                loss = loss_fn(pred, y, epoch)
                for m in metrics: m(pred[-1].detach(),y.detach(), epoch)
            else: 
                loss = loss_fn(pred, y)
                for m in metrics: m(pred.detach(),y.detach())

            # Backpropagation
            # for param in optimizer.parameters():
            #     param.grad = None
            optimizer.zero_grad() # set gradient to zero, why is that needed?
            loss.backward() # compute gradient
            optimizer.step() # apply gradient
            loss.detach()

            callbacks.on_batch_end(batch * len(patch_loader) + patch)

#---------------------------------------------------------------------------