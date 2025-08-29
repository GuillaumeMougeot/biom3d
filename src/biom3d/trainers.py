"""
The Trainers are Python functions that take as input a dataloader, a model, a loss function and an optimizer function to start a training process. Optionally, a list of biom3d.metrics.Metric and a biom3d.callback.Callbacks can be provided to the trainer to enrich the training loop.

Validaters, which optionally perform validation in the end of each epoch, are also defined in biom3d.trainers.
"""
# TODO: re-structure with classes maybe? Don't feel necessary for the moment, maybe to define an interface to ease extension.

import torch 
from tqdm import tqdm
from time import time 
from contextlib import nullcontext

from typing import Any
from torch.utils.data.dataloader import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.nn import Module
from biom3d.metrics import Metric
from biom3d.callbacks import Callbacks
#---------------------------------------------------------------------------
# model trainers for segmentation

def seg_train(
    dataloader:DataLoader, 
    scaler:GradScaler,
    model:Module, 
    loss_fn:Metric,
    metrics:list[Metric], 
    optimizer:Optimizer, 
    callbacks:Callbacks, 
    epoch:int | None = None, # required by deep supervision
    use_deep_supervision:bool=False,
    )->None:
    """
    Train a segmentation model.
    
    Call the dataloader to get a batch of images and masks, pass through the model, compute the loss using model output and masks, update model parameters. 

    Work with both CUDA, Metal or CPU. CPU is much slower.

    Work with half precision (fp16, CUDA only) and with standard precision (fp32).

    Use gradient clipping during backpropagation. 

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader for training data. A Dataloader is a Python class with an overloaded `__getitem__` method. In this case, `__getitem__` should return a batch of images and a batch of masks.
    scaler : torch.amp.GradScaler
        For halp precision.
    model : torch.nn.Module
        The model to train.
    loss_fn : biom3d.metrics.Metric
        The loss function.
    metrics : list of biom3d.metrics.Metric
        List of metrics to compute during training.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    callbacks : biom3d.callbacks.Callbacks
        Callbacks to be called during training.
    epoch : int, optional
        Current epoch number, required for deep supervision.
    use_deep_supervision : bool, default=False
        If True, deep supervision is used during training.

    Returns
    -------
    None
    """
    model.train()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    t_start_epoch = time()
    print("[time] start epoch")

    for batch, (X, y) in enumerate(dataloader):
        
        callbacks.on_batch_begin(batch)
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            X, y = X.to('mps'), y.to('mps')
            torch.mps.synchronize()
        t_data_loading = time()

        batch_duration = t_data_loading - t_start_epoch
        if batch_duration > 1:
            print(f"[Warning] Batch {batch} took {batch_duration:.2f}s — possible slowdown.")

        # Compute prediction error        
        with torch.amp.autocast("cuda") if scaler is not None and torch.cuda.is_available() else nullcontext():
            pred = model(X); del X
            loss = loss_fn(pred, y)
            with torch.no_grad():
                if use_deep_supervision:
                    for m in metrics: m(pred[-1],y)
                else: 
                    for m in metrics: m(pred,y)

        # Backpropagation
        optimizer.zero_grad() # set gradient to zero, why is that needed?

        if scaler is not None:
            scaler.scale(loss).backward() # compute gradient

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)

            scaler.step(optimizer) # apply gradient
            scaler.update()
        else: 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()

        del loss, pred, y 

        callbacks.on_batch_end(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_start_epoch = time()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def seg_validate(
    dataloader:DataLoader,
    model:Module,
    loss_fn:Metric,
    metrics:list[Metric],
    use_fp16:bool,
    use_deep_supervision:bool=False,
    )->None:
    """
    Validate a segmentation model.

    Call the validation dataloader to get a batch of images and masks, pass through the model, compute the loss using model output and masks.

    Work with both CUDA, Metal or CPU. CPU is much slower.

    Work with half precision (fp16, CUDA only) and with standard precision (fp32).
    
    Parameters
    ----------
    dataloader : DataLoader
        DataLoader for validation data. A Dataloader is a Python class with an overloaded `__getitem__` method. In this case, `__getitem__` should return a batch of images and a batch of masks.
    model : torch.nn.Module
        The model to validate.
    loss_fn : biom3d.metrics.Metric
        The validation loss function.
    metrics : list of biom3d.metrics.Metric
        List of metrics to compute during validation.
    use_fp16 : bool
        Flag to indicate if half-precision (fp16) is used.
    use_deep_supervision : bool, default=False
        If True, deep supervision is used during validation.

    Returns
    -------
    None
    """
    for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval() # set the module in evaluation mode (only useful for dropout or batchnorm like layers)
    with torch.no_grad(): # set all the requires_grad flags to zeros
        for X, y in dataloader:

            # with CUDA
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            elif torch.backends.mps.is_available():
                X,y = X.to('mps'), y.to('mps')
            
            with torch.amp.autocast("cuda") if use_fp16 and torch.cuda.is_available else nullcontext():
                pred=model(X)
                del X
                loss_fn(pred, y)
                loss_fn.update()
                for m in metrics:
                    if use_deep_supervision:
                        m(pred[-1],y)
                    else:
                        m(pred, y)
                    m.update()
            del pred, y
            
                
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
    for m in metrics: template += ", " + str(m)
    print(template)

#---------------------------------------------------------------------------
# model trainers for segmentation with patches 

def seg_patch_validate(dataloader:DataLoader, 
                       model:Module, 
                       loss_fn:Metric, 
                       metrics:list[Metric],
                       **kwargs:dict[str,Any],
                       )->None:
    """
    Validate the segmentation model with TorchIO patch-based approach.

    Parameters
    ----------
    dataloader : TorchIO DataLoader
        TorchIO DataLoader (such as generated using biom3d.datasets.semseg_torchio) containing validation data in patches. A Dataloader is a Python class with an overloaded `__getitem__` method. In this case, `__getitem__` should return a batch of images and a batch of masks.
    model : torch.nn.Module
        The model to validate.
    loss_fn : biom3d.metrics.Metric
        The validation loss function.
    metrics : list of biom3d.metrics.Metric
        List of metrics to compute during validation.
    **kwargs: dict from str to any
        Just for compatibility.

    Returns
    -------
    None
    """
    print("Start validation...")
    for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval() # set the module in evaluation mode (only useful for dropout or batchnorm like layers)
    with torch.no_grad(): # set all the requires_grad flags to zeros
        for it in tqdm(dataloader):
            patch_loader = torch.utils.data.DataLoader(it, batch_size=dataloader.batch_size,num_workers=0)
            for (X,y) in patch_loader:
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                elif torch.backends.mps.is_available():
                    X,y = X.to('mps'), y.to('mps')
                pred=model(X).detach()

                loss_fn(pred, y)
                loss_fn.update()
                for m in metrics:
                    m(pred, y)
                    m.update()

    template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
    for m in metrics: template += ", " + str(m)
    print(template)

def seg_patch_train(
    dataloader:DataLoader, 
    model:Module, 
    loss_fn:Metric,
    metrics:list[Metric], 
    optimizer:Optimizer, 
    callbacks:Callbacks, 
    epoch: int | None = None, # required by deep supervision
    use_deep_supervision:bool=False,
    **kwargs:dict[str,Any],
    )->None:
    """
    Train the segmentation model using a TorchIO patch-based approach.

    Parameters
    ----------
    dataloader : TorchIO DataLoader
        TorchIO DataLoader (such as generated using biom3d.datasets.semseg_torchio) containing training data in patches. A Dataloader is a Python class with an overloaded `__getitem__` method. In this case, `__getitem__` should return a batch of images and a batch of masks.
    model : torch.nn.Module
        The model to train.
    loss_fn : biom3d.metrics.Metric
        The loss function.
    metrics : list of metrics
        List of metrics to calculate during patch-based training.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    callbacks : biom3d.callbacks.Callbacks
        Callbacks to be called during training.
    epoch : int, optional
        Current epoch number, required for deep supervision.
    use_deep_supervision : bool, default=False
        If True, deep supervision is used during training.
    **kwargs: dict from str to any
        Just for compatibility

    Returns
    -------
    None
    """
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

            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            elif torch.backends.mps.is_available():
                X, y = X.to('mps'), y.to('mps')

            # Compute prediction error
            pred = model(X)
            if use_deep_supervision:
                loss = loss_fn(pred, y, epoch)
                for m in metrics: m(pred[-1].detach(),y.detach(), epoch)
            else: 
                loss = loss_fn(pred, y)
                for m in metrics: m(pred.detach(),y.detach())

            # Backpropagation :
            optimizer.zero_grad() # set gradient to zero, why is that needed?
            loss.backward() # compute gradient
            optimizer.step() # apply gradient
            loss.detach()

            callbacks.on_batch_end(batch * len(patch_loader) + patch)

#---------------------------------------------------------------------------