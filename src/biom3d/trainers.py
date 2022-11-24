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
# dino trainer

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def dino_train(
    dataloader, 
    scaler,
    model, 
    loss_fn,
    metrics, 
    optimizer, 
    callbacks, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):

    student, teacher = model
    
    torch.cuda.synchronize()
    t_start_epoch = time()
    print("[time] start epoch")

    for batch, images in enumerate(dataloader):
        
        callbacks.on_batch_begin(batch)
        images = [im.cuda(non_blocking=True) for im in images]

        torch.cuda.synchronize()
        t_data_loading = time()
        # print("[time] data loading:", t_data_loading-t_start_epoch)
        if t_data_loading-t_start_epoch > 1:
            print("SLOW!", batch, "[time] data loading:", t_data_loading-t_start_epoch)

        # Compute prediction error
        # teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
        # student_output = student(images)
        # del images
        # loss = loss_fn(student_output, teacher_output, epoch)

        with torch.cuda.amp.autocast(scaler is not None):
            teacher_output = teacher(images[:dataloader.dataset.nbof_global_patch])  # only the 2 global views pass through the teacher
            student_output = student(images)
            del images
            loss = loss_fn(teacher_output, student_output, epoch)

            with torch.no_grad():
                for m in metrics: m(teacher_output.detach(), student_output.detach(), epoch)

            # torch.cuda.synchronize()
            # t_model_pred = time()
            # print("[time] model prediction:", t_model_pred-t_data_loading)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # Backpropagation
        optimizer.zero_grad() # set gradient to zero, why is that needed?
        if scaler is not None:
            scaler.scale(loss).backward() # compute gradient
            scaler.unscale_(optimizer)

            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(student.parameters(), 12)
            clip_gradients(student, 3.0)
            cancel_gradients_last_layer(epoch, student, freeze_last_layer=1)

            # torch.cuda.synchronize()
            # t_backward = time()
            # print("[time] backward computation:", t_backward-t_model_pred)

            scaler.step(optimizer) # apply gradient
            scaler.update()
        else:
            loss.backward() # compute gradient

            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(student.parameters(), 12)
            clip_gradients(student, 3.0)
            cancel_gradients_last_layer(epoch, student, freeze_last_layer=1)

            # torch.cuda.synchronize()
            # t_backward = time()
            # print("[time] backward computation:", t_backward-t_model_pred)
            
            optimizer.step()

        # torch.cuda.synchronize()
        # t_optim_update = time()
        # print("[time] optimizer update:", t_optim_update-t_backward)

         # EMA update for the teacher
        with torch.no_grad():
            m = callbacks["momentum_scheduler"][epoch]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        loss.detach()
        del teacher_output, student_output, loss
        # del pred, y 

        callbacks.on_batch_end(batch)

        torch.cuda.synchronize()
        t_start_epoch = time()
        
    torch.cuda.empty_cache()

# def dino_validate(
#     dataloader,
#     model,
#     loss_fn,
#     metrics):
#     for m in [loss_fn]+metrics: m.reset() # reset metrics
#     model.eval() # set the module in evaluation mode (only useful for dropout or batchnorm like layers)
#     with torch.no_grad(): # set all the requires_grad flags to zeros
#         for X, y in dataloader:
#             X, y = X.cuda(), y.cuda()
#             with torch.cuda.amp.autocast():
#                 pred=model(X).detach()
#                 del X
#                 loss_fn(pred, y)
#                 loss_fn.update()
#                 for m in metrics:
#                     m(pred, y)
#                     m.update()
#                 del pred, y
#     torch.cuda.empty_cache()
#     template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
#     for m in metrics: template += ", " + str(m)
#     print(template)


#---------------------------------------------------------------------------
# triplet loss trainer

def triplet_train(   
    dataloader, 
    scaler,
    model, 
    loss_fn,
    optimizer, 
    callbacks, 
    metrics = None, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):
    model.train()
    for batch, (anc, pos, neg) in enumerate(dataloader):
        callbacks.on_batch_begin(batch)
        anc, pos, neg = anc.cuda(), pos.cuda(), neg.cuda()

        # Compute prediction error
        with torch.cuda.amp.autocast():
            pred_anc, pred_pos, pred_neg =  model(anc, use_encoder=True, use_last=False), model(pos, use_encoder=True, use_last=False), model(neg, use_encoder=True, use_last=False)
            loss = loss_fn(pred_anc, pred_pos, pred_neg)
        # for m in metrics: m(pred,y)

        # Backpropagation
        # optimizer.zero_grad() # set gradient to zero, why is that needed?
        # loss.backward() # compute gradient
        # optimizer.step() # apply gradient

        optimizer.zero_grad() # set gradient to zero, why is that needed?
        scaler.scale(loss).backward() # compute gradient

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)

        scaler.step(optimizer) # apply gradient
        scaler.update()

        loss.detach()
        del pred_anc, pred_pos, pred_neg
   
        callbacks.on_batch_end(batch)

def triplet_val(   
    dataloader, 
    model, 
    loss_fn,
    metrics = None, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):
    for m in [loss_fn]: m.reset() # reset metrics
    # for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval()
    with torch.no_grad():
        for anc, pos, neg in dataloader:
            anc, pos, neg = anc.cuda(), pos.cuda(), neg.cuda()

            # Compute prediction error
            pred_anc, pred_pos, pred_neg = model(anc, use_encoder=True, use_last=False), model(pos, use_encoder=True, use_last=False), model(neg, use_encoder=True, use_last=False)
            loss_fn(pred_anc, pred_pos, pred_neg)
            loss_fn.update()

            # for m in metrics: m(pred,y)
        template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
        print(template)


#---------------------------------------------------------------------------
# triplet segmentation trainer

def triplet_seg_train(   
    dataloader, 
    scaler,
    model, 
    loss_fn,
    optimizer, 
    callbacks, 
    metrics = None, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):
    model.train()

    seg_dataloader = iter(dataloader.seg_dataloader)
    for batch, (anc, pos, neg) in enumerate(dataloader.triplet_dataloader):
        callbacks.on_batch_begin(batch)
        anc, pos, neg = anc.cuda(), pos.cuda(), neg.cuda()

        # Compute triplet loss 
        with torch.cuda.amp.autocast():
            pred_anc, pred_pos, pred_neg = model(anc, use_encoder=True), model(pos, use_encoder=True), model(neg, use_encoder=True)
            # inputs = torch.cat([anc, pos, neg],dim=0)
            # pred = model(inputs)
            # pred_anc, pred_pos, pred_neg = pred.chunk(3)
            loss = loss_fn.triplet_loss(pred_anc, pred_pos, pred_neg)
        
        # Compute segmentation loss
        X, y = next(seg_dataloader)
        X, y = X.cuda(), y.cuda()

        # Compute prediction error
        with torch.cuda.amp.autocast():
            pred = model(X, use_encoder=False) 
            loss += loss_fn.dice_loss(pred, y)

        # for m in metrics: m(pred,y)
        loss_fn.update_val()

        # backprop
        optimizer.zero_grad() # set gradient to zero, why is that needed?
        scaler.scale(loss).backward() # compute gradient

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)

        scaler.step(optimizer) # apply gradient
        scaler.update()

        loss.detach()
        del pred_anc, pred_pos, pred_neg, pred, X, y
   
        callbacks.on_batch_end(batch)

#---------------------------------------------------------------------------
# arcface loss trainer

def arcface_train(   
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
    loss_fn.cuda()
    loss_fn.train()
    for batch, (img, idx) in enumerate(dataloader):
        callbacks.on_batch_begin(batch)
        img = img.cuda()
        idx = idx.cuda()

        with torch.cuda.amp.autocast(scaler is not None):
            # Compute prediction error
            pred_img = model(img, use_encoder=True)
            loss = loss_fn(pred_img, idx)
            # for m in metrics: m(pred,y)

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

        loss.detach()
        del pred_img
   
        callbacks.on_batch_end(batch)

def arcface_val(   
    dataloader, 
    model, 
    loss_fn,
    metrics = None, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):
    for m in [loss_fn]: m.reset() # reset metrics
    # for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval()
    loss_fn.eval()
    with torch.no_grad():
        for img, idx in dataloader:
            img = img.cuda()
            idx = idx.cuda()

            # Compute prediction error
            pred_img = model(img, use_encoder=True)
            loss_fn(pred_img, idx)
            loss_fn.update()
        template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
        print(template)

#---------------------------------------------------------------------------
# adversarial trainer

def adverse_train(
    dataloader, 
    model, 
    loss_fn,
    metrics, 
    optimizer, 
    callbacks, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):
    model.train()
    for batch, (img, rnd, msk) in enumerate(dataloader):
        callbacks.on_batch_begin(batch)
        img, rnd, msk = img.cuda(), rnd.cuda(), msk.cuda()

        # update discriminator
        fake = model(rnd)
        loss_fn.train()
        loss_fn(fake, msk, None)

        if epoch >= loss_fn.warmup:
            # update generator
            model.zero_grad()
            loss_fn.eval()
            pred = model(img)
            loss = loss_fn(fake, msk, pred)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
   
        callbacks.on_batch_end(batch)

#---------------------------------------------------------------------------
# co-training loss trainer

def cotrain_train(   
    dataloader, 
    model, 
    loss_fn,
    metrics, 
    optimizer, 
    callbacks, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):

    model.train()
    for batch, (X, y, X_arc, idx) in enumerate(dataloader):
        callbacks.on_batch_begin(batch)

        X = torch.cat([X, X_arc], dim=0)
        X, y, idx = X.cuda(), y.cuda(), idx.cuda()
        # Compute prediction error
        pred_enc, pred_dec = model(X)
        # bs = pred.shape[0]//2
        # pred_enc = pred[bs:]
        # pred_dec = pred[:bs]
        # pred_dec = model(X, use_encoder=False)
        # pred_dec = pred
        # pred_enc = pred_enc[:]
        # print(pred_enc.shape)
        # print(pred_dec[-1].shape)
        loss = loss_fn(pred_enc, pred_dec, y, idx, epoch)
        for m in metrics: m(pred_dec[-1].detach(),y.detach(), epoch)

        # Backpropagation
        optimizer.zero_grad() # set gradient to zero, why is that needed?
        loss.backward() # compute gradient
        optimizer.step() # apply gradient
        loss.detach()
   
        callbacks.on_batch_end(batch)

def cotrain_validate(dataloader, model, loss_fn, metrics):
    for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval() # set the module in evaluation mode (only useful for dropout or batchnorm like layers)
    with torch.no_grad(): # set all the requires_grad flags to zeros
        for X, y, _, _ in dataloader:
            # X, y, idx = X.cuda(), y.cuda(), idx.cuda()
            X, y = X.cuda(), y.cuda()
            # print(X.shape)
            # print(y.shape)
            pred_dec = model(X)

            # print(pred_dec.shape)
            pred_dec = pred_dec.detach()
            loss_fn(pred_dec, y)
            loss_fn.update()
            for m in metrics:
                m(pred_dec, y)
                m.update()
    template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
    for m in metrics: template += ", " + str(m)
    print(template)

#---------------------------------------------------------------------------
# model trainers for denoiseg

def denoiseg_train(
    dataloader, 
    model, 
    loss_fn,
    metrics, 
    optimizer, 
    callbacks, 
    epoch = None, # required by deep supervision
    use_deep_supervision=False):

    model.train()
    for batch, (img, msk, gt_denois) in enumerate(dataloader):
        callbacks.on_batch_begin(batch)
        img, msk, gt_denois = img.cuda(), msk.cuda(), gt_denois.cuda()

        # Compute prediction error
        pred = model(img)
        # if use_deep_supervision:
        #     loss = loss_fn(pred, y, epoch)
        #     for m in metrics: m(pred[-1].detach(),y.detach(), epoch)
        # else: 
        loss = loss_fn(pred, msk, gt_denois)
        # for m in metrics: m(pred.detach(),msk.detach())

        # Backpropagation
        optimizer.zero_grad() # set gradient to zero, why is that needed?
        loss.backward() # compute gradient
        optimizer.step() # apply gradient
        loss.detach()
   
        callbacks.on_batch_end(batch)

def denoiseg_val(dataloader, model, loss_fn, metrics):
    """
    similar to seg_val but only use one of the output channels
    """
    for m in [loss_fn]+metrics: m.reset() # reset metrics
    model.eval() # set the module in evaluation mode (only useful for dropout or batchnorm like layers)
    with torch.no_grad(): # set all the requires_grad flags to zeros
        for X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            pred=model(X).detach()
            pred, _ = torch.split(tensor=pred, split_size_or_sections=1, dim=1) # modif
            loss_fn(pred, y)
            loss_fn.update()
            for m in metrics:
                m(pred, y)
                m.update()
    template = "val error: avg loss {:.3f}".format(loss_fn.avg.item())
    for m in metrics: template += ", " + str(m)
    print(template)

#---------------------------------------------------------------------------