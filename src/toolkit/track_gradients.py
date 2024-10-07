import sys
import numpy as np
import torch
from typing import Optional
#!/usr/bin/env python3
import copy
import os
from collections import defaultdict

import torch
import torch.nn as nn

from avalanche.benchmarks.scenarios import OnlineCLExperience
from avalanche.models.dynamic_modules import IncrementalClassifier
from avalanche.benchmarks.utils import concat_classification_datasets
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (ExemplarsBuffer, ExperienceBalancedBuffer)
from avalanche.models import avalanche_forward

from src.toolkit.grads_norm import get_grad_norm
from src.toolkit.metrics import track_metrics

from typing import Union, Iterable, List, Dict, Tuple, Optional, cast

import torch
from torch import Tensor, inf
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def get_grad_normL2(strategy, norm_type: float = 2):
    """Returns the gradient norm of the model.
    Calculated the same way as torch.clip_grad_norm_"""

    # Params with grad
    parameters = model.parameters()
    if isinstance(model.parameters(), torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return None
    device = parameters[0].grad.device

    # calc norm
    total_norm = torch.norm(torch.stack(
        [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.item()

def get_grads_and_norms(
    strategy: "SupervisedTemplate",
    full_mb: bool = False,
    data: Optional[Tensor] = None,
    norm_type: float = 2.0,
    **kwargs
    ):
    """ 
    Get gradients for a given set of data from buffer, train stream or val stream. 
    
    TO DO: code refactoring of this function.
    """
    if strategy.is_training == False:
        #loss = strategy.loss
        #loss.backward()
            # ---> This is pytorch code to compute norm of gradients (used in gradient clipping)
        grads = [p.grad for p in strategy.model.parameters() if p.grad is not None]
        norm_type = float(norm_type) 
        if len(grads) == 0:
            return torch.tensor(0.)
        first_device = grads[0].device
        grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
            = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
        norms = []
        for ((device, _), ([grads], _)) in grouped_grads.items():
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])
        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
        return total_norm
    
    else: 
        if full_mb == False:
            strategy.optimizer.zero_grad()
            
            x_data, y_data, t_data = data[0], data[1], data[2]
            x_data, y_data = x_data.to(strategy.device), y_data.to(strategy.device)

            out = avalanche_forward(strategy.model, x_data, t_data)
            loss = strategy._criterion(out, y_data)
            loss.backward()
        
        # ---> This is pytorch code to compute norm of gradients (used in gradient clipping)
        grads = [p.grad for p in strategy.model.parameters() if p.grad is not None]
        norm_type = float(norm_type) 
        if len(grads) == 0:
            return torch.tensor(0.)
        first_device = grads[0].device
        grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
            = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
        norms = []
        for ((device, _), ([grads], _)) in grouped_grads.items():
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])
        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

        # --> This gem-style gradients computations (used in gradient gem strategy)
        grads_list = [
            p.grad.view(-1)
            if p.grad is not None
            else torch.zeros(p.numel(), device=strategy.device)
            for n, p in strategy.model.named_parameters()
        ]

        return torch.cat(grads_list), total_norm

class BufferLabels(DataAttribute):
    """
    Buffer labels are `DataAttribute`s that are 
    automatically appended to the minibatch.
        0 -> data is not from buffer 
        1 -> data is from buffer
    """
    
    def __init__(self, buffer_labels):
        super().__init__(buffer_labels, "buffer_labels", use_in_getitem=True)


class TrackGradientsPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Monitor gradients at each training iteration. 
    """

    def __init__(
        self,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        
        # grads metrics
        self.grads_norm = None
        self.grads_norm_new = None
        self.grads_norm_old = None
        self.grads_cos_sim_tr = None
        self.mean_eval_iter_norms = []


        self.training_model = None 


        if storage_policy is not None:
            self.storage_policy = storage_policy
    
    def after_train_dataset_adaptation(
        self,
        strategy: "SupervisedTemplate",
        **kwargs
    ):

        if len(self.storage_policy.buffer) == 0:
            return

        # update buffer labels on new train data from stream
        assert strategy.adapted_dataset is not None
        strategy.adapted_dataset = strategy.adapted_dataset.update_data_attribute(
            "buffer_labels", BufferLabels(np.full(len(strategy.adapted_dataset), 0)))
        
        # update buffer labels on old buffer data
        for k, v in self.storage_policy.buffer_groups.items():
            self.storage_policy.buffer_groups[k].buffer = v.buffer.update_data_attribute(
            "buffer_labels", BufferLabels(np.full(len(v.buffer), 1)) )

    def after_backward(self, strategy, **kwargs): 
        """
        Monitor 2-Norm of the gradients computed 
        over the entire minibatch (both new data and old data)
        """
        _, self.grads_norm = get_grads_and_norms(strategy, True)
        #print(strategy.mbatch[-1]) # able this to check correct labels –

        if self.storage_policy is not None: 
            if len(self.storage_policy.buffer) == 0:
                return
                  
            # Given the currrent minibatch, separate old buffer data and new train data
            new_data = [strategy.mbatch[i][strategy.mbatch[-1] == 0] for i in range(len(strategy.mbatch))]
            old_data = [strategy.mbatch[i][strategy.mbatch[-1] == 1] for i in range(len(strategy.mbatch))]

            # Compute gradients 2-norm
            grads_new, self.grads_norm_new = get_grads_and_norms(strategy, False, new_data)
            grads_old, self.grads_norm_old = get_grads_and_norms(strategy, False, old_data)

            # Compute gradients cosine similarity
            cosine_similarity = torch.nn.CosineSimilarity(dim=0) 
            self.grads_cos_sim_tr = cosine_similarity(grads_old, grads_new) 
        
        # add metrics to loggers 
        track_metrics(strategy, "data_grads_norm",self.grads_norm.item(), strategy.clock.train_iterations)
        track_metrics(strategy, "new_data_grads_norm", self.grads_norm_new.item(), strategy.clock.train_iterations) 
        track_metrics(strategy, "old_data_grads_norm", self.grads_norm_old.item(), strategy.clock.train_iterations) 
        track_metrics(strategy, "sim_grads_norm_tr", self.grads_cos_sim_tr.item(), strategy.clock.train_iterations) 

    '''    
    @torch.enable_grad()
    def before_eval(self, strategy, **kwargs):
        torch.set_grad_enabled(True)

    def before_eval_iteration(self, strategy, **kwargs):
        #torch.enable_grad()

        #def after_eval_iteration(self, strategy, **kwargs):
        #with torch.enable_grad():
        #weights.requires_grad_()
        with torch.set_grad_enabled(True):
            for param in strategy.model.parameters():
                param.requires_grad_()
            
            strategy.optimizer.zero_grad()
            #strategy.mb_x.requires_grad_
            data = strategy.mbatch
            #data.requires_grad(True)
            x_data, y_data, t_data = data[0], data[1], data[2]
            x_data, y_data = x_data.to(strategy.device), y_data.to(strategy.device)

            out = avalanche_forward(strategy.model, x_data, t_data)
            #out.requires_grad_()
            loss = strategy._criterion(out, y_data)
            #loss.backward()
        
            grads = torch.autograd.grad(loss, strategy.model.named_parameters())
            norm_type = float(norm_type) 
            if len(grads) == 0:
                return torch.tensor(0.)
            first_device = grads[0].device
            grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
                = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
            norms = []
            for ((device, _), ([grads], _)) in grouped_grads.items():
                norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])
            total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

            #_ , val_grads_norm = get_grads_and_norms(strategy)
            self.mean_eval_iter_norms.append( total_norm.item())
    
    def after_eval(self, strategy, **kwargs):
        track_metrics(strategy, "Eval/grads_norm_iter", np.mean(self.mean_eval_iter_norms.item(), strategy.clock.train_iterations))
        self.mean_eval_iter_norms   = []
        

    '''


    @torch.enable_grad()
    def after_eval_iteration(self, strategy, **kwargs):
        model_copy = copy.deepcopy(strategy.model)
        self.training_model = strategy.model
        strategy.model = model_copy
        
        strategy.model.train()
        optimizer = torch.optim.SGD(
            strategy.model.parameters(), lr=strategy.optimizer.param_groups[0]["lr"]
        )
        mbatch = strategy.mbatch
        strategy.mb_output = strategy.forward()
        loss = strategy.criterion()

        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad for p in strategy.model.parameters() if p.grad is not None]
        norm_type = float(2.0) 
        if len(grads) == 0:
            return torch.tensor(0.)
        first_device = grads[0].device
        grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
            = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
        norms = []
        for ((device, _), ([grads], _)) in grouped_grads.items():
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])
        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
        #print(total_norm)                
        
        #_ , val_grads_norm = get_grads_and_norms(strategy)
        self.mean_eval_iter_norms.append(total_norm.item())
        
        strategy.model.eval()
        strategy.model = self.training_model
    
    def after_eval(self, strategy, **kwargs):
        track_metrics(strategy, "ValidStream/mean_grads_norm_iter", np.mean(self.mean_eval_iter_norms), strategy.clock.train_iterations)
        self.mean_eval_iter_norms   = []

        
   
        
       
       