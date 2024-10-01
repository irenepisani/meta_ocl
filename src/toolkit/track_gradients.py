import sys
import numpy as np
import torch
from typing import Optional

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

def get_grads_and_norms(
    data, 
    strategy, 
    all_minibatch: bool = False,
    norm_type: float = 2.0,
    **kwargs
    ):
    """ 
    Get gradients for a given set of data from buffer, train stream or val stream. 
    
    TO DO: code refactoring of this function.
    """
    
    if all_minibatch == False:
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
        
        self.grads_norm = None
        self.grads_norm_new = None
        self.grads_norm_old = None
        self.grads_norm_val = None
        self.grads_cos_sim_tr = None
        self.grads_cos_sim_vl = None


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
        _, self.grads_norm = get_grads_and_norms(strategy.mbatch, strategy, all_minibatch=True)
        #print(strategy.mbatch[-1]) # able this to check correct labels –

        if self.storage_policy is not None: 
            if len(self.storage_policy.buffer) == 0:
                return
                  
            # Given the currrent minibatch, separate old buffer data and new train data
            new_data = [strategy.mbatch[i][strategy.mbatch[-1] == 0] for i in range(len(strategy.mbatch))]
            old_data = [strategy.mbatch[i][strategy.mbatch[-1] == 1] for i in range(len(strategy.mbatch))]

            # Compute gradients 2-norm
            grads_new, self.grads_norm_new = get_grads_and_norms(new_data, strategy, **kwargs)
            grads_old, self.grads_norm_old = get_grads_and_norms(old_data, strategy, **kwargs)

            # Compute gradients cosine similarity
            cosine_similarity = torch.nn.CosineSimilarity(dim=0) 
            self.grads_cos_sim_tr = cosine_similarity(grads_old, grads_new) 
        
        # add metrics to loggers 
        track_metrics(strategy, "data_grads_norm",self.grads_norm.item(), strategy.clock.train_iterations)
        track_metrics(strategy, "new_data_grads_norm", self.grads_norm_new.item(), strategy.clock.train_iterations) 
        track_metrics(strategy, "old_data_grads_norm", self.grads_norm_old.item(), strategy.clock.train_iterations) 
        track_metrics(strategy, "sim_grads_norm_tr", self.grads_cos_sim_tr.item(), strategy.clock.train_iterations) 
        

'''
    def after_eval_iteration(self, strategy, **kwargs):
        val_data = strategy.current_eval_stream
        print(val_data)
        
        to do:
            - get forward output on current_eval_stream (that should be validation)
            - compute the following:
            #grads_val = get_grads(val_data, strategy, **kwargs)
            #self.grads_norm_val = get_grad_norm(strategy.model.parameters())
            #self.grads_cos_sim_vl = cosine_similarity(grads_old, grads_val)
            #track_metrics(strategy, "val_data_grads_norm", self.grads_norm_val.item(), strategy.clock.train_iterations) 
            #track_metrics(strategy, "sim_grads_norm_vl", self.grads_cos_sim_vl.item(), strategy.clock.train_iterations) 
        

– '''

    
   
        
       
       