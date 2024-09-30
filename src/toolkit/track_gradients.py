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

def get_grads(data, strategy, **kwargs):
    
    strategy.optimizer.zero_grad()
    
    x_data, y_data, t_data = data[0], data[1], data[2]
    x_data, y_data = x_data.to(strategy.device), y_data.to(strategy.device)

    out = avalanche_forward(strategy.model, x_data, t_data)
    loss = strategy._criterion(out, y_data)
    loss.backward()
    
    grads_list = [
        p.grad.view(-1)
        if p.grad is not None
        else torch.zeros(p.numel(), device=strategy.device)
        for n, p in strategy.model.named_parameters()
    ]

    return torch.cat(grads_list)

class BufferLabels(DataAttribute):
    """
    Buffer labels are `DataAttribute`s that are 
    automatically appended to the minibatch.
    """
    
    def __init__(self, buffer_labels):
        super().__init__(buffer_labels, "buffer_labels", use_in_getitem=True)


class TrackGradientsPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Monitor gradients insight at each training iteration. 
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader
        
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
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        if len(self.storage_policy.buffer) == 0:
            print("here 1")
            return

        assert strategy.adapted_dataset is not None
        strategy.adapted_dataset = strategy.adapted_dataset.update_data_attribute(
            "buffer_labels", BufferLabels(np.full(len(strategy.adapted_dataset), 0)))
        
        for k, v in self.storage_policy.buffer_groups.items():
            self.storage_policy.buffer_groups[k].buffer = v.buffer.update_data_attribute(
            "buffer_labels", BufferLabels(np.full(len(v.buffer), 1)) )

            '''
            for i in range(len(self.storage_policy.buffer_groups)):
                # print(self.storage_policy.buffer_groups[i].max_size)
                self.storage_policy.buffer_groups[i].buffer = self.storage_policy.buffer_groups[i].buffer.update_data_attribute(
                "buffer_labels", BufferLabels(np.full(len(self.storage_policy.buffer_groups[i].buffer), 1)) )
            '''

    def after_backward(self, strategy, **kwargs): 
        """
        Monitor 2-Norm of the gradients computed 
        over the entire minibatch (both new data and old data)
        """
        self.grads_norm = get_grad_norm(strategy.model.parameters())
        #print(strategy.mbatch[-1])

        if self.storage_policy is not None: 
            if len(self.storage_policy.buffer) == 0:
                return
                  
            # Separate old buffer data and new train data
            new_data = [strategy.mbatch[i][strategy.mbatch[-1] == 0] for i in range(len(strategy.mbatch))]
            old_data = [strategy.mbatch[i][strategy.mbatch[-1] == 1] for i in range(len(strategy.mbatch))]
            #val_data = strategy.current_eval_stream
            #print(val_data)

            grads_new = get_grads(new_data, strategy, **kwargs)
            self.grads_norm_new = get_grad_norm(strategy.model.parameters())
            grads_old = get_grads(old_data, strategy, **kwargs)
            self.grads_norm_old = get_grad_norm(strategy.model.parameters())
            #grads_val = get_grads(val_data, strategy, **kwargs)
            #self.grads_norm_val = get_grad_norm(strategy.model.parameters())

            cosine_similarity = torch.nn.CosineSimilarity(dim=0) 
            self.grads_cos_sim_tr = cosine_similarity(grads_old, grads_new) 
            #self.grads_cos_sim_vl = cosine_similarity(grads_old, grads_val)
        
        track_metrics(strategy, "data_grads_norm",self.grads_norm.item(), strategy.clock.train_iterations)
        track_metrics(strategy, "new_data_grads_norm", self.grads_norm_new.item(), strategy.clock.train_iterations) 
        track_metrics(strategy, "old_data_grads_norm", self.grads_norm_old.item(), strategy.clock.train_iterations) 
        #track_metrics(strategy, "val_data_grads_norm", self.grads_norm_val.item(), strategy.clock.train_iterations) 
        track_metrics(strategy, "sim_grads_norm_tr", self.grads_cos_sim_tr.item(), strategy.clock.train_iterations) 
        #track_metrics(strategy, "sim_grads_norm_vl", self.grads_cos_sim_vl.item(), strategy.clock.train_iterations) 

'''
    def after_eval_iteration(self, strategy, **kwargs):
        val_data = strategy.current_eval_stream
        print(val_data)
    
    def after_eval_dataset_adaptation(self, strategy: "SupervisedTemplate",**kwargs):
        
        assert strategy.adapted_dataset is not None
        strategy.adapted_dataset = strategy.adapted_dataset.update_data_attribute(
            "buffer_labels", BufferLabels(np.full(len(strategy.adapted_dataset), 0)))

 '''
'''

class TrackGradientsPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Monitor gradients insight at each training iteration. 
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )
        print(len(self.storage_policy.buffer))
        self.grads_norm = None
        self.grads_norm_new = None
        self.grads_norm_old = None
        self.grads_cos_sim  = None

        if storage_policy is not None:
            self.storage_policy = storage_policy
    
    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        if len(self.storage_policy.buffer) == 0:
            print("here 1")
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size
        
        assert strategy.adapted_dataset is not None
        strategy.adapted_dataset = strategy.adapted_dataset.update_data_attribute(
            "buffer_labels", BufferLabels(np.full(len(strategy.adapted_dataset), 0)))
        
        for k, v in self.storage_policy.buffer_groups.items():
            self.storage_policy.buffer_groups[k].buffer = v.buffer.update_data_attribute(
            "buffer_labels", BufferLabels(np.full(len(v.buffer), 1)) )

            
            #for i in range(len(self.storage_policy.buffer_groups)):
                # print(self.storage_policy.buffer_groups[i].max_size)
                #self.storage_policy.buffer_groups[i].buffer = self.storage_policy.buffer_groups[i].buffer.update_data_attribute(
                #"buffer_labels", BufferLabels(np.full(len(self.storage_policy.buffer_groups[i].buffer), 1)) )
            

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks = True,
            batch_size = batch_size,
            batch_size_mem = batch_size_mem,
            task_balanced_dataloader = self.task_balanced_dataloader,
            num_workers = num_workers,
            shuffle = shuffle,
            drop_last = drop_last,
        )
                        
    def after_backward(self, strategy, **kwargs): 
        """
        Monitor 2-Norm of the gradients computed 
        over the entire minibatch (both new data and old data)
        """
        self.grads_norm = get_grad_norm(strategy.model.parameters())

        if self.storage_policy is not None: 
            if len(self.storage_policy.buffer) == 0:
                return
                  
            # Separate old buffer data and new data
            new_data = [strategy.mbatch[i][strategy.mbatch[-1] == 0] for i in range(len(strategy.mbatch))]
            old_data = [strategy.mbatch[i][strategy.mbatch[-1] == 1] for i in range(len(strategy.mbatch))]

            grads_new = get_grads(new_data, strategy, **kwargs)
            self.grads_norm_new = get_grad_norm(strategy.model.parameters())
            grads_old = get_grads(old_data, strategy, **kwargs)
            self.grads_norm_old = get_grad_norm(strategy.model.parameters())

            cosine_similarity = torch.nn.CosineSimilarity(dim=0) 
            self.grads_cos_sim = cosine_similarity(grads_old, grads_new) 
        
        track_metrics(strategy, "data_grads_norm",self.grads_norm.item(), strategy.clock.train_iterations)
        track_metrics(strategy, "new_data_grads_norm", self.grads_norm_new.item(), strategy.clock.train_iterations) 
        track_metrics(strategy, "old_data_grads_norm", self.grads_norm_old.item(), strategy.clock.train_iterations) 
        track_metrics(strategy, "sim_grads_norm", self.grads_cos_sim.item(), strategy.clock.train_iterations) 

    def after_training_exp(
        self,
        strategy: "SupervisedTemplate",
        **kwargs
    ):
        self.storage_policy.update(strategy, **kwargs)
'''
    
   
        
       
       