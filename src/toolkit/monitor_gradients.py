from typing import Optional

from avalanche.benchmarks.utils import concat_classification_datasets
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (ExemplarsBuffer, ExperienceBalancedBuffer)
import sys
import numpy as np
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from src.toolkit.grads_norm import get_grad_norm

class BufferLabels(DataAttribute):
    """
    Buffer labels labels are `DataAttribute`s that are automatically appended to the
    mini-batch.
    """

    def __init__(self, buffer_labels):
        super().__init__(buffer_labels, "buffer_labels", use_in_getitem=True)

class MonitorGradientsReplayPlugin(SupervisedPlugin, supports_distributed=True):

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

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        if len(self.storage_policy.buffer) == 0:
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None
        
        try:
            strategy.adapted_dataset = strategy.adapted_dataset.update_data_attribute(
            "buffer_labels",
            BufferLabels(np.full(len(strategy.adapted_dataset), 0))
            )
        except:
            print("error when adding buffer labels to new data")
            sys.exit(1)
        
        #try:
       
        for i in range(len(self.storage_policy.buffer_groups)):
            print(self.storage_policy.buffer_groups[i].max_size)
            self.storage_policy.buffer_groups[i].buffer = self.storage_policy.buffer_groups[i].buffer.update_data_attribute(
            "buffer_labels",
            BufferLabels(np.full(len(self.storage_policy.buffer_groups[i].buffer), 1))
            )
        
        #except:
            #print("error when adding buffer labels to old data")
            #sys.exit(1)
        
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

    def before_backward(self, strategy, **kwargs):

        if self.storage_policy is not None: 
            if len(self.storage_policy.buffer) == 0:
                return
                  
            # Separate old buffer data and new data
            new_data = [strategy.mbatch[i][strategy.mbatch[-1] == 0] for i in range(len(strategy.mbatch))]
            old_data = [strategy.mbatch[i][strategy.mbatch[-1] == 1] for i in range(len(strategy.mbatch))]

    
    def after_backward(self, strategy, **kwargs):
        
        norm_grads = get_grad_norm(strategy.model.parameters())
        #print(norm_grads)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)



'''
class MonitorGradientsReplayPlugin(SupervisedPlugin):
    
    def __init__(
        self,
        mem_size: int = 200,
        batch_size_mem: int = None,
        storage_policy: Optional["ClassBalancedBuffer"] = None
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.storage_policy = storage_policy
        self.gradient_norm = 0
        self.mbatch_old_data = None
        self.mbatch_new_data = None

    def after_train_dataset_adaptation(
        self,
        strategy: "SupervisedTemplate",
        **kwargs
    ):
        if self.storage_policy is not None: 
            if len(self.storage_policy.buffer) != 0:
                try:
                    #strategy.adapted_dataset = strategy.adapted_dataset.update_data_attribute("is_from_buffer", 0 )
                    #print("mem_size", self.mem_size)
                    #print("len storage policy buffer", len(self.storage_policy.buffer))
                    #print("batch_size_mem", self.batch_size_mem)
                    self.storage_policy.buffer = self.storage_policy.buffer.update_data_attribute("is_from_buffer", np.full(len(self.storage_policy.buffer), 1))
                    print("adapted buffer dataset", self.storage_policy.buffer.targets)
                    
                except ValueError:
                    sys.exit(1)
                    print("not able to add attribute to buffer dataset")
                try:
                    strategy.adapted_dataset = strategy.adapted_dataset.update_data_attribute("is_from_buffer", np.full(len(strategy.adapted_dataset),0))
                except ValueError:
                    sys.exit(1)
                    print("not able to add attribute to new dataset")
                
                print(strategy.adapted_dataset.is_from_buffer[0])
                print(self.storage_policy.buffer)
                sys.exit(1)

        else:
            print("Not possible to monitor gradients for buffer and stream data separatly")
    
    #def before_backward(self, strategy, **kwargs):
        
        #if self.storage_policy is not None: 
            #if len(self.storage_policy.buffer) != 0:
                #print("current minibatch",strategy.mbatch[-1])
                  
                # Separate buffer data and new data
                #self.mbatch_old_data = strategy.mbatch[strategy.mbatch[2] == 1]
                #self.mbatch_new_data = strategy.mbatch[strategy.mbatch[2] == 0]


    def after_backward(self, strategy, **kwargs):
        
        self.gradient_norm = 0
        for p in strategy.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                self.gradient_norm += param_norm.item() ** 2
        
        self.gradient_norm = self.gradient_norm ** 0.5
        
        strategy.evaluator.publish_metric_value(
            MetricValue(
                "Metric",
                "gradients norm",
                self.gradient_norm,
                x_plot=strategy.clock.train_iterations,
            )
        )
'''
'''

class GradientNormMonitor(SupervisedPlugin):
    def __init__(self):
        super().__init__()

    def before_backward(self, strategy, **kwargs):
        # Separate buffer data and new data
        buffer_data = strategy.mbatch[strategy.mbatch == 0]
        new_data = strategy.mbatch[strategy.mbatch == 1]

        # Compute gradients for buffer data
        buffer_data.requires_grad_(True)
        buffer_output = strategy.model(buffer_data)
        buffer_loss = strategy.criterion(buffer_output, strategy.mbatch[strategy.mbatch == 0])
        buffer_loss.backward(retain_graph=True)
        buffer_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in strategy.model.parameters() if p.grad is not None]))

        # Compute gradients for new data
        new_data.requires_grad_(True)
        new_output = strategy.model(new_data)
        new_loss = strategy.criterion(new_output, strategy.mbatch[strategy.mbatch == 1])
        new_loss.backward()
        new_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in strategy.model.parameters() if p.grad is not None]))

        # Log the gradient norms
        print(f"Buffer Data Gradient Norm: {buffer_grad_norm.item()}")
        print(f"New Data Gradient Norm: {new_grad_norm.item()}")

'''
