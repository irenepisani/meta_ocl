from typing import (
    Dict,
    List,
    Any,
    Optional,
    Sequence,
    TypeVar,
    Callable,
    Union,
    overload,
)
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue

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

    def after_dataset_adaptation(
        self,
        strategy: "SupervisedTemplate",
        **kwargs
    ):
        if self.storage_policy is not None: 
            strategy.adapted_dataset = strategy.adapted_dataset.update_data_attribute("is_from_buffer", True)
            self.storage_policy.buffer = self.storage_policy.buffer.update_data_attiribute("is_from_buffer", False)
        else:
            print("Not possible to monitor gradients for buffer and stream data separatly")
    
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