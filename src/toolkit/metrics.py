#!/usr/bin/env python3
import torch
import time
from torch import Tensor
from collections import defaultdict
from avalanche.training.plugins import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue

class ClockLoggingPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()
    
    def before_eval(self, strategy, **kwargs):
        strategy.evaluator.publish_metric_value(
            MetricValue(
                "Metric",
                "mb_index",
                strategy.clock.train_exp_counter,
                x_plot=strategy.clock.train_iterations,
            )
        )


class TimeSinceStart(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def before_eval(self, strategy, **kwargs):
        strategy.evaluator.publish_metric_value(
            MetricValue(
                "Metric",
                "time",
                time.time() - self.start_time,
                x_plot=strategy.clock.train_iterations,
            )
        )

class GradientNormPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        
        self.gradient_norm = 0
    
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

def track_metrics(
    strategy: "SupervisedTemplate",
    metric_name, 
    metric_val,
    clock_plot
):
    strategy.evaluator.publish_metric_value(
        MetricValue(
            "Metric",
            metric_name,
            metric_val,
            x_plot=clock_plot 
        )
    )