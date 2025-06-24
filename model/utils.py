import os
import glob
import argparse
import math
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import transformers
import lightning as pl

from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any
from lightning.pytorch.strategies import FSDPStrategy, SingleDeviceStrategy, DeepSpeedStrategy, DDPStrategy, ModelParallelStrategy
from torch.optim.lr_scheduler import LRScheduler
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.trainer.states import RunningStage
from torchmetrics import Metric

class CosineLRSchedule(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup: Optional[int]=10000,
        max_lr: Optional[float]=1e-4,
        min_lr: Optional[float]=1e-6,
        max_steps: Optional[float]=150000
    ):
        self.warmup = warmup
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_steps = max_steps

        super(CosineLRSchedule, self).__init__(optimizer)
    
    # override
    def get_lr(self) ->list[float]:
        step = max(1, self._step_count)
        if step <= self.warmup:
            scale = step / self.warmup
            return [min(lr * scale, self.max_lr) for lr in self.base_lrs]
        else:
            scale = (self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                    (1.0 + math.cos(((step - self.warmup) / (max(self.max_steps, step) - self.warmup)) * math.pi))) / self.max_lr
            if scale * self.max_lr < self.min_lr:
                scale = self.min_lr / self.max_lr
            return [min(lr * scale, self.max_lr) for lr in self.base_lrs]

@rank_zero_only
def process_json(json_path: str, tgt: Optional[list[dict]]=[], mode: Optional[str]="read"):
    if mode == "read":
        with open(json_path, 'r') as f: return json.load(f)
    elif mode == "write":
        with open(json_path, 'w') as f: f.write(json.dumps(tgt, indent=4))
    elif mode == "append":
        if not os.path.exists(json_path):
            with open(json_path, 'w') as f: f.write(json.dumps(tgt, indent=4))
        else:
            data = []
            with open(json_path, 'r') as f: data = json.load(f)
            data.extend(tgt)
            with open(json_path, 'w') as f: f.write(json.dumps(data, indent=4))

def clean_metrics(metrics: dict):
    for v in metrics.values():
        if isinstance(v, Metric): v.reset()

def convert_metrics(metrics: dict) -> dict[str:float]:
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, Metric):
            new_metrics[k] = v.compute().item()
        else:
            new_metrics[k] = v
    return new_metrics


class CheckpointCallback(Callback):
    def __init__(self, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.train_config = train_config
        self.eval_start = train_config.eval_start
        self.eval_step = train_config.eval_step
        self.max_steps = train_config.max_steps
        self.top_k = train_config.top_k
        self.core_metric = train_config.core_metric
        self.ascending = train_config.ascending
        self.ensemble = train_config.ensemble
        self.ensemble_only = train_config.ensemble_only
        self.test_in_end = train_config.test_in_end
        self.save_best = train_config.save_best

        # json path
        self.pretrain_json_path = os.path.join(train_config.pretrained_ckpt_path, "finetune.json") if train_config.pretrained_ckpt_path != "" else None
        self.finetune_json_path = os.path.join(train_config.ckpt_path, "task.json")

        # remove previous ckpt
        if os.path.exists(self.finetune_json_path): os.remove(self.finetune_json_path)
        prev_ckpt = glob.glob(self.train_config.ckpt_path + "/*.bin")
        for _ckpt in prev_ckpt: os.remove(_ckpt)
        
        self.prev_step = -1
    
    def _safe_save(self, trainer: pl.Trainer, pl_module: pl.LightningModule, ckpt_path: str):
        if trainer.num_devices > 1 and isinstance(self.trainer.strategy, ModelParallelStrategy):
            sharded_sd = pl_module.model.state_dict()
            state_dict = {}
            for param_name, sharded_param in sharded_sd.items():
                full_param = sharded_param.full_tensor()
                if trainer.is_global_zero:
                    state_dict[param_name] = full_param.cpu()
                else:
                    del full_param
        else:
            state_dict = pl_module.model.state_dict()

        if trainer.global_rank == 0:
            if not os.path.exists(os.path.dirname(ckpt_path)):
                os.makedirs(os.path.dirname(ckpt_path))
            torch.save(state_dict, ckpt_path)
    
    def _safe_load(self, trainer: pl.Trainer, pl_module: pl.LightningModule, ckpt_path: str):
        pl_module.model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    def _safe_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, process_stage: Optional[RunningStage]=RunningStage.VALIDATING):
        _first_loop_iter = trainer._logger_connector._first_loop_iter
        trainer.training = False
        pl_module.eval()
        stage = trainer.state.stage
        trainer.state.stage = process_stage

        if process_stage == RunningStage.TESTING:
            trainer.strategy.barrier()
            trainer.test_loop.setup_data()
            trainer.test_loop.run()
        else:
            trainer.strategy.barrier()
            trainer.validate_loop.setup_data()
            trainer._run_stage()
        
        trainer.state.stage = stage
        trainer.training = True
        pl_module.train()
        trainer._logger_connector._epoch_end_reached = False
        trainer._logger_connector._first_loop_iter = _first_loop_iter
    
    def save(self, ckpt_name: str | int, trainer: pl.Trainer, pl_module: pl.LightningModule, record: Optional[bool]=True):
        self._safe_save(trainer, pl_module, os.path.join(trainer.default_root_dir, f"{ckpt_name}.bin"))
        metrics = convert_metrics(pl_module.eval_metrics)
        process_json(self.finetune_json_path, [{"name": str(ckpt_name), "metrics": metrics}], mode="append")
        if trainer.is_global_zero: print(f"Finish evaling at {trainer.global_step} steps, metrics: {metrics[self.core_metric]:.4f}")
    
    @rank_zero_only
    def _sort_and_remove(self, trainer: pl.Trainer, pl_module: pl.LightningModule, top_k: Optional[int]=None, target: Optional[str]="", only_sort: Optional[bool]=False):
        current: list[dict] = process_json(self.finetune_json_path, mode="read")
        if top_k is None: top_k = self.top_k

        current = sorted(current, key=lambda x: x["metrics"][self.core_metric], reverse=False if self.ascending else True)
        if not only_sort:
            if target == "":
                if len(current) > top_k:
                    for ckpt in current[top_k:]:
                        remove_tgt = os.path.join(trainer.default_root_dir, f"{ckpt["name"]}.bin")
                        if os.path.exists(remove_tgt) and trainer.is_global_zero: os.remove(remove_tgt)
                    
                    current = current[:top_k]
                    process_json(self.finetune_json_path, current, mode="write")
            else:
                tmp = []
                for ckpt in current:
                    if ckpt["name"] != target:
                        remove_tgt = os.path.join(trainer.default_root_dir, f"{ckpt['name']}.bin")
                        if os.path.exists(remove_tgt) and trainer.is_global_zero: os.remove(remove_tgt)
                    else:
                        tmp.append(ckpt)
                current = tmp
                process_json(self.finetune_json_path, current, mode="write")
            
        return current

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_step = trainer.global_step
        need_validate = (current_step >= self.eval_start and (current_step - self.eval_start) % self.eval_step == 0) or (current_step == self.max_steps)

        # for gradient accumulation
        need_validate = (current_step != self.prev_step) and need_validate
        self.prev_step = current_step

        if need_validate:
            self._safe_eval(trainer, pl_module)
            self.save(trainer.global_step, trainer, pl_module)

            # check current checkpoints
            self._sort_and_remove(trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module):
        # ensemble
        if self.ensemble:
            current: list[dict] = self._sort_and_remove(trainer, pl_module, only_sort=True)
            current = trainer.strategy.broadcast(current)

            ensemble_param = None
            for i, _data in enumerate(current[:self.top_k]):
                _name = _data["name"]
                model_path = os.path.join(trainer.default_root_dir, f"{_name}.bin")
                param = torch.load(model_path, map_location='cpu', weights_only=True)

                if i == 0:
                    ensemble_param = param
                    for k, v in ensemble_param.items(): ensemble_param[k] = ensemble_param[k].float()
                else:
                    for k, v in ensemble_param.items(): ensemble_param[k].mul_(i).add_(param[k].float()).div_(i + 1)
            
            pl_module.model.load_state_dict(ensemble_param)
            self._safe_eval(trainer, pl_module)
            self.save("Ensemble", trainer, pl_module)
        
        # get the best
        current: list[dict] = self._sort_and_remove(trainer, pl_module, top_k=1, target="Ensemble" if self.ensemble_only else "")
        current = trainer.strategy.broadcast(current)
        
        final_ckpt = current[0]["name"]
        if trainer.is_global_zero: os.rename(os.path.join(trainer.default_root_dir, f"{final_ckpt}.bin"), os.path.join(trainer.default_root_dir, "final.bin"))

        # test set validation
        if self.test_in_end:
            self._safe_load(trainer, pl_module, os.path.join(trainer.default_root_dir, "final.bin"))
            self._safe_eval(trainer, pl_module, RunningStage.TESTING)
            metrics = convert_metrics(pl_module.eval_metrics)

            if trainer.is_global_zero and self.pretrain_json_path is not None:
                process_json(
                    self.pretrain_json_path,
                    [{"dataset": self.train_config.data_path, "seed": self.train_config.seed, "metrics": metrics}],
                    mode="append"
                )
                print(f"Finish testing {self.train_config.data_name}, metrics: {metrics[self.core_metric]:.4f}")
            
            if not self.save_best and trainer.is_global_zero: os.remove(os.path.join(trainer.default_root_dir, "final.bin"))