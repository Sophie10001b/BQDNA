import torch
import torch.nn as nn
from .modeling_vq import VQModel
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

def parallelize_setting(model: VQModel, device_mesh: DeviceMesh):
    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]

    # no tp parallel setting
    
    # examples from:
    # https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/tensor_parallel/parallelism.py
    if dp_mesh.size() > 1:
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        
        for layer_id, block in enumerate(model.pre_encoder.encoder_tower):
            reshard_after_forward = int(layer_id) < len(model.pre_encoder.encoder_tower) - 1
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward
            )
        
        for layer_id, block in enumerate(model.pre_encoder.norm_tower):
            reshard_after_forward = int(layer_id) < len(model.pre_encoder.norm_tower) - 1
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward
            )
        
        for layer_id, block in enumerate(model.encoder.downsample):
            reshard_after_forward = int(layer_id) < len(model.encoder.downsample) - 1
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward
            )
        
        for layer_id, block in enumerate(model.decoder.upsample):
            reshard_after_forward = int(layer_id) < len(model.decoder.upsample) - 1
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward
            )
        
        # fully_shard(model.pre_linear, **fsdp_config)
        # fully_shard(model.post_linear, **fsdp_config)
        # fully_shard(model.lm_head, **fsdp_config)
        # fully_shard(model.base_vocab, **fsdp_config)
        # fully_shard(model.codebook, **fsdp_config)

        fully_shard(model.encoder, **fsdp_config)
        fully_shard(model.decoder, **fsdp_config)
        fully_shard(model.vq, **fsdp_config)
        fully_shard(model, **fsdp_config)
    return model
        