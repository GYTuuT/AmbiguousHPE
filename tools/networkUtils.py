import contextlib
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import torch



# ----------------
@contextlib.contextmanager
def gpu_running_timer():
    """ Count the gpu running time.

    Usage:
        >>> with gpu_running_timer():
                ...(gpu operation)

    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    yield
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('Elapsed time: {:.3f} ms'.format(elapsed_time_ms))


# ----------------
def get_least_used_gpu():
    gpu_info = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader", 
        shell=True).decode("utf-8")
    gpu_memory = [int(x) for x in gpu_info.strip().split('\n')]
    return gpu_memory.index(min(gpu_memory))


## -----------------
def load_state_dict_partial(module:torch.nn.Module, state_dict:Dict):
    """When input state_dict and module.state_dict have the same key_value, load
       the same part of state_dict into module 
    """
    assert isinstance(state_dict, Dict)

    imwrite_state_dict = module.state_dict()
    for k, v in imwrite_state_dict.items():
        try:
            imwrite_state_dict[k] = state_dict[k]
        except:
            continue
    module.load_state_dict(imwrite_state_dict)

    return module



## =============
class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


# -----------------------------------
def print_module_summary(
        module:torch.nn.Module, 
        inputs:Union[Tuple, List], 
        max_nesting:int=3, 
        skip_redundant:bool=True):
    
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, inputs=_inputs, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_inputs = [t for t in e.inputs if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_inputs + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_inputs) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Input_shape', 'Output_shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        input_shapes = [str(list(e.inputs[0].shape) if ((e.inputs[0] is not None) and hasattr(e.inputs[0], 'shape')) else '-') for t in e.inputs]
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (input_shapes + ['-'])[0],
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        input_shapes = input_shapes + ['-'] * max(0, (len(output_shapes)-len(input_shapes)))
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', input_shapes[idx], output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs
