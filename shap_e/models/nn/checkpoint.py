from typing import Callable, Iterable, Sequence, Union

import torch
from torch.cuda.amp import custom_bwd, custom_fwd


def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.length = length
        input_tensors = list(args[:length])
        input_params = list(args[length:])
        ctx.save_for_backward(*input_tensors, *input_params)
        with torch.no_grad():
            output_tensors = ctx.run_function(*input_tensors)
        return output_tensors

    @staticmethod
    @custom_bwd
    def backward(ctx, *output_grads):
        inputs = ctx.saved_tensors
        input_tensors = inputs[: ctx.length]
        input_params = inputs[ctx.length :]
        res = CheckpointFunctionGradFunction.apply(
            ctx.run_function,
            len(input_tensors),
            len(input_params),
            *input_tensors,
            *input_params,
            *output_grads
        )
        return (None, None) + res


class CheckpointFunctionGradFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, run_function, length_1, length_2, *args):
        ctx.run_function = run_function
        ctx.length_1 = length_1
        ctx.length_2 = length_2
        input_tensors = [x.detach().requires_grad_(True) for x in args[:length_1]]
        input_params = list(args[length_1 : length_1 + length_2])
        output_grads = list(args[length_1 + length_2 :])
        ctx.save_for_backward(*input_tensors, *input_params, *output_grads)

        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            input_tensors + input_params,
            output_grads,
            allow_unused=True,
        )
        return input_grads

    @staticmethod
    @custom_bwd
    def backward(ctx, *all_output_grads):
        args = ctx.saved_tensors
        input_tensors = [x.detach().requires_grad_(True) for x in args[: ctx.length_1]]
        input_params = list(args[ctx.length_1 : ctx.length_1 + ctx.length_2])
        output_grads = [
            x.detach().requires_grad_(True) for x in args[ctx.length_1 + ctx.length_2 :]
        ]

        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
            input_grads = torch.autograd.grad(
                output_tensors,
                input_tensors + input_params,
                output_grads,
                allow_unused=True,
                create_graph=True,
                retain_graph=True,
            )
        input_grads_grads = torch.autograd.grad(
            input_grads,
            input_tensors + input_params + output_grads,
            all_output_grads,
            allow_unused=True,
        )
        del input_grads
        return (None, None, None) + input_grads_grads
