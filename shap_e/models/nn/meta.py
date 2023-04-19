"""
Meta-learning modules based on: https://github.com/tristandeleu/pytorch-meta

MIT License

Copyright (c) 2019-2020 Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import itertools
import re
from collections import OrderedDict

import torch.nn as nn

from shap_e.util.collections import AttrDict

__all__ = [
    "MetaModule",
    "subdict",
    "superdict",
    "leveldict",
    "leveliter",
    "batch_meta_parameters",
    "batch_meta_state_dict",
]


def subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ""):
        return dictionary
    key_re = re.compile(r"^{0}\.(.+)".format(re.escape(key)))
    return AttrDict(
        OrderedDict(
            (key_re.sub(r"\1", k), value)
            for (k, value) in dictionary.items()
            if key_re.match(k) is not None
        )
    )


def superdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ""):
        return dictionary
    return AttrDict(OrderedDict((key + "." + k, value) for (k, value) in dictionary.items()))


def leveldict(dictionary, depth=0):
    return AttrDict(leveliter(dictionary, depth=depth))


def leveliter(dictionary, depth=0):
    """
    depth == 0 is root
    """
    for key, value in dictionary.items():
        if key.count(".") == depth:
            yield key, value


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).

    Based on SIREN's torchmeta with some additional features/changes.

    All meta weights must not have the batch dimension, as they are later tiled
    to the given batch size after unsqueezing the first dimension (e.g. a
    weight of dimension [d_out x d_in] is tiled to have the dimension [batch x
    d_out x d_in]).  Requiring all meta weights to have a batch dimension of 1
    (e.g. [1 x d_out x d_in] from the earlier example) could be a more natural
    choice, but this results in silent failures.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta_state_dict = set()
        self._meta_params = set()

    def register_meta_buffer(self, name: str, param: nn.Parameter):
        """
        Registers a trainable or nontrainable parameter as a meta buffer. This
        can be later retrieved by meta_state_dict
        """
        self.register_buffer(name, param)
        self._meta_state_dict.add(name)

    def register_meta_parameter(self, name: str, parameter: nn.Parameter):
        """
        Registers a meta parameter so it is included in named_meta_parameters
        and meta_state_dict.
        """
        self.register_parameter(name, parameter)
        self._meta_params.add(name)
        self._meta_state_dict.add(name)

    def register_meta(self, name: str, parameter: nn.Parameter, trainable: bool = True):
        if trainable:
            self.register_meta_parameter(name, parameter)
        else:
            self.register_meta_buffer(name, parameter)

    def register(self, name: str, parameter: nn.Parameter, meta: bool, trainable: bool = True):
        if meta:
            if trainable:
                self.register_meta_parameter(name, parameter)
            else:
                self.register_meta_buffer(name, parameter)
        else:
            if trainable:
                self.register_parameter(name, parameter)
            else:
                self.register_buffer(name, parameter)

    def named_meta_parameters(self, prefix="", recurse=True):
        """
        Returns an iterator over all the names and meta parameters
        """

        def meta_iterator(module):
            meta = module._meta_params if isinstance(module, MetaModule) else set()
            for name, param in module._parameters.items():
                if name in meta:
                    yield name, param

        gen = self._named_members(
            meta_iterator,
            prefix=prefix,
            recurse=recurse,
        )
        for name, param in gen:
            yield name, param

    def named_nonmeta_parameters(self, prefix="", recurse=True):
        def _iterator(module):
            meta = module._meta_params if isinstance(module, MetaModule) else set()
            for name, param in module._parameters.items():
                if name not in meta:
                    yield name, param

        gen = self._named_members(
            _iterator,
            prefix=prefix,
            recurse=recurse,
        )
        for name, param in gen:
            yield name, param

    def nonmeta_parameters(self, prefix="", recurse=True):
        for _, param in self.named_nonmeta_parameters(prefix=prefix, recurse=recurse):
            yield param

    def meta_state_dict(self, prefix="", recurse=True):
        """
        Returns an iterator over all the names and meta parameters/buffers.

        One difference between module.state_dict() is that this preserves
        requires_grad, because we may want to compute the gradient w.r.t. meta
        buffers, but don't necessarily update them automatically.
        """

        def meta_iterator(module):
            meta = module._meta_state_dict if isinstance(module, MetaModule) else set()
            for name, param in itertools.chain(module._buffers.items(), module._parameters.items()):
                if name in meta:
                    yield name, param

        gen = self._named_members(
            meta_iterator,
            prefix=prefix,
            recurse=recurse,
        )
        return dict(gen)

    def update(self, params=None):
        """
        Updates the parameter list before the forward prop so that if `params`
        is None or doesn't have a certain key, the module uses the default
        parameter/buffer registered in the module.
        """
        if params is None:
            params = AttrDict()
        params = AttrDict(params)
        named_params = set([name for name, _ in self.named_parameters()])
        for name, param in self.named_parameters():
            params.setdefault(name, param)
        for name, param in self.state_dict().items():
            if name not in named_params:
                params.setdefault(name, param)
        return params


def batch_meta_parameters(net, batch_size):
    params = AttrDict()
    for name, param in net.named_meta_parameters():
        params[name] = param.clone().unsqueeze(0).repeat(batch_size, *[1] * len(param.shape))
    return params


def batch_meta_state_dict(net, batch_size):
    state_dict = AttrDict()
    meta_parameters = set([name for name, _ in net.named_meta_parameters()])
    for name, param in net.meta_state_dict().items():
        state_dict[name] = param.clone().unsqueeze(0).repeat(batch_size, *[1] * len(param.shape))
    return state_dict
