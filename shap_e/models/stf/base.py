from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from shap_e.models.query import Query
from shap_e.models.renderer import append_tensor
from shap_e.util.collections import AttrDict


class Model(ABC):
    @abstractmethod
    def forward(
        self,
        query: Query,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        """
        Predict an attribute given position
        """

    def forward_batched(
        self,
        query: Query,
        query_batch_size: int = 4096,
        params: Optional[Dict[str, torch.Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> AttrDict[str, Any]:
        if not query.position.numel():
            # Avoid torch.cat() of zero tensors.
            return self(query, params=params, options=options)

        if options.cache is None:
            created_cache = True
            options.cache = AttrDict()
        else:
            created_cache = False

        results_list = AttrDict()
        for i in range(0, query.position.shape[1], query_batch_size):
            out = self(
                query=query.map_tensors(lambda x, i=i: x[:, i : i + query_batch_size]),
                params=params,
                options=options,
            )
            results_list = results_list.combine(out, append_tensor)

        if created_cache:
            del options["cache"]

        return results_list.map(lambda key, tensor_list: torch.cat(tensor_list, dim=1))
