from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional
from typing import OrderedDict, Generic, TypeVar

K = TypeVar('K')
V = TypeVar('V')

class AttrDict(OrderedDict[K, V], Generic[K, V]):
    """
    An attribute dictionary that automatically handles nested keys joined by "/".

    Originally copied from: https://stackoverflow.com/questions/3031219/recursively-access-dict-via-attributes-as-well-as-index-access
    """

    MARKER = object()

    # pylint: disable=super-init-not-called
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            for key, value in kwargs.items():
                self.__setitem__(key, value)
        else:
            assert len(args) == 1
            assert isinstance(args[0], (dict, AttrDict))
            for key, value in args[0].items():
                self.__setitem__(key, value)

    def __contains__(self, key):
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            return key in self and next_key in self[key]
        return super(AttrDict, self).__contains__(key)

    def __setitem__(self, key, value):
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            if key not in self:
                self[key] = AttrDict()
            self[key].__setitem__(next_key, value)
            return

        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(**value)
        if isinstance(value, list):
            value = [AttrDict(val) if isinstance(val, dict) else val for val in value]
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            val = self[key]
            if not isinstance(val, AttrDict):
                raise ValueError
            return val.__getitem__(next_key)

        return self.get(key, None)

    def all_keys(
        self,
        leaves_only: bool = False,
        parent: Optional[str] = None,
    ) -> List[str]:
        keys = []
        for key in self.keys():
            cur = key if parent is None else f"{parent}/{key}"
            if not leaves_only or not isinstance(self[key], dict):
                keys.append(cur)
            if isinstance(self[key], dict):
                keys.extend(self[key].all_keys(leaves_only=leaves_only, parent=cur))
        return keys

    def dumpable(self, strip=True):
        """
        Casts into OrderedDict and removes internal attributes
        """

        def _dump(val):
            if isinstance(val, AttrDict):
                return val.dumpable()
            elif isinstance(val, list):
                return [_dump(v) for v in val]
            return val

        if strip:
            return {k: _dump(v) for k, v in self.items() if not k.startswith("_")}
        return {k: _dump(v if not k.startswith("_") else repr(v)) for k, v in self.items()}

    def map(
        self,
        map_fn: Callable[[Any, Any], Any],
        should_map: Optional[Callable[[Any, Any], bool]] = None,
    ) -> "AttrDict":
        """
        Creates a copy of self where some or all values are transformed by
        map_fn.

        :param should_map: If provided, only those values that evaluate to true
            are converted; otherwise, all values are mapped.
        """

        def _apply(key, val):
            if isinstance(val, AttrDict):
                return val.map(map_fn, should_map)
            elif should_map is None or should_map(key, val):
                return map_fn(key, val)
            return val

        return AttrDict({k: _apply(k, v) for k, v in self.items()})

    def __eq__(self, other):
        return self.keys() == other.keys() and all(self[k] == other[k] for k in self.keys())

    def combine(
        self,
        other: Dict[str, Any],
        combine_fn: Callable[[Optional[Any], Optional[Any]], Any],
    ) -> "AttrDict":
        """
        Some values may be missing, but the dictionary structures must be the
        same.

        :param combine_fn: a (possibly non-commutative) function to combine the
            values
        """

        def _apply(val, other_val):
            if val is not None and isinstance(val, AttrDict):
                assert isinstance(other_val, AttrDict)
                return val.combine(other_val, combine_fn)
            return combine_fn(val, other_val)

        # TODO nit: this changes the ordering..
        keys = self.keys() | other.keys()
        return AttrDict({k: _apply(self[k], other[k]) for k in keys})

    __setattr__, __getattr__ = __setitem__, __getitem__
