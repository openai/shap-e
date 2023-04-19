import io
from contextlib import contextmanager
from typing import Any, BinaryIO, Iterator, Union

import blobfile as bf
import yaml

from shap_e.util.collections import AttrDict


def read_config(path_or_file: Union[str, io.IOBase]) -> Any:
    if isinstance(path_or_file, io.IOBase):
        obj = yaml.load(path_or_file, Loader=yaml.SafeLoader)
    else:
        with bf.BlobFile(path_or_file, "rb") as f:
            try:
                obj = yaml.load(f, Loader=yaml.SafeLoader)
            except Exception as exc:
                with bf.BlobFile(path_or_file, "rb") as f:
                    print(f.read())
                raise exc
    if isinstance(obj, dict):
        return AttrDict(obj)
    return obj


@contextmanager
def buffered_writer(raw_f: BinaryIO) -> Iterator[io.BufferedIOBase]:
    if isinstance(raw_f, io.BufferedIOBase):
        yield raw_f
    else:
        f = io.BufferedWriter(raw_f)
        yield f
        f.flush()
