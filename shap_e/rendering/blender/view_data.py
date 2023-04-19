import itertools
import json
import zipfile
from typing import BinaryIO, List, Tuple

import numpy as np
from PIL import Image

from shap_e.rendering.view_data import Camera, ProjectiveCamera, ViewData


class BlenderViewData(ViewData):
    """
    Interact with a dataset zipfile exported by view_data.py.
    """

    def __init__(self, f_obj: BinaryIO):
        self.zipfile = zipfile.ZipFile(f_obj, mode="r")
        self.infos = []
        with self.zipfile.open("info.json", "r") as f:
            self.info = json.load(f)
        self.channels = list(self.info.get("channels", "RGBAD"))
        assert set("RGBA").issubset(
            set(self.channels)
        ), "The blender output should at least have RGBA images."
        names = set(x.filename for x in self.zipfile.infolist())
        for i in itertools.count():
            name = f"{i:05}.json"
            if name not in names:
                break
            with self.zipfile.open(name, "r") as f:
                self.infos.append(json.load(f))

    @property
    def num_views(self) -> int:
        return len(self.infos)

    @property
    def channel_names(self) -> List[str]:
        return list(self.channels)

    def load_view(self, index: int, channels: List[str]) -> Tuple[Camera, np.ndarray]:
        for ch in channels:
            if ch not in self.channel_names:
                raise ValueError(f"unsupported channel: {ch}")

        # Gather (a superset of) the requested channels.
        channel_map = {}
        if any(x in channels for x in "RGBA"):
            with self.zipfile.open(f"{index:05}.png", "r") as f:
                rgba = np.array(Image.open(f)).astype(np.float32) / 255.0
                channel_map.update(zip("RGBA", rgba.transpose([2, 0, 1])))
        if "D" in channels:
            with self.zipfile.open(f"{index:05}_depth.png", "r") as f:
                # Decode a 16-bit fixed-point number.
                fp = np.array(Image.open(f))
                inf_dist = fp == 0xFFFF
                channel_map["D"] = np.where(
                    inf_dist,
                    np.inf,
                    self.infos[index]["max_depth"] * (fp.astype(np.float32) / 65536),
                )
        if "MatAlpha" in channels:
            with self.zipfile.open(f"{index:05}_MatAlpha.png", "r") as f:
                channel_map["MatAlpha"] = np.array(Image.open(f)).astype(np.float32) / 65536

        # The order of channels is user-specified.
        combined = np.stack([channel_map[k] for k in channels], axis=-1)

        h, w, _ = combined.shape
        return self.camera(index, w, h), combined

    def camera(self, index: int, width: int, height: int) -> ProjectiveCamera:
        info = self.infos[index]
        return ProjectiveCamera(
            origin=np.array(info["origin"], dtype=np.float32),
            x=np.array(info["x"], dtype=np.float32),
            y=np.array(info["y"], dtype=np.float32),
            z=np.array(info["z"], dtype=np.float32),
            width=width,
            height=height,
            x_fov=info["x_fov"],
            y_fov=info["y_fov"],
        )
