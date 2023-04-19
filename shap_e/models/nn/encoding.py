import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn


def encode_position(version: str, *, position: torch.Tensor):
    if version == "v1":
        freqs = get_scales(0, 10, position.dtype, position.device).view(1, -1)
        freqs = position.reshape(-1, 1) * freqs
        return torch.cat([freqs.cos(), freqs.sin()], dim=1).reshape(*position.shape[:-1], -1)
    elif version == "nerf":
        return posenc_nerf(position, min_deg=0, max_deg=15)
    else:
        raise ValueError(version)


def encode_channels(version: str, *, channels: torch.Tensor):
    if version == "v1":
        freqs = get_scales(0, 10, channels.dtype, channels.device).view(1, -1)
        freqs = channels.reshape(-1, 1) * freqs
        return torch.cat([freqs.cos(), freqs.sin()], dim=1).reshape(*channels.shape[:-1], -1)
    elif version == "nerf":
        return posenc_nerf(channels, min_deg=0, max_deg=15)
    else:
        raise ValueError(version)


def position_encoding_channels(version: Optional[str] = None) -> int:
    if version is None:
        return 1
    return encode_position(version, position=torch.zeros(1, 1)).shape[-1]


def channel_encoding_channels(version: Optional[str] = None) -> int:
    if version is None:
        return 1
    return encode_channels(version, channels=torch.zeros(1, 1)).shape[-1]


class PosEmbLinear(nn.Linear):
    def __init__(
        self, posemb_version: Optional[str], in_features: int, out_features: int, **kwargs
    ):
        super().__init__(
            in_features * position_encoding_channels(posemb_version),
            out_features,
            **kwargs,
        )
        self.posemb_version = posemb_version

    def forward(self, x: torch.Tensor):
        if self.posemb_version is not None:
            x = encode_position(self.posemb_version, position=x)
        return super().forward(x)


class MultiviewPoseEmbedding(nn.Conv2d):
    def __init__(
        self,
        posemb_version: Optional[str],
        n_channels: int,
        out_features: int,
        stride: int = 1,
        **kwargs,
    ):
        in_features = (
            n_channels * channel_encoding_channels(version=posemb_version)
            + 3 * position_encoding_channels(version=posemb_version)
            + 3 * position_encoding_channels(version=posemb_version)
        )
        super().__init__(
            in_features,
            out_features,
            kernel_size=3,
            stride=stride,
            padding=1,
            **kwargs,
        )
        self.posemb_version = posemb_version

    def forward(
        self, channels: torch.Tensor, position: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        """
        :param channels: [batch_shape, inner_batch_shape, n_channels, height, width]
        :param position: [batch_shape, inner_batch_shape, 3, height, width]
        :param direction: [batch_shape, inner_batch_shape, 3, height, width]
        :return: [*batch_shape, out_features, height, width]
        """

        if self.posemb_version is not None:
            channels = channels.permute(0, 1, 3, 4, 2)
            position = position.permute(0, 1, 3, 4, 2)
            direction = direction.permute(0, 1, 3, 4, 2)
            channels = encode_channels(self.posemb_version, channels=channels).permute(
                0, 1, 4, 2, 3
            )
            direction = maybe_encode_direction(
                self.posemb_version, position=position, direction=direction
            ).permute(0, 1, 4, 2, 3)
            position = encode_position(self.posemb_version, position=position).permute(
                0, 1, 4, 2, 3
            )
        x = torch.cat([channels, position, direction], dim=-3)
        *batch_shape, in_features, height, width = x.shape
        return (
            super()
            .forward(x.view(-1, in_features, height, width))
            .view(*batch_shape, -1, height, width)
        )


class MultiviewPointCloudEmbedding(nn.Conv2d):
    def __init__(
        self,
        posemb_version: Optional[str],
        n_channels: int,
        out_features: int,
        stride: int = 1,
        **kwargs,
    ):
        in_features = (
            n_channels * channel_encoding_channels(version=posemb_version)
            + 3 * position_encoding_channels(version=posemb_version)
            + 3 * position_encoding_channels(version=posemb_version)
        )
        super().__init__(
            in_features,
            out_features,
            kernel_size=3,
            stride=stride,
            padding=1,
            **kwargs,
        )
        self.posemb_version = posemb_version
        self.register_parameter(
            "unk_token", nn.Parameter(torch.randn(in_features, **kwargs) * 0.01)
        )
        self.unk_token: torch.Tensor

    def forward(
        self,
        channels: torch.Tensor,
        origin: torch.Tensor,
        position: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param channels: [batch_shape, inner_batch_shape, n_channels, height, width]
        :param origin: [batch_shape, inner_batch_shape, 3, height, width]
        :param position: [batch_shape, inner_batch_shape, 3, height, width]
        :return: [*batch_shape, out_features, height, width]
        """

        if self.posemb_version is not None:
            channels = channels.permute(0, 1, 3, 4, 2)
            origin = origin.permute(0, 1, 3, 4, 2)
            position = position.permute(0, 1, 3, 4, 2)
            channels = encode_channels(self.posemb_version, channels=channels).permute(
                0, 1, 4, 2, 3
            )
            origin = encode_position(self.posemb_version, position=origin).permute(0, 1, 4, 2, 3)
            position = encode_position(self.posemb_version, position=position).permute(
                0, 1, 4, 2, 3
            )
        x = torch.cat([channels, origin, position], dim=-3)
        unk_token = torch.broadcast_to(self.unk_token.view(1, 1, -1, 1, 1), x.shape)
        x = torch.where(mask, x, unk_token)
        *batch_shape, in_features, height, width = x.shape
        return (
            super()
            .forward(x.view(-1, in_features, height, width))
            .view(*batch_shape, -1, height, width)
        )


def maybe_encode_direction(
    version: str,
    *,
    position: torch.Tensor,
    direction: Optional[torch.Tensor] = None,
):

    if version == "v1":
        sh_degree = 4
        if direction is None:
            return torch.zeros(*position.shape[:-1], sh_degree**2).to(position)
        return spherical_harmonics_basis(direction, sh_degree=sh_degree)
    elif version == "nerf":
        if direction is None:
            return torch.zeros_like(posenc_nerf(position, min_deg=0, max_deg=8))
        return posenc_nerf(direction, min_deg=0, max_deg=8)
    else:
        raise ValueError(version)


def posenc_nerf(x: torch.Tensor, min_deg: int = 0, max_deg: int = 15) -> torch.Tensor:
    """
    Concatenate x and its positional encodings, following NeRF.

    Reference: https://arxiv.org/pdf/2210.04628.pdf
    """
    if min_deg == max_deg:
        return x
    scales = get_scales(min_deg, max_deg, x.dtype, x.device)
    *shape, dim = x.shape
    xb = (x.reshape(-1, 1, dim) * scales.view(1, -1, 1)).reshape(*shape, -1)
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    emb = torch.cat([xb, xb + math.pi / 2.0], axis=-1).sin()
    return torch.cat([x, emb], dim=-1)


@lru_cache
def get_scales(
    min_deg: int,
    max_deg: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return 2.0 ** torch.arange(min_deg, max_deg, device=device, dtype=dtype)


def spherical_harmonics_basis(
    coords: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """
    Calculate the spherical harmonics basis

    :param coords: [batch_size, *shape, 3] of unit norm
    :param sh_degree: Spherical harmonics degree
    :return: [batch_size, *shape, sh_degree**2]
    """
    if sh_degree > 8:
        raise NotImplementedError

    batch_size, *shape, _ = coords.shape
    x, y, z = coords.reshape(-1, 3).split(1, dim=-1)
    x = x.squeeze(dim=-1)
    y = y.squeeze(dim=-1)
    z = z.squeeze(dim=-1)

    xy, xz, yz = x * y, x * z, y * z
    x2, y2, z2 = x * x, y * y, z * z
    x4, y4, z4 = x2 * x2, y2 * y2, z2 * z2
    x6, y6, z6 = x4 * x2, y4 * y2, z4 * z2
    xyz = xy * z

    # https://github.com/NVlabs/tiny-cuda-nn/blob/8575542682cb67cddfc748cc3d3cfc12593799aa/include/tiny-cuda-nn/encodings/spherical_harmonics.h#L76

    out = torch.zeros(x.shape[0], sh_degree**2, dtype=x.dtype, device=x.device)

    def _sh():
        out[:, 0] = 0.28209479177387814  # 1/(2*sqrt(pi))
        if sh_degree <= 1:
            return
        out[:, 1] = -0.48860251190291987 * y  # -sqrt(3)*y/(2*sqrt(pi))
        out[:, 2] = 0.48860251190291987 * z  # sqrt(3)*z/(2*sqrt(pi))
        out[:, 3] = -0.48860251190291987 * x  # -sqrt(3)*x/(2*sqrt(pi))
        if sh_degree <= 2:
            return
        out[:, 4] = 1.0925484305920792 * xy  # sqrt(15)*xy/(2*sqrt(pi))
        out[:, 5] = -1.0925484305920792 * yz  # -sqrt(15)*yz/(2*sqrt(pi))
        out[:, 6] = (
            0.94617469575755997 * z2 - 0.31539156525251999
        )  # sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
        out[:, 7] = -1.0925484305920792 * xz  # -sqrt(15)*xz/(2*sqrt(pi))
        out[:, 8] = (
            0.54627421529603959 * x2 - 0.54627421529603959 * y2
        )  # sqrt(15)*(x2 - y2)/(4*sqrt(pi))
        if sh_degree <= 3:
            return
        out[:, 9] = (
            0.59004358992664352 * y * (-3.0 * x2 + y2)
        )  # sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
        out[:, 10] = 2.8906114426405538 * xy * z  # sqrt(105)*xy*z/(2*sqrt(pi))
        out[:, 11] = (
            0.45704579946446572 * y * (1.0 - 5.0 * z2)
        )  # sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
        out[:, 12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0)  # sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
        out[:, 13] = (
            0.45704579946446572 * x * (1.0 - 5.0 * z2)
        )  # sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
        out[:, 14] = 1.4453057213202769 * z * (x2 - y2)  # sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
        out[:, 15] = (
            0.59004358992664352 * x * (-x2 + 3.0 * y2)
        )  # sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
        if sh_degree <= 4:
            return
        out[:, 16] = 2.5033429417967046 * xy * (x2 - y2)  # 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
        out[:, 17] = (
            1.7701307697799304 * yz * (-3.0 * x2 + y2)
        )  # 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
        out[:, 18] = (
            0.94617469575756008 * xy * (7.0 * z2 - 1.0)
        )  # 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
        out[:, 19] = (
            0.66904654355728921 * yz * (3.0 - 7.0 * z2)
        )  # 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
        out[:, 20] = (
            -3.1735664074561294 * z2 + 3.7024941420321507 * z4 + 0.31735664074561293
        )  # 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
        out[:, 21] = (
            0.66904654355728921 * xz * (3.0 - 7.0 * z2)
        )  # 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
        out[:, 22] = (
            0.47308734787878004 * (x2 - y2) * (7.0 * z2 - 1.0)
        )  # 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
        out[:, 23] = (
            1.7701307697799304 * xz * (-x2 + 3.0 * y2)
        )  # 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
        out[:, 24] = (
            -3.7550144126950569 * x2 * y2 + 0.62583573544917614 * x4 + 0.62583573544917614 * y4
        )  # 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
        if sh_degree <= 5:
            return
        out[:, 25] = (
            0.65638205684017015 * y * (10.0 * x2 * y2 - 5.0 * x4 - y4)
        )  # 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
        out[:, 26] = (
            8.3026492595241645 * xy * z * (x2 - y2)
        )  # 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
        out[:, 27] = (
            -0.48923829943525038 * y * (3.0 * x2 - y2) * (9.0 * z2 - 1.0)
        )  # -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
        out[:, 28] = (
            4.7935367849733241 * xy * z * (3.0 * z2 - 1.0)
        )  # sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
        out[:, 29] = (
            0.45294665119569694 * y * (14.0 * z2 - 21.0 * z4 - 1.0)
        )  # sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
        out[:, 30] = (
            0.1169503224534236 * z * (-70.0 * z2 + 63.0 * z4 + 15.0)
        )  # sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
        out[:, 31] = (
            0.45294665119569694 * x * (14.0 * z2 - 21.0 * z4 - 1.0)
        )  # sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
        out[:, 32] = (
            2.3967683924866621 * z * (x2 - y2) * (3.0 * z2 - 1.0)
        )  # sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
        out[:, 33] = (
            -0.48923829943525038 * x * (x2 - 3.0 * y2) * (9.0 * z2 - 1.0)
        )  # -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
        out[:, 34] = (
            2.0756623148810411 * z * (-6.0 * x2 * y2 + x4 + y4)
        )  # 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
        out[:, 35] = (
            0.65638205684017015 * x * (10.0 * x2 * y2 - x4 - 5.0 * y4)
        )  # 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
        if sh_degree <= 6:
            return
        out[:, 36] = (
            1.3663682103838286 * xy * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)
        )  # sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
        out[:, 37] = (
            2.3666191622317521 * yz * (10.0 * x2 * y2 - 5.0 * x4 - y4)
        )  # 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
        out[:, 38] = (
            2.0182596029148963 * xy * (x2 - y2) * (11.0 * z2 - 1.0)
        )  # 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
        out[:, 39] = (
            -0.92120525951492349 * yz * (3.0 * x2 - y2) * (11.0 * z2 - 3.0)
        )  # -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
        out[:, 40] = (
            0.92120525951492349 * xy * (-18.0 * z2 + 33.0 * z4 + 1.0)
        )  # sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
        out[:, 41] = (
            0.58262136251873131 * yz * (30.0 * z2 - 33.0 * z4 - 5.0)
        )  # sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
        out[:, 42] = (
            6.6747662381009842 * z2
            - 20.024298714302954 * z4
            + 14.684485723822165 * z6
            - 0.31784601133814211
        )  # sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
        out[:, 43] = (
            0.58262136251873131 * xz * (30.0 * z2 - 33.0 * z4 - 5.0)
        )  # sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
        out[:, 44] = (
            0.46060262975746175 * (x2 - y2) * (11.0 * z2 * (3.0 * z2 - 1.0) - 7.0 * z2 + 1.0)
        )  # sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
        out[:, 45] = (
            -0.92120525951492349 * xz * (x2 - 3.0 * y2) * (11.0 * z2 - 3.0)
        )  # -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
        out[:, 46] = (
            0.50456490072872406 * (11.0 * z2 - 1.0) * (-6.0 * x2 * y2 + x4 + y4)
        )  # 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
        out[:, 47] = (
            2.3666191622317521 * xz * (10.0 * x2 * y2 - x4 - 5.0 * y4)
        )  # 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
        out[:, 48] = (
            10.247761577878714 * x2 * y4
            - 10.247761577878714 * x4 * y2
            + 0.6831841051919143 * x6
            - 0.6831841051919143 * y6
        )  # sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
        if sh_degree <= 7:
            return
        out[:, 49] = (
            0.70716273252459627 * y * (-21.0 * x2 * y4 + 35.0 * x4 * y2 - 7.0 * x6 + y6)
        )  # 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
        out[:, 50] = (
            5.2919213236038001 * xy * z * (-10.0 * x2 * y2 + 3.0 * x4 + 3.0 * y4)
        )  # 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
        out[:, 51] = (
            -0.51891557872026028 * y * (13.0 * z2 - 1.0) * (-10.0 * x2 * y2 + 5.0 * x4 + y4)
        )  # -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
        out[:, 52] = (
            4.1513246297620823 * xy * z * (x2 - y2) * (13.0 * z2 - 3.0)
        )  # 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
        out[:, 53] = (
            -0.15645893386229404
            * y
            * (3.0 * x2 - y2)
            * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)
        )  # -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
        out[:, 54] = (
            0.44253269244498261 * xy * z * (-110.0 * z2 + 143.0 * z4 + 15.0)
        )  # 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
        out[:, 55] = (
            0.090331607582517306 * y * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0)
        )  # sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
        out[:, 56] = (
            0.068284276912004949 * z * (315.0 * z2 - 693.0 * z4 + 429.0 * z6 - 35.0)
        )  # sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
        out[:, 57] = (
            0.090331607582517306 * x * (-135.0 * z2 + 495.0 * z4 - 429.0 * z6 + 5.0)
        )  # sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
        out[:, 58] = (
            0.07375544874083044
            * z
            * (x2 - y2)
            * (143.0 * z2 * (3.0 * z2 - 1.0) - 187.0 * z2 + 45.0)
        )  # sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
        out[:, 59] = (
            -0.15645893386229404
            * x
            * (x2 - 3.0 * y2)
            * (13.0 * z2 * (11.0 * z2 - 3.0) - 27.0 * z2 + 3.0)
        )  # -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
        out[:, 60] = (
            1.0378311574405206 * z * (13.0 * z2 - 3.0) * (-6.0 * x2 * y2 + x4 + y4)
        )  # 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
        out[:, 61] = (
            -0.51891557872026028 * x * (13.0 * z2 - 1.0) * (-10.0 * x2 * y2 + x4 + 5.0 * y4)
        )  # -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
        out[:, 62] = (
            2.6459606618019 * z * (15.0 * x2 * y4 - 15.0 * x4 * y2 + x6 - y6)
        )  # 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
        out[:, 63] = (
            0.70716273252459627 * x * (-35.0 * x2 * y4 + 21.0 * x4 * y2 - x6 + 7.0 * y6)
        )  # 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))

    _sh()
    return out.view(batch_size, *shape, sh_degree**2)
