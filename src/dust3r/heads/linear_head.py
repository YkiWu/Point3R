# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R & CUT3R

import torch
import torch.nn as nn
import torch.nn.functional as F
from dust3r.heads.postprocess import (
    postprocess,
    postprocess_desc,
    postprocess_pose_conf,
    postprocess_pose,
    reg_dense_conf,
)
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa
from dust3r.utils.geometry import geotrf
from dust3r.utils.camera import pose_encoding_to_camera, PoseDecoder
from dust3r.blocks import ConditionModulationBlock

class LinearPts3dPose(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(
        self, net, has_conf=False, has_pose=False, mlp_ratio=4.0
    ):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode
        self.has_conf = has_conf
        self.has_pose = has_pose

        self.proj = Mlp(
            net.dec_embed_dim,
            hidden_features=int(mlp_ratio * net.dec_embed_dim),
            out_features=(3 + has_conf) * self.patch_size**2,
        )
        if has_pose:
            self.pose_head = PoseDecoder(hidden_size=net.dec_embed_dim)
            self.final_transform = nn.ModuleList(
                [
                    ConditionModulationBlock(
                        net.dec_embed_dim,
                        net.dec_num_heads,
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        rope=net.rope,
                    )
                    for _ in range(2)
                ]
            )
            self.cross_proj = Mlp(
                net.dec_embed_dim,
                hidden_features=int(mlp_ratio * net.dec_embed_dim),
                out_features=(3 + has_conf) * self.patch_size**2,
            )

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape, **kwargs):
        H, W = img_shape
        tokens = decout[-1]
        if self.has_pose:
            pose_token = tokens[:, 0]
            tokens = tokens[:, 1:]
            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)
            cross_tokens = tokens
            for blk in self.final_transform:
                cross_tokens = blk(cross_tokens, pose_token, kwargs.get("pos"))

        with torch.cuda.amp.autocast(enabled=False):
            B, S, D = tokens.shape

            feat = self.proj(tokens)  # B,S,D
            feat = feat.transpose(-1, -2).view(
                B, -1, H // self.patch_size, W // self.patch_size
            )
            feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W
            final_output = postprocess(
                feat, self.depth_mode, self.conf_mode, pos_z=True
            )
            final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            final_output["conf_self"] = final_output.pop("conf")

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7

                cross_feat = self.cross_proj(cross_tokens)  # B,S,D
                cross_feat = cross_feat.transpose(-1, -2).view(
                    B, -1, H // self.patch_size, W // self.patch_size
                )
                cross_feat = F.pixel_shuffle(cross_feat, self.patch_size)  # B,3,H,W
                tmp = postprocess(cross_feat, self.depth_mode, self.conf_mode)
                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                final_output["conf"] = tmp.pop("conf")

            return final_output
