# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R & CUT3R

from .linear_head import LinearPts3dPose
from .dpt_head import DPTPts3dPose


def head_factory(
    head_type,
    output_mode,
    net,
    has_conf=False,
    has_depth=False,
    has_pose_conf=False,
    has_pose=False,
):
    """ " build a prediction head for the decoder"""
    if head_type == "linear" and output_mode == "pts3d+pose":
        return LinearPts3dPose(net, has_conf, has_pose)
    elif head_type == "dpt" and output_mode == "pts3d+pose":
        return DPTPts3dPose(net, has_conf, has_pose)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
