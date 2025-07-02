import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.point3r_blocks import (
    Block,
    MemoryDecoderBlock,
    DecoderBlock,
    PosDecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """
    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None

def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def from_dust3r_to_ours(state_dict):
    
    new_state_dict = OrderedDict()
        
    for k, v in state_dict.items():
        if k.startswith("dec_blocks2."):
            k = k.replace("dec_blocks2.", "dec_blocks_memory.")
        elif k.startswith("downstream_head1.dpt."):
            k = k.replace("downstream_head1.dpt.", "downstream_head.dpt_self.")
        elif k.startswith("downstream_head2.dpt."):
            k = k.replace("downstream_head2.dpt.", "downstream_head.dpt_cross.")
        name = k
        new_state_dict[name] = v
        
    return new_state_dict

def load_model(model_path, device):
    
    print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")  
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace("landscape_only=True", "landscape_only=False")
    assert "landscape_only=False" in args
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    print(s)
    return net.to(device)


class Point3RConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="dpt",
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        local_mem_size=256,
        memory_dec_num_heads=16,
        depth_head=False,
        pose_conf_head=False,
        pose_head=False,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.memory_dec_num_heads = memory_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.croco_kwargs = croco_kwargs

# thanks to CUT3R (https://github.com/CUT3R)
class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )
        self.write_blocks = nn.ModuleList(
            [
                PosDecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                PosDecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query) 
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


class Point3R(CroCoNet):
    config_class = Point3RConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: Point3RConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True)
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,)
        
        self._set_memory_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.memory_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )

        self._set_value_encoder(
            enc_depth=6, 
            enc_embed_dim=1024, 
            out_dim=1024, 
            enc_num_heads=16,
            mlp_ratio=4, 
            norm_layer=self.croco_args.get("norm_layer", None),
        )

        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.memory_attn_head = nn.Sequential(
            nn.Linear(self.enc_embed_dim+self.dec_embed_dim, self.enc_embed_dim+self.dec_embed_dim),
            nn.GELU(),
            nn.Linear(self.enc_embed_dim+self.dec_embed_dim, self.enc_embed_dim))

        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(Point3R, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.pts_patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                    rope3d=self.rope3d,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_memory_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_memory = dec_depth
        self.dec_embed_dim_memory = dec_embed_dim
        self.decoder_embed_memory = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_memory = nn.ModuleList(
            [
                MemoryDecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                    rope3d=self.rope3d,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_memory = norm_layer(dec_embed_dim)
    
    def _set_value_encoder(
        self,
        enc_depth, 
        enc_embed_dim, 
        out_dim, 
        enc_num_heads,
        mlp_ratio, 
        norm_layer
    ):
        self.value_encoder = nn.ModuleList(
            [
                Block(
                    enc_embed_dim, 
                    enc_num_heads, 
                    mlp_ratio, 
                    qkv_bias=True, 
                    norm_layer=norm_layer, 
                    rope=self.rope
                )
                for i in range(enc_depth)
            ]
        )
        self.value_norm = norm_layer(enc_embed_dim)
        self.value_out = nn.Linear(enc_embed_dim, out_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_memory") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_memory")] = value
        if not any(k.startswith("pts_patch_embed") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("patch_embed"):
                    new_ckpt[key.replace("patch_embed", "pts_patch_embed")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze): 
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "encoder": [
                self.patch_embed,
                self.enc_blocks,
                self.enc_norm,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_views(self, views, img_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        img_mask = torch.stack(
            [view["img_mask"] for view in views], dim=0
        ) 
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
        full_pos[img_masks_flat] += img_pos
        
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, i, mask_memory, f_memory, pos_memory, f_img, pos_img, f_pose, point3r_tag=False):
        if isinstance(f_memory, torch.Tensor):
            assert f_memory.shape[-1] == self.dec_embed_dim
        else:
            assert f_memory[-1].shape[-1] == self.dec_embed_dim
        
        final_output = [(f_memory, f_img)] 
        f_img = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)
        final_output.append((f_memory, f_img))
        
        for blk_memory, blk_img in zip(self.dec_blocks_memory, self.dec_blocks):
            f_memory, _ = blk_memory(i, *final_output[-1][::+1], mask_memory, pos_memory, pos_img, point3r_tag=point3r_tag)
            f_img, _ = blk_img(i, *final_output[-1][::-1], mask_memory, pos_img, pos_memory, point3r_tag=point3r_tag)
            final_output.append((f_memory, f_img))
        del final_output[1] 
        final_output[-1] = (
            self.dec_norm_memory(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_memory(self, image_tokens, image_pos):
        
        memory_feat = self.decoder_embed_memory(image_tokens)
        return memory_feat, None

    def _recurrent_rollout(
        self,
        i,
        mask_memory,
        memory_feat,
        memory_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        point3r_tag=False,
    ):
        new_memory_feat, dec = self._decoder(
            i,
            mask_memory,
            memory_feat, memory_pos, current_feat, current_pos, pose_feat,
            point3r_tag=point3r_tag,
        )
        new_memory_feat = new_memory_feat[-1]
        return new_memory_feat, dec

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)
  
    def enc_pts_value(self, pts, shape):
        out, pos = self.pts_patch_embed(pts.permute(0, 3, 1, 2), true_shape=shape)
        for block in self.value_encoder:
            out = block(out, pos)
        out = self.value_norm(out)
        out = self.value_out(out)
        return out

    def _forward_addmemory(
        self,
        i,
        pts3d,
        init_memory_feat,
        memory_feat,
        memory_pos,
        feat_i,
        dec_i,
        shape_i,
        ):
        bs, img_h, img_w, _ = pts3d.shape
        img_pos_len_h = img_h // 16
        img_pos_len_w = img_w // 16
        img_pos = pts3d.permute(0, 3, 1, 2)
        img_pos = img_pos.unfold(2, 16, 16)
        img_pos = img_pos.unfold(3, 16, 16)
        img_pos = img_pos.reshape(bs, 3, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1).permute(0, 2, 3, 1).reshape(bs, -1, 3)
        
        feat_key = self.memory_attn_head(torch.cat((feat_i, dec_i), dim=-1))
        feat_pts = self.enc_pts_value(pts3d, shape_i)
        memory_add = self.decoder_embed_memory(feat_key+feat_pts)
        memory_add = memory_add.float()

        if i == 0:
            memory_feat = memory_add
            init_memory_feat = memory_feat.clone().detach()
            chosen_pts = img_pos
        else:
            memory_feat = torch.cat((memory_feat, memory_add), dim=1)
            init_memory_feat = torch.cat((init_memory_feat, memory_add.clone().detach()), dim=1)
            chosen_pts = torch.cat((memory_pos, img_pos), dim=1)
          
        return memory_feat, chosen_pts, init_memory_feat, img_pos

    def _forward_addmemory_merge(
        self,
        i,
        pts3d,
        init_memory_feat,
        memory_feat,
        memory_pos,
        feat_i,
        dec_i,
        shape_i,
        ):
        bs, img_h, img_w, _ = pts3d.shape
        img_pos_len_h = img_h // 16
        img_pos_len_w = img_w // 16
        img_pos = pts3d.permute(0, 3, 1, 2)
        img_pos = img_pos.unfold(2, 16, 16)
        img_pos = img_pos.unfold(3, 16, 16)
        img_pos = img_pos.reshape(bs, 3, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1).permute(0, 2, 3, 1).reshape(bs, -1, 3)
        
        feat_key = self.memory_attn_head(torch.cat((feat_i, dec_i), dim=-1))
        feat_pts = self.enc_pts_value(pts3d, shape_i)
        memory_add = self.decoder_embed_memory(feat_key+feat_pts)
        memory_add = memory_add.float()

        if i == 0:
            memory_feat = memory_add
            init_memory_feat = memory_feat.clone().detach()
            chosen_pts = img_pos
        else:
            len_unit = 20
            memory_feat_list = []
            memory_pos_list = []
            for j in range(bs):
                memory_pos_j = memory_pos[j]
                memory_feat_j = memory_feat[j]
                img_pos_j = img_pos[j]
                new_feat_j = memory_add[j]
                unit_j = (torch.cat((memory_pos_j, img_pos_j), dim=0).max(dim=0).values - torch.cat((memory_pos_j, img_pos_j), dim=0).min(dim=0).values) / len_unit
                threshold_j = torch.norm(unit_j)
                distances = torch.cdist(img_pos_j, memory_pos_j)
                min_dists, min_indices = distances.min(dim=-1)
                mask_add = min_dists >= threshold_j
                mask_merge = ~mask_add
                if mask_merge.sum() > 0:
                    indices_merge = min_indices[mask_merge]
                    pos_merge = img_pos_j[mask_merge]
                    feat_merge = new_feat_j[mask_merge]
                    unique_indices, inverse_indices = torch.unique(indices_merge, return_inverse=True)
                    num_merge = unique_indices.shape[0]
                    pos_sum = torch.zeros((num_merge, 3), device=memory_pos_j.device)
                    feat_sum = torch.zeros((num_merge, 768), device=memory_feat_j.device)
                    count = torch.zeros((num_merge, 1), device=memory_feat_j.device)
                    pos_sum.index_add_(0, inverse_indices, pos_merge)
                    feat_sum.index_add_(0, inverse_indices, feat_merge)
                    count.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32).unsqueeze(1))
                    pos_avg = pos_sum / count
                    pos_avg = pos_avg.float()
                    feat_avg = feat_sum / count
                    feat_avg = feat_avg.float()
                    memory_pos_j[unique_indices] = pos_avg
                    memory_feat_j[unique_indices] = feat_avg
                if mask_add.sum() > 0:
                    pos_add = img_pos_j[mask_add]
                    feat_add = new_feat_j[mask_add]
                    memory_pos_j = torch.cat([memory_pos_j, pos_add], dim=0)
                    memory_feat_j = torch.cat([memory_feat_j, feat_add], dim=0)
                memory_feat_list.append(memory_feat_j)
                memory_pos_list.append(memory_pos_j)
            init_memory_feat_list = [memory_feat_index.clone().detach() for memory_feat_index in memory_feat_list]
        
        if i == 0:
            return memory_feat, chosen_pts, init_memory_feat, img_pos
        else:
            return memory_feat_list, memory_pos_list, init_memory_feat_list, img_pos

    def _forward_merge(self, views, point3r_tag=False):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        memory_feat, _ = self._init_memory(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_memory_feat = memory_feat.clone()
        ress = []
        pos_decode_img = None
        pos_decode_memory = None
        merge_tag = False
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if i >= 2:
                merge_tag = True
            if merge_tag:
                memory_len_max = max(f_memory_j.shape[0] for f_memory_j in memory_feat)
                f_memory_list_padded = []
                pos_memory_list_padded = []
                mask_memory_list_padded = []
                for j in range(len(memory_feat)):
                    f_memory_j = memory_feat[j]
                    pos_memory_j = pos_decode_memory[j]
                    padding_size = memory_len_max - f_memory_j.shape[0]
                    padding = torch.zeros(padding_size, f_memory_j.shape[1]).to(f_memory_j.device)
                    padding_pos = torch.zeros(padding_size, pos_memory_j.shape[1]).to(pos_memory_j.device)
                    mask_valid = torch.ones(f_memory_j.shape[0]).to(f_memory_j.device)
                    mask_invalid = torch.zeros(padding_size).to(f_memory_j.device)
                    padded_memory_j = torch.cat((f_memory_j, padding), dim=0)
                    padded_pos_memory_j = torch.cat((pos_memory_j, padding_pos), dim=0)
                    padded_mask_j = torch.cat((mask_valid, mask_invalid), dim=0)
                    f_memory_list_padded.append(padded_memory_j)
                    pos_memory_list_padded.append(padded_pos_memory_j)
                    mask_memory_list_padded.append(padded_mask_j)
                memory_feat = torch.stack(f_memory_list_padded, dim=0)
                pos_decode_memory = torch.stack(pos_memory_list_padded, dim=0)
                mask_memory = torch.stack(mask_memory_list_padded, dim=0)
            else:
                mask_memory = None
            
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = None
            else:
                pose_feat_i = None
                pose_pos_i = None

            new_memory_feat, dec = self._recurrent_rollout(
                i,
                mask_memory,
                memory_feat,
                pos_decode_memory,
                feat_i,
                pos_decode_img,
                pose_feat_i,
                pose_pos_i,
                point3r_tag=point3r_tag,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]

            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)

            update_mask_memory = torch.tensor([False]*memory_feat.shape[0], device=memory_feat.device)
            update_mask_memory = update_mask_memory[:, None, None].float()
            memory_feat = new_memory_feat * update_mask_memory + memory_feat * (1 - update_mask_memory)  
            update_mask_mem = torch.tensor([True]*mem.shape[0], device=mem.device)
            update_mask_mem = update_mask_mem[:, None, None].float()
            mem = new_mem * update_mask_mem + mem * (1 - update_mask_mem)
            if mask_memory is not None:
                memory_feat_new_list = []
                pos_decode_memory_new_list = []
                for j in range(mask_memory.shape[0]):
                    j_mask_memory = mask_memory[j]
                    j_mask_memory = j_mask_memory.bool()
                    j_memory_feat = memory_feat[j]
                    j_pos_decode_memory = pos_decode_memory[j]
                    j_memory_feat = j_memory_feat[j_mask_memory]
                    j_pos_decode_memory = j_pos_decode_memory[j_mask_memory]
                    memory_feat_new_list.append(j_memory_feat)
                    pos_decode_memory_new_list.append(j_pos_decode_memory)
                memory_feat = memory_feat_new_list
                pos_decode_memory = pos_decode_memory_new_list
                init_memory_feat = [memory_feat_in.clone().detach() for memory_feat_in in memory_feat]

            if point3r_tag:
                this_pts3d = res['pts3d_in_other_view'].clone().detach()
                if pos_decode_memory is not None:
                    if isinstance(pos_decode_memory, torch.Tensor):
                        pos_decode_memory = pos_decode_memory.clone().detach()
                    else:
                        pos_decode_memory = [pos_decode_memory_in.clone().detach() for pos_decode_memory_in in pos_decode_memory]
                memory_feat, pos_decode_memory, init_memory_feat, pos_decode_img= self._forward_addmemory_merge(
                    i,
                    pts3d=this_pts3d,
                    init_memory_feat=init_memory_feat,
                    memory_feat=memory_feat,
                    memory_pos=pos_decode_memory,
                    feat_i=feat_i.clone().detach(),
                    dec_i=dec[-1][:, 1:].clone().detach(),
                    shape_i=views[i]['true_shape'],
                )
                
        return ress, views

    def _forward(self, views, point3r_tag=False):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        memory_feat, _ = self._init_memory(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_memory_feat = memory_feat.clone()
        ress = []
        pos_decode_img = None
        pos_decode_memory = None
        
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            
            mask_memory = None

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = None
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_memory_feat, dec = self._recurrent_rollout(
                i,
                mask_memory,
                memory_feat,
                pos_decode_memory,
                feat_i,
                pos_decode_img,
                pose_feat_i,
                pose_pos_i,
                point3r_tag=point3r_tag,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)

            update_mask_memory = torch.tensor([False]*memory_feat.shape[0], device=memory_feat.device)
            update_mask_memory = update_mask_memory[:, None, None].float()
            memory_feat = new_memory_feat * update_mask_memory + memory_feat * (1 - update_mask_memory)  
            update_mask_mem = torch.tensor([True]*mem.shape[0], device=mem.device)
            update_mask_mem = update_mask_mem[:, None, None].float()
            mem = new_mem * update_mask_mem + mem * (1 - update_mask_mem)  

            if point3r_tag:
                this_pts3d = res['pts3d_in_other_view'].clone().detach()
                if pos_decode_memory is not None:
                    if isinstance(pos_decode_memory, torch.Tensor):
                        pos_decode_memory = pos_decode_memory.clone().detach()
                    else:
                        pos_decode_memory = [pos_decode_memory_in.clone().detach() for pos_decode_memory_in in pos_decode_memory]
                memory_feat, pos_decode_memory, init_memory_feat, pos_decode_img = self._forward_addmemory(
                    i,
                    pts3d=this_pts3d,
                    init_memory_feat=init_memory_feat,
                    memory_feat=memory_feat,
                    memory_pos=pos_decode_memory,
                    feat_i=feat_i.clone().detach(),
                    dec_i=dec[-1][:, 1:].clone().detach(),
                    shape_i=views[i]['true_shape'],
                )

        return ress, views

    def forward(self, views, point3r_tag=False):
        ress, views= self._forward_merge(views, point3r_tag=point3r_tag)
        return ARCroco3DStereoOutput(ress=ress, views=views)
        # stage1
        # ress, views = self._forward(views, point3r_tag=point3r_tag)
        # return ARCroco3DStereoOutput(ress=ress, views=views)

    