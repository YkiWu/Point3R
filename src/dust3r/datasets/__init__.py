from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .arkitscenes import ARKitScenes_Multi  # noqa
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .blendedmvs import BlendedMVS_Multi  # noqa
from .co3d import Co3d_Multi  # noqa
from .hypersim import HyperSim_Multi
from .megadepth import MegaDepth_Multi  # noqa
from .mvs_synth import MVS_Synth_Multi
from .omniobject3d import OmniObject3D_Multi
from .pointodyssey import PointOdyssey_Multi
from .scannet import ScanNet_Multi
from .scannetpp import ScanNetpp_Multi  # noqa
from .spring import Spring
from .vkitti2 import VirtualKITTI2_Multi  # noqa
from .waymo import Waymo_Multi  # noqa
from .wildrgbd import WildRGBD_Multi  # noqa


from accelerate import Accelerator


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=accelerator.num_processes,
            fixed_length=fixed_length
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )

    return data_loader
