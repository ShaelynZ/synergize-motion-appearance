from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_l1
from .pytorch_fid.fid_score import calculate_fid
from .cmp_lpips import calculate_lpips
from .extract import cmp_face_akd, cmp_face_aed
from .cmp_id_similarity import calculate_id_similarity
from .norm_pose.cmp_pose_accuracy import calculate_pose_distance

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_fid', 'calculate_l1', 'calculate_lpips', 'cmp_face_akd', 'cmp_face_aed', 'calculate_id_similarity', 'calculate_pose_distance']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must constain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
