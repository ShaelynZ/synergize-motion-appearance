import os
import cv2
import argparse
import imageio
import ffmpeg
import torch
import yaml
import numpy as np
import sys
sys.path.insert(0, ".")

from copy import deepcopy
from tqdm.auto import tqdm
from shutil import copyfileobj
from scipy.spatial import ConvexHull
from tempfile import NamedTemporaryFile
from torchvision.transforms.functional import normalize

from basicsr.archs import build_network
from basicsr.utils.options import ordered_yaml
from basicsr.utils import img2tensor, tensor2img

# relative motion
def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False, adjust_shape_movement=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    print(f'Loading {net.__class__.__name__} model from {load_path}.')
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)

    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            print('Loading: params_ema does not exist, use params.')
        load_net = load_net[param_key]

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)
    return net

def find_best_frame(source, driving, cpu=False):
    import face_alignment  # type: ignore (local file)
    from scipy.spatial import ConvexHull

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except Exception as e:
            print(e)
    return frame_num

def make_animation(source_image, driving_video, net_g, motion_estimator, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        driving_imgs = []

        source_img = source_image.unsqueeze(0)
        driving_video = [frame.unsqueeze(0) for frame in driving_video]
        if not cpu:
            source_img = source_img.cuda()
            driving_video[0] = driving_video[0].cuda()
        
        kp_source = motion_estimator.estimate_kp(source_img)
        kp_driving_initial = motion_estimator.estimate_kp(driving_video[0])

        for frame_idx in tqdm(range(len(driving_video))):
            driving_frame = driving_video[frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = motion_estimator.estimate_kp(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale
                                )
            
            dense_motion = motion_estimator.estimate_motion_w_kp(kp_source=kp_source, kp_driving=kp_norm, source_image=source_img)


            out_dict = net_g(source_img, dense_motion, w=1, inference=True)
            predictions.append(tensor2img([out_dict['out'].detach().cpu()], rgb2bgr=False, min_max=(-1, 1)))
            driving_imgs.append(tensor2img([driving_video[frame_idx].detach().cpu()], rgb2bgr=False, min_max=(-1, 1)))

    return predictions, driving_imgs 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")

    parser.add_argument("--source_image", default='source.png', help="path to source image")
    parser.add_argument("--driving_video", default='driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    parser.add_argument("--visual_video", default=None, help="path to visual output")

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("--audio", dest="audio", action="store_true", help="copy audio to output from the driving video" )

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    parser.set_defaults(audio_on=False)

    opt = parser.parse_args()
    with open(opt.config, mode='r') as f:
        Loader, _ = ordered_yaml()
        config = yaml.load(f, Loader=Loader)

    source_image = cv2.imread(opt.source_image, cv2.IMREAD_COLOR)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    # prepare data
    source_image = cv2.resize(source_image, (256, 256), interpolation=cv2.INTER_LINEAR) 
    driving_video = [cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR) for frame in driving_video]

    source = img2tensor(source_image.astype(np.float32) / 255., bgr2rgb=True, float32=True)
    driving = [img2tensor(frame.astype(np.float32) / 255., bgr2rgb=False, float32=True) for frame in driving_video]

    normalize(source, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    for i in range(len(driving)):
        normalize(driving[i], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

    # load model
    net_g = build_network(config['network_g'])
    load_path = config['path'].get('pretrain_network_g', None)
    if load_path is not None:
        param_key = config['path'].get('param_key_g', 'params')
        net_g = load_network(net_g, load_path, config['path'].get('strict_load_g', True), param_key)
    net_g.eval()
    
    motion_estimator = build_network(config['network_motion_estimator'])
    load_path = config['path'].get('pretrain_network_motion_estimator', None)
    if load_path is not None:
        param_key = config['path'].get('param_key_m', 'params')
        motion_estimator = load_network(motion_estimator, load_path, config['path'].get('strict_load_motion_estimator', True), param_key)
    motion_estimator.eval()

    if not opt.cpu:
        net_g.cuda()
        motion_estimator.cuda()

    # animate
    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB), [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in driving_video], cpu=opt.cpu)
        print("Best frame: " + str(i))
        driving_forward = driving[i:]
        driving_backward = driving[:(i+1)][::-1]

        predictions_forward, driving_forward_list = make_animation(source, driving_forward, net_g, motion_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions_backward, driving_backward_list = make_animation(source, driving_backward, net_g, motion_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)

        predictions = predictions_backward[::-1] + predictions_forward[1:]
        drivings = driving_backward_list[::-1] + driving_forward_list[1:]
    else:
        predictions, drivings = make_animation(source, driving, net_g, motion_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    
    # save video
    imageio.mimwrite(opt.result_video, predictions, fps=fps)

    if opt.visual_video is not None:
        source = tensor2img([source.detach().cpu()], rgb2bgr=False, min_max=(-1, 1))
        visual = []
        for i in range(len(predictions)):
            vis = np.concatenate((source, drivings[i], predictions[i]), axis=1)
            visual.append(vis)
        imageio.mimwrite(opt.visual_video, visual, fps=fps)

    # copy audio
    if opt.audio:
        try:
            with NamedTemporaryFile(suffix=os.path.splitext(opt.result_video)[1]) as output:
                ffmpeg.output(ffmpeg.input(opt.result_video).video, ffmpeg.input(opt.driving_video).audio, output.name, c='copy').run()
                with open(opt.result_video, 'wb') as result:
                    copyfileobj(output, result)
        except ffmpeg.Error:
            print("Failed to copy audio: the driving video may have no audio track or the audio format is invalid.")
        
        if opt.visual_video is not None:
            try:
                with NamedTemporaryFile(suffix=os.path.splitext(opt.visual_video)[1]) as output:
                    ffmpeg.output(ffmpeg.input(opt.visual_video).video, ffmpeg.input(opt.driving_video).audio, output.name, c='copy').run()
                    with open(opt.visual_video, 'wb') as result:
                        copyfileobj(output, result)
            except ffmpeg.Error:
                print("Failed to copy audio: the driving video may have no audio track or the audio format is invalid.")
