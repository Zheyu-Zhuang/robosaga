import os
import random
from math import cos, pi

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from robomimic.models.obs_core import VisualCore
except ImportError:
    from robomimic.models.base_nets import VisualCore

from torch import nn


class SODA:
    def __init__(self, model, ema_model, blend_factor=0.5, **kwargs):
        self.model = model
        self.ema_model = ema_model
        self.porj = nn.Linear(128, 128)
        self.background_images = self.preload_all_backgrounds(kwargs["background_path"])
        # augmentation index fixed across obs pairs
        self.blend_factor = blend_factor
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.epoch_idx = 0  # epoch index
        self.batch_idx = 0  # batch index

    # --------------------------------------------------------------------------- #
    #                         Training Specific Functions                         #
    # --------------------------------------------------------------------------- #

    # the main function to be called for data augmentation
    def step_train_epoch(self, obs_dict, epoch_idx, batch_idx, validate=False):
        self.epoch_idx, self.batch_idx = epoch_idx, batch_idx
        obs_dict, obs_meta = self.prepare_obs_dict(obs_dict)
        vec_ema = []
        vec = []
        for i, obs_key in enumerate(obs_meta["visual_modalities"]):
            im = obs_dict[obs_key]
            rand_bg_idx = random.sample(range(self.background_images.shape[0]), len(im))
            bg = self.background_images[rand_bg_idx]
            aug_im = im * self.blend_factor + bg * (1 - self.blend_factor)
            vec_ema.append(self.ema_model.obs_nets[obs_key](obs_dict[obs_key]))
            vec.append(self.model.obs_nets[obs_key](aug_im))
        vec_ema = torch.cat(vec_ema, dim=1)
        vec = torch.cat(vec, dim=1)
        vec_ema = torch.nn.functional.normalize(vec_ema, p=2, dim=1).detach()
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        vec_ema = self.porj(vec_ema)
        vec = self.porj(vec)
        loss = self.loss(vec, vec_ema)
        if not validate:
            loss.backward()
            self.optimizer.step()
        with torch.no_grad():
            for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
                m = 0.995
                ema_p.data.mul_(m).add_((1 - m) * p.data)
        return loss.item()

    def prepare_obs_dict(self, obs_dict):
        obs_encoder = self.get_obs_encoder()
        # get visual modalities and randomisers
        visual_modalities = [
            k for k, v in obs_encoder.obs_nets.items() if isinstance(v, VisualCore)
        ]
        randomisers = [obs_encoder.obs_randomizers[k] for k in visual_modalities]

        vis_obs_dim = obs_dict[visual_modalities[0]].shape
        has_temporal_dim = len(vis_obs_dim) > 4
        n_samples = vis_obs_dim[0] if not has_temporal_dim else vis_obs_dim[0] * vis_obs_dim[1]

        obs_meta = {
            "temporal_dim": vis_obs_dim[1] if has_temporal_dim else 0,
            "visual_modalities": visual_modalities,
            "n_samples": n_samples,
            "randomisers": randomisers,
        }

        def flatten_temporal_dim(x):
            raw_dim = x.shape
            return (x.view(n_samples, *raw_dim[2:]), raw_dim) if has_temporal_dim else (x, raw_dim)

        for k in obs_encoder.obs_shapes.keys():
            obs_dict[k], raw_shape = flatten_temporal_dim(obs_dict[k])
            if not self.disable_buffer:
                self.check_buffer(k, raw_shape[-2:], device=obs_dict[k].device)
            obs_meta[k] = {
                "raw_shape": raw_shape,
                "is_visual": k in visual_modalities,
                "input_shape": list(raw_shape),
            }

        for vis_obs, randomiser in zip(visual_modalities, randomisers):
            x, crop_inds = randomiser.forward_in(obs_dict[vis_obs], return_inds=True)
            obs_meta[vis_obs]["input_shape"][-3:] = x.shape[-3:]
            obs_meta[vis_obs]["crop_inds"] = crop_inds
            obs_dict[vis_obs] = x

        return obs_dict, obs_meta

    def save_debug_images(self, obs_dict, obs_meta, sample_idx=0):
        if not self.debug_save:
            return
        save_on_this_batch = self.batch_idx % self.save_debug_im_every_n_batches == 0
        if not save_on_this_batch:
            return
        vis_ims = []
        for obs_key in obs_meta["visual_modalities"]:
            image = obs_dict[obs_key][sample_idx].unsqueeze(0)
            net_input_dict = None
            if self.mode == "full_policy":
                net_input_dict = {k: obs_dict[k][sample_idx].unsqueeze(0) for k in obs_dict.keys()}
            smaps = self.extractors[obs_key].saliency(image, net_input_dict).detach()
            vis_smap = self.linear_normalisation(smaps)[0]
            vis_ims_ = [self.denormalize_image(image, obs_key)]
            vis_ims.append(self.compose_saga_images(vis_ims_, vis_smap))
        im_name = f"batch_{self.batch_idx}_saliency.jpg"
        im_name = os.path.join(self.save_dir, f"epoch_{self.epoch_idx}", im_name)
        self.create_saliency_dir()
        cv2.imwrite(im_name, self.vstack_images(vis_ims))

    def denormalize_image(self, x, obs_key):
        if self.normalizer is None:
            return x.squeeze(0) if x.shape[0] == 1 else x
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.normalizer[obs_key].unnormalize(x)
        return x.squeeze(0) if x.shape[0] == 1 else x

    @staticmethod
    def vstack_images(images, padding=10):
        images = [
            cv2.copyMakeBorder(
                im, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            for im in images
        ]
        return cv2.vconcat(images)

    def get_obs_encoder(self):
        if hasattr(self.model, "obs_nets"):
            return self.model
        elif hasattr(self.model, "nets"):
            return self.model.nets["encoder"].nets["obs"]
        else:
            raise ValueError("obs_encoder cannot be found in the model")

    def create_saliency_dir(self):
        saliency_dir = os.path.join(self.save_dir, f"epoch_{self.epoch_idx}")
        if not os.path.exists(saliency_dir):
            os.makedirs(saliency_dir)

    @staticmethod
    def restore_obs_dict_shape(obs_dict, obs_meta):
        for k in obs_dict.keys():
            obs_dict[k] = obs_dict[k].view(obs_meta[k]["input_shape"])
        return obs_dict

    @staticmethod
    def get_debug_image(x, bgr_to_rgb=False):
        im = x.permute(1, 2, 0).detach().cpu().numpy()
        im = np.clip(im, 0, 1) * 255
        im = im.astype(np.uint8)
        if bgr_to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def compose_saga_images(self, x: list, smap: torch.Tensor, im_path: str = None):
        assert isinstance(x, list), "x should be a list of torch tensors"
        x = [self.get_debug_image(x_, bgr_to_rgb=True) for x_ in x]
        smap = self.get_debug_image(smap)
        x.append(cv2.applyColorMap(smap, cv2.COLORMAP_JET))
        im_pad = np.ones((x[0].shape[0], 5, 3), dtype=np.uint8)
        for i in range(len(x) - 1):
            x.insert(2 * i + 1, im_pad)
        vis = cv2.hconcat(x)
        if im_path is not None:
            cv2.imwrite(im_path, vis)
        if self.debug_vis:
            cv2.imshow("saliency", vis)
            cv2.waitKey(0)
        return vis

    @staticmethod
    def preload_all_backgrounds(background_path, im_size=(84, 84)):
        all_f_names = os.listdir(background_path)
        all_im_names = []
        for f_name in all_f_names:
            if f_name.endswith(".jpg") or f_name.endswith(".png"):
                all_im_names.append(f_name)
        n_images = len(all_im_names)
        backgrounds = torch.zeros((n_images, 3, im_size[0], im_size[1]))
        for i, im_name in enumerate(all_im_names):
            im = Image.open(os.path.join(background_path, im_name))
            if im.size != im_size:
                im = im.resize(im_size)
            im = transforms.ToTensor()(im)
            backgrounds[i] = im
        return backgrounds.to("cuda")
