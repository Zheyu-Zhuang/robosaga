import os
import random
from math import cos, pi

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import robomimic.utils.obs_utils as ObsUtils

try:
    from robomimic.models.obs_core import VisualCore
except ImportError:
    from robomimic.models.base_nets import VisualCore

from robosaga.fullgrad import FullGrad
from robosaga.tensor_extractors import EncoderOnly, FullPolicy


class SaliencyGuidedAugmentation:
    def __init__(self, model, **kwargs):
        self.model = model
        self.check_required_args(kwargs, print_args=True)

        self.mode = kwargs.get("mode", "encoder_only")  # full_policy or encoder_only
        self.m = kwargs.get("momentum", None)
        self.background_images = self.preload_all_backgrounds(kwargs["background_path"])
        self.save_debug_im_every_n_batches = kwargs.get("save_debug_im_every_n_batches", 5)
        self.update_ratio_per_batch = kwargs.get("update_ratio_per_batch", None)
        self.debug_vis = kwargs.get("debug_vis", False)
        self.debug_save = kwargs.get("debug_save", True)
        self.buffer_depth = kwargs.get("buffer_depth", None)
        self.buffer_shape = kwargs.get("buffer_shape", None)
        self.save_dir = kwargs.get("save_dir", None)
        self.disable_buffer = kwargs.get("disable_buffer", False)
        self.augmentation_ratio = kwargs.get("augmentation_ratio", None)
        self.normalizer = kwargs.get("normalizer", None)
        #
        self.epoch_idx = 0  # epoch index
        self.batch_idx = 0  # batch index
        self.extractors = self.initialise_extractors()
        self.buffer = {}
        self.is_registered = True
        if self.m is None:
            self.disable_buffer = True
            print("Warning: Momentum value not provided, disabling buffer")

    # --------------------------------------------------------------------------- #
    #                         Training Specific Functions                         #
    # --------------------------------------------------------------------------- #

    def prepare_obs_dict(self, obs_dict, validate):
        if not validate:
            self.model.train()
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

    def _update_saliency_on_batch(self, buffer_ids, obs_dict, obs_meta, validate):
        update_ratio = 0 if validate else self.update_ratio_per_batch
        assert update_ratio >= 0 and update_ratio <= 1, "update_ratio should be in [0, 1]"
        if self.mode == "full_policy":
            assert not obs_meta["has_temporal_dim"], "full policy with temporal dim not supported"
        n_samples = obs_meta["n_samples"]
        n_augmentations = int(n_samples * self.augmentation_ratio)
        n_updates = int(n_samples * update_ratio)
        assert n_updates <= n_augmentations, "update ratio should be less than augmentation ratio"

        save_on_this_batch = self.batch_idx % self.save_debug_im_every_n_batches == 0
        out = {}

        if validate and not save_on_this_batch:
            return out
        vis_ims = []
        for k in obs_meta["visual_modalities"]:
            augmentation_indices = random.sample(range(n_samples), n_augmentations)
            update_indices = augmentation_indices[:n_updates] if not validate else [0]
            updated_ids = buffer_ids[update_indices]
            crop_inds_ = obs_meta[k]["crop_inds"]
            crop_inds_ = None if crop_inds_ is None else crop_inds_[update_indices]
            buffer_shape = obs_meta[k]["raw_shape"][-2:]
            #
            net_input_dict = None
            image = obs_dict[k][update_indices]
            if self.mode == "full_policy":
                net_input_dict = {k: obs_dict[k][update_indices] for k in obs_dict.keys()}
            smaps = self.extractors[k].saliency(image, net_input_dict).detach()
            norm_smaps = self.linear_normalisation(smaps)
            if not validate:
                self.update_buffer(norm_smaps, updated_ids, k, buffer_shape, crop_inds_)
            else:
                idx = 0
                vis_ims_ = [self.denormalize_image(image[idx], k)]
                vis_smap = norm_smaps[idx]
                vis_ims.append(self.save_debug_images(vis_ims_, vis_smap))
            out[k] = {
                "augmentations": augmentation_indices,
                "updates": update_indices,
                "smaps": norm_smaps,
            }
        if save_on_this_batch:
            im_name = f"batch_{self.batch_idx}_saliency.jpg"
            im_name = os.path.join(self.save_dir, f"epoch_{self.epoch_idx}", im_name)
            self.create_saliency_dir()
            cv2.imwrite(im_name, cv2.vconcat(vis_ims))
        return out

    def denormalize_image(self, x, obs_key):
        if self.normalizer is None:
            return x
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.normalizer[obs_key].unnormalize(x)
        return x.squeeze(0) if x.shape[0] == 1 else x

    # the main function to be called for data augmentation
    def saliency_guided_augmentation_on_batch(
        self, obs_dict, buffer_ids, epoch_idx, batch_idx, validate
    ):
        self.epoch_idx, self.batch_idx = epoch_idx, batch_idx
        self.regitration_check()
        obs_dict, obs_meta = self.prepare_obs_dict(obs_dict, validate)
        self.model.eval()
        update_dict = self._update_saliency_on_batch(buffer_ids, obs_dict, obs_meta, validate)
        vis_ims = []
        if validate:
            return self.restore_obs_dict_shape(obs_dict, obs_meta)
        for i, obs_key in enumerate(obs_meta["visual_modalities"]):
            if not self.disable_buffer:
                augment_indices = update_dict[obs_key]["augmentations"]
                ids_ = buffer_ids[augment_indices]
                crop_inds_ = obs_meta[obs_key]["crop_inds"][augment_indices]
                buffer_shape = obs_meta[obs_key]["raw_shape"][-2:]
                in_shape = obs_meta[obs_key]["input_shape"][-2:]
                smaps = self.saliency_from_buffer(
                    obs_key, ids_, crop_inds_, buffer_shape, in_shape
                )
                smaps = self.linear_normalisation(smaps)
            else:
                augment_indices = update_dict[obs_key]["updates"]
                smaps = update_dict[obs_key]["smaps"]
            rand_bg_idx = random.sample(
                range(self.background_images.shape[0]), len(augment_indices)
            )
            bg = obs_meta["randomisers"][i].forward_in(self.background_images[rand_bg_idx])
            x = obs_dict[obs_key][augment_indices]
            bg = self.normalizer[obs_key].normalize(bg) if self.normalizer is not None else bg
            x_aug = x * smaps + bg * (1 - smaps)
            obs_dict[obs_key][augment_indices] = x_aug
            if self.batch_idx % 50 == 0:
                idx = 0
                vis_ims_ = [x[idx], bg[idx]]
                vis_ims_ = [self.denormalize_image(im, obs_key) for im in vis_ims_]
                vis_ims.append(self.save_debug_images(vis_ims, smaps[idx]))
        if len(vis_ims) >= 1:
            vis_ims = cv2.vconcat(vis_ims)
            cv2.imwrite("augmentation_vis", vis_ims)
        return self.restore_obs_dict_shape(obs_dict, obs_meta)

    # --------------------------------------------------------------------------- #
    #                           Saliency Core Functions                           #
    # --------------------------------------------------------------------------- #

    def update_buffer(self, s_map, buffer_ids, obs_key, buffer_shape=None, crop_inds=None):
        if self.disable_buffer:
            return
        assert obs_key in self.extractors, "obs_key not in extractors"
        assert s_map.shape[0] == buffer_ids.shape[0], "saliency and ids size mismatch"
        assert s_map.min() >= 0 and s_map.max() <= 1, "s_map not in [0, 1] range"
        if crop_inds is not None:
            assert crop_inds.shape[0] == buffer_ids.shape[0], "crop_inds and ids size mismatch"

        # ids may have repeated values, remove duplicates to avoid repeatitive saliency update
        unique_ids = self._first_occurrence_indices(buffer_ids)
        buffer_ids = buffer_ids[unique_ids]
        crop_inds = crop_inds[unique_ids] if crop_inds is not None else None

        s_map = s_map[unique_ids]
        s_map = (s_map * 255).to(torch.uint8)

        h_buffer, w_buffer = buffer_shape if buffer_shape is not None else s_map.shape[-2:]
        device = s_map.device

        # buffers are maintained in [0, 255] range as uint8 for memory efficiency
        self.check_buffer(obs_key, (h_buffer, w_buffer), device=device)

        map_from_buffer = self.buffer[obs_key][buffer_ids]

        if crop_inds is not None:
            padded_s_map = torch.zeros_like(map_from_buffer)
            for i in range(buffer_ids.shape[0]):
                h_0, w_0 = crop_inds[i, 0, 0], crop_inds[i, 0, 1]
                h_1, w_1 = h_0 + s_map.shape[-2], w_0 + s_map.shape[-1]
                padded_s_map[i, :, h_0:h_1, w_0:w_1] = s_map[i]
            s_map = padded_s_map
        updated_s_map = self.m * map_from_buffer + (1 - self.m) * s_map
        self.buffer[obs_key][buffer_ids] = updated_s_map.to(torch.uint8)

    def saliency_from_buffer(
        self,
        obs_key,
        buffer_ids,
        crop_inds=None,
        buffer_shape=None,
        out_shape=(76, 76),
    ):
        assert obs_key in self.extractors, "obs_key not in extractors"
        if crop_inds is not None:
            assert crop_inds.shape[0] == buffer_ids.shape[0], "crop_inds and ids size mismatch"

        # buffers are maintained in [0, 255] range as uint8 for memory efficiency
        h_buffer, w_buffer = buffer_shape if buffer_shape is not None else out_shape
        self.check_buffer(obs_key, (h_buffer, w_buffer))

        # retrieve saliency map from buffer, convert to [0, 1] range

        s_map = self.buffer[obs_key][buffer_ids] / 255.0

        if crop_inds is not None:
            if out_shape is None:
                return s_map
            h_out, w_out = out_shape
            s_map = ObsUtils.crop_image_from_indices(s_map, crop_inds, h_out, w_out).squeeze(1)
        else:
            t_h, t_w = out_shape
            s_map = ObsUtils.center_crop(s_map, t_h, t_w).squeeze(1)
        return s_map

    # --------------------------------------------------------------------------- #
    #                                    Utils                                    #
    # --------------------------------------------------------------------------- #
    def initialise_extractors(self):
        assert self.mode in ["full_policy", "encoder_only"]
        extractors = {}
        obs_encoder = self.get_obs_encoder()
        for k, v in obs_encoder.obs_nets.items():
            if isinstance(v, VisualCore):
                if self.mode == "full_policy":
                    extractors[k] = FullGrad(self.model, k, FullPolicy)
                else:
                    extractors[k] = FullGrad(v, k, EncoderOnly)
        return extractors

    def check_buffer(self, obs_key, buffer_shape, device="cuda"):
        h_buffer, w_buffer = buffer_shape
        if obs_key in self.buffer:
            return
        self.buffer[obs_key] = (
            torch.ones(self.buffer_depth, 1, h_buffer, w_buffer, device=device)
            .mul_(255)
            .to(torch.uint8)
        )

    def set_buffer_depth(self, buffer_depth):
        self.buffer_depth = buffer_depth

    def suppress_low_saliency(
        self, s_map, cutoff_threshold, scheduler="constant", iter=None, saturation_iter=None
    ):
        assert scheduler in ["linear", "constant", "cosine"]
        if scheduler in ["linear", "cosine"]:
            assert iter is not None, "iter must be provided for linear and cosine scheduler"

        if scheduler == "linear":
            lambda_ = iter / saturation_iter if iter < saturation_iter else 1
        elif scheduler == "cosine":
            lambda_ = 0.5 * (1 - cos(iter * pi / saturation_iter)) if iter < saturation_iter else 1
        else:
            lambda_ = 1
        cutoff_threshold = (
            torch.tensor(cutoff_threshold).to(s_map.device).view(s_map.shape[0], 1, 1, 1)
        ) * lambda_
        s_map[s_map < cutoff_threshold] = 0
        return s_map

    def linear_normalisation(self, s_map, min_delta=0.0):
        in_shape = s_map.shape
        if len(s_map.shape) == 4:
            s_map = s_map.squeeze(1)
        s_map = s_map.view(in_shape[0], -1)
        s_min = s_map.min(dim=1, keepdim=True)[0]
        s_max = s_map.max(dim=1, keepdim=True)[0]
        s_map_norm = (s_map + 1e-6 - s_min) / (s_max - s_min + 1e-6)
        if min_delta > 0:
            norm_idx = (s_max - s_min) > min_delta
            norm_idx = norm_idx.squeeze(1)
            s_map[norm_idx] = s_map_norm[norm_idx]
        else:
            s_map = s_map_norm
        s_map = torch.clip(s_map, 0, 1)
        return s_map.view(in_shape)

    @staticmethod
    def _first_occurrence_indices(buffer_ids):
        id_dict = {}
        buffer_ids = buffer_ids.tolist()
        for i, id_ in enumerate(buffer_ids):
            if id_ not in id_dict:
                id_dict[id_] = i
        return torch.tensor([id_dict[id_] for id_ in buffer_ids])

    def get_obs_encoder(self):
        if hasattr(self.model, "obs_nets"):
            return self.model
        elif hasattr(self.model, "nets"):
            return self.model.nets["encoder"].nets["obs"]
        else:
            raise ValueError("obs_encoder cannot be found in the model")

    def check_required_args(self, kwargs, print_args=False):
        required_args = [
            "momentum",
            "update_ratio_per_batch",
            "augmentation_ratio",
            "buffer_shape",
            "background_path",
            "save_dir",
        ]
        for arg in required_args:
            if arg not in kwargs:
                raise ValueError(f"Argument {arg} is required for MomentumSaliency")
        if print_args:
            print("\n==================== Momentum Saliency Arguments ====================")
            for arg in required_args:
                if isinstance(kwargs[arg], torch.Tensor):
                    print(f"{arg} shape: {list(kwargs[arg].shape)}")
                else:
                    print(f"{arg}: {kwargs[arg]}")
            print("======================================================================\n")

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
    def get_debug_image(x, idx=0, bgr_to_rgb=False):
        assert idx < x.shape[0], "idx out of bounds"
        im = x.permute(1, 2, 0).detach().cpu().numpy()
        im = np.clip(im, 0, 1) * 255
        im = im.astype(np.uint8)
        if bgr_to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def save_debug_images(self, x: list, smap: torch.Tensor, im_path: str):
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

    def unregister_hooks(self):
        for k, v in self.extractors.items():
            v.unregister_hooks()
        self.is_registered = False

    def register_hooks(self):
        for k, v in self.extractors.items():
            v.register_hooks()
        self.is_registered = True

    def regitration_check(self):
        if not self.is_registered:
            self.register_hooks()
