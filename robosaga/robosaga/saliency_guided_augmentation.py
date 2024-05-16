import os
import random

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
from robosaga.tensor_extractors import EncoderOnly


class SaliencyGuidedAugmentation:
    def __init__(self, model, **kwargs):
        # Model and extractors
        self.model = model
        self.extractors = self.initialise_extractors()

        # Augmentation related attributes
        self.aug_ratio = self.get_kwarg(kwargs, "aug_ratio", None)
        self.aug_strategy = self.get_kwarg(kwargs, "aug_strategy", "saga_mixup")
        self.aug_obs_pairs = self.get_kwarg(kwargs, "aug_obs_pairs", False)
        self.update_ratio = self.get_kwarg(kwargs, "update_ratio", None)

        # Buffer related attributes
        self.buffer = {}
        self.buffer_watcher = {}  # to keep track of the progress of the updates
        self.buffer_depth = self.get_kwarg(kwargs, "buffer_depth", None)
        self.buffer_shape = self.get_kwarg(kwargs, "buffer_shape", None)
        self.disable_buffer = self.get_kwarg(kwargs, "disable_buffer", False)
        if self.disable_buffer:
            print("Warning: Saliency Buffer is Disabled, aug_ratio will be set to update_ratio")

        # Other attributes
        self.backgrounds = self.preload_all_backgrounds(kwargs["background_path"])
        self.save_debug_im_every_n_batches = self.get_kwarg(
            kwargs, "save_debug_im_every_n_batches", 5
        )
        self.debug_vis = self.get_kwarg(kwargs, "debug_vis", False)
        self.debug_save = self.get_kwarg(kwargs, "debug_save", True)
        self.save_dir = self.get_kwarg(kwargs, "save_dir", None)
        self.normalizer = self.get_kwarg(kwargs, "normalizer", None)
        self.disable_during_training = self.get_kwarg(kwargs, "disable_during_training", False)

        # Indexes
        self.epoch_idx = 0  # epoch index
        self.batch_idx = 0  # batch index

        # Registration status
        self.is_registered = True
        self.is_training = True

        self.check_augmentation_strategy(kwargs)
        self.check_required_args(print_args=True)

    def get_kwarg(self, kwargs, key, default):
        return kwargs.get(key, default)

    def check_augmentation_strategy(self, kwargs):
        assert self.aug_strategy in [
            "saga_mixup",
            "saga_erase",
            "simple_overlay",
        ], "Invalid aug_strategy"
        assert self.aug_ratio is not None, "aug_ratio is required"
        if self.aug_strategy == "saga_erase":
            assert "erase_thresh" in kwargs, "erase_thresh is required for saga_erase strategy"
            assert 0 < kwargs["erase_thresh"] <= 1, "erase_thresh should be in (0, 1]"
            self.erase_thresh = kwargs["erase_thresh"]
        if self.aug_strategy == "simple_overlay":
            if not self.disable_buffer:
                self.disable_buffer = True
                print("SaGA Warning: Buffer is disabled for simple_overlay strategy")
            # TODO: add blending ratio for simple_overlay

    # --------------------------------------------------------------------------- #
    #                         Training Specific Functions                         #
    # --------------------------------------------------------------------------- #

    # the main function to be called for data augmentation
    def __call__(self, obs_dict, buffer_ids, epoch_idx, batch_idx):
        self.is_training = self.model.training
        self.epoch_idx, self.batch_idx = epoch_idx, batch_idx
        is_turned_off = self.is_training and self.disable_during_training
        if is_turned_off:
            self.unregister_hooks()
            return obs_dict
        obs_dict, obs_meta = self.prepare_obs_dict(obs_dict)
        self.model.eval()  # required for saliency computation
        if self.is_training and not self.disable_during_training:
            if self.aug_strategy == "simple_overlay":
                self.unregister_hooks()
                obs_dict = self.simple_overlay(obs_dict, obs_meta)
            else:
                self.register_hooks()
                update_dict = self.update_saliency_buffer(buffer_ids, obs_dict, obs_meta)
                obs_dict = self.saliency_guided_augmentation(
                    obs_dict, buffer_ids, obs_meta, update_dict
                )
        elif not self.is_training:
            self.register_hooks()
            self.save_debug_images(obs_dict, obs_meta)
        self.model.train() if self.is_training else self.model.eval()
        # n_updated = torch.sum(self.buffer_watcher[obs_meta["visual_modalities"][0]] > 0).item()
        # print(f"Updated {n_updated/self.buffer_depth*100:.2f}% of buffer")
        return self.restore_obs_dict_shape(obs_dict, obs_meta)

    def saliency_guided_augmentation(self, obs_dict, buffer_ids, obs_meta, update_dict):
        if update_dict == {} or not self.is_training or self.disable_during_training:
            return obs_dict
        vis_ims = []
        for i, obs_key in enumerate(obs_meta["visual_modalities"]):
            if self.disable_buffer:
                aug_inds = update_dict[obs_key]["updates"]
                smaps = update_dict[obs_key]["smaps"]
            if not self.disable_buffer:
                aug_inds = update_dict[obs_key]["augmentations"]
                global_ids = buffer_ids[aug_inds]
                crop_inds = obs_meta[obs_key]["crop_inds"][aug_inds]
                out_shape = obs_meta[obs_key]["input_shape"][-2:]
                smaps = self.fetch_saliency_from_buffer(obs_key, global_ids, crop_inds, out_shape)
                smaps = self.linear_normalisation(smaps)
            rand_bg_idx = random.sample(range(self.backgrounds.shape[0]), len(aug_inds))
            bg = obs_meta["randomisers"][i].forward_in(self.backgrounds[rand_bg_idx])
            bg = self.normalizer[obs_key].normalize(bg) if self.normalizer is not None else bg
            if self.aug_strategy == "saga_mixup":
                # blending_factor = torch.rand(smaps.shape[0], 1, 1, 1).to(smaps.device) * 0.5 + 0.5
                # smaps = torch.clip(smaps, 0, blending_factor)
                # smaps[smaps < 0.3] = 0
                # smaps = torch.clip(smaps, 0, 0.8)
                x_aug = obs_dict[obs_key][aug_inds] * smaps + bg * (1 - smaps)
            elif self.aug_strategy == "saga_erase":
                smaps[smaps < self.erase_thresh] = 0
                smaps[smaps >= self.erase_thresh] = 1
                x_aug = obs_dict[obs_key][aug_inds] * smaps + bg * (1 - smaps)
            if self.batch_idx % 50 == 0:
                idx = 0
                x_vis, x_aug_vis = obs_dict[obs_key][idx], obs_dict[obs_key][idx]
                vis_smap, bg_vis = torch.ones_like(smaps[idx]), torch.zeros_like(bg[idx])
                if idx in aug_inds:
                    idx_ = aug_inds.tolist().index(idx)
                    vis_smap, x_aug_vis, bg_vis = smaps[idx_], x_aug[idx_], bg[idx_]
                vis_ims_ = [x_vis, x_aug_vis, bg_vis]
                vis_ims_ = [self.denormalize_image(im, obs_key) for im in vis_ims_]
                vis_ims.append(self.compose_saga_images(vis_ims_, vis_smap))
            obs_dict[obs_key][aug_inds] = x_aug
        if len(vis_ims) >= 1:
            cv2.imwrite("augmentation_vis.jpg", self.vstack_images(vis_ims))
        return obs_dict

    def sample_update_indices(self, n_samples, buffer_ids, obs_key, mode="random"):
        n_augs = int(n_samples * self.aug_ratio)
        n_updates = int(n_samples * self.update_ratio)
        n_updates = n_augs if n_updates > n_augs else n_updates
        if n_updates == 0:
            return torch.tensor([]), torch.tensor([])
        if mode == "random":
            aug_batch_inds = torch.randperm(n_samples)[:n_augs]
            update_batch_inds = aug_batch_inds[:n_updates]
        elif mode == "frequency":
            batch_inds = torch.randperm(n_samples)
            buffer_ids = buffer_ids[batch_inds]
            #
            update_freq = self.buffer_watcher[obs_key][buffer_ids]
            _, sorted_inds = torch.sort(update_freq)
            aug_batch_inds = batch_inds[sorted_inds[:n_augs]]
            update_batch_inds = aug_batch_inds[:n_updates]
        return update_batch_inds, aug_batch_inds

    def update_saliency_buffer(self, buffer_ids, obs_dict, obs_meta):
        if self.disable_during_training or not self.is_training:
            return {}
        self.model.eval()
        n_samples = obs_meta["n_samples"]
        # get update frequency
        shared_update_inds = None
        shared_aug_inds = None
        out = {}
        for k in obs_meta["visual_modalities"]:
            if shared_update_inds is None or not self.aug_obs_pairs:
                update_inds, aug_inds = self.sample_update_indices(n_samples, buffer_ids, k)
                if shared_update_inds is None:
                    shared_update_inds, shared_aug_inds = update_inds, aug_inds
            else:
                update_inds, aug_inds = shared_update_inds, shared_aug_inds  #
            if len(update_inds) == 0:
                self.unregister_hooks()
                continue
            image_for_update = obs_dict[k][update_inds]  # batch indices
            smaps = self.extractors[k].saliency(image_for_update).detach()
            norm_smaps = self.linear_normalisation(smaps)
            if not self.disable_buffer:
                crop_inds_ = obs_meta[k]["crop_inds"]
                crop_inds_ = None if crop_inds_ is None else crop_inds_[update_inds]
                self.update_buffer(norm_smaps, buffer_ids[update_inds], k, crop_inds_)
                self.buffer_watcher[k][buffer_ids[update_inds]] += 1
            out[k] = {
                "augmentations": aug_inds,
                "updates": update_inds,
                "smaps": norm_smaps,
            }
        return out

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

    def simple_overlay(self, obs_dict, obs_meta):
        if not self.is_training or self.disable_during_training:
            return obs_dict
        for i, obs_key in enumerate(obs_meta["visual_modalities"]):
            n_samples = obs_meta["n_samples"]
            aug_inds = torch.randperm(n_samples)
            aug_inds = aug_inds[: int(n_samples * self.aug_ratio)]
            rand_bg_idx = random.sample(range(self.backgrounds.shape[0]), len(aug_inds))
            bg = obs_meta["randomisers"][i].forward_in(self.backgrounds[rand_bg_idx])
            bg = self.normalizer[obs_key].normalize(bg) if self.normalizer is not None else bg
            blend_factor = 0.5
            # blend_factor = torch.rand(aug_inds.shape[0], 1, 1, 1).to(bg.device) * 0.5 + 0.5
            x_aug = obs_dict[obs_key][aug_inds] * blend_factor + bg * (1 - blend_factor)
            obs_dict[obs_key][aug_inds] = x_aug
        return obs_dict

    # --------------------------------------------------------------------------- #
    #                           Saliency Core Functions                           #
    # --------------------------------------------------------------------------- #

    def update_buffer(self, s_map, buffer_ids, obs_key, crop_inds=None):
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

        map_from_buffer = self.buffer[obs_key][buffer_ids]

        if crop_inds is not None:
            padded_s_map = torch.zeros_like(map_from_buffer)
            for i in range(buffer_ids.shape[0]):
                h_0, w_0 = crop_inds[i, 0, 0], crop_inds[i, 0, 1]
                h_1, w_1 = h_0 + s_map.shape[-2], w_0 + s_map.shape[-1]
                padded_s_map[i, :, h_0:h_1, w_0:w_1] = s_map[i]
            s_map = padded_s_map
        self.buffer[obs_key][buffer_ids] = s_map.to(torch.uint8)

    def fetch_saliency_from_buffer(
        self,
        obs_key,
        buffer_ids,
        crop_inds=None,
        out_shape=(76, 76),
    ):
        assert obs_key in self.extractors, "obs_key not in extractors"
        if crop_inds is not None:
            assert crop_inds.shape[0] == buffer_ids.shape[0], "crop_inds and ids size mismatch"

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

    def save_debug_images(self, obs_dict, obs_meta, sample_idx=0):
        if not self.debug_save:
            return
        save_on_this_batch = self.batch_idx % self.save_debug_im_every_n_batches == 0
        if not save_on_this_batch:
            return
        vis_ims = []
        for obs_key in obs_meta["visual_modalities"]:
            image = obs_dict[obs_key][sample_idx].unsqueeze(0)
            smaps = self.extractors[obs_key].saliency(image).detach()
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

    def initialise_extractors(self):
        extractors = {}
        obs_encoder = self.get_obs_encoder()
        for k, v in obs_encoder.obs_nets.items():
            if isinstance(v, VisualCore):
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
        self.buffer_watcher[obs_key] = torch.zeros(self.buffer_depth).to(device)

    def set_buffer_depth(self, buffer_depth):
        self.buffer_depth = buffer_depth

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

    def check_required_args(self, print_args=False):
        required_args = [
            "aug_strategy",
            "aug_ratio",
            "aug_obs_pairs",
            "update_ratio",
            "disable_buffer",
            "buffer_shape",
            "backgrounds",
            "disable_during_training",
            "save_dir",
        ]

        for arg in required_args:
            if arg not in self.__dict__:
                raise ValueError(f"Argument {arg} is required for MomentumSaliency")
        if print_args:
            print("\n==================== Momentum Saliency Arguments ====================")
            for arg in required_args:
                if isinstance(self.__dict__[arg], torch.Tensor):
                    print(f"{arg} shape: {list(self.__dict__[arg].shape)}")
                else:
                    print(f"{arg}: {self.__dict__[arg]}")
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

    def unregister_hooks(self):
        if not self.is_registered:
            return
        for k, v in self.extractors.items():
            v.unregister_hooks()
        self.is_registered = False

    def register_hooks(self):
        if self.is_registered:
            return
        for k, v in self.extractors.items():
            v.register_hooks()
        self.is_registered = True
