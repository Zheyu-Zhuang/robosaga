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

        self.mode = kwargs.get("mode", "encoder_only")  # full_policy or encoder_only
        self.momentum = kwargs.get("momentum", None)
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
        self.disable_during_training = kwargs.get("disable_during_training", False)
        self.disable_first_n_epochs = kwargs.get("disable_first_n_epochs", 0)
        self.augment_scheduler = kwargs.get("augment_scheduler", None)
        # augmentation index fixed across obs pairs
        self.augment_obs_pairs = kwargs.get("augment_obs_pairs", False)
        self.epoch_idx = 0  # epoch index
        self.batch_idx = 0  # batch index
        self.extractors = self.initialise_extractors()
        self.buffer = {}
        self.buffer_watcher = {}  # to keep track of the progress of the updates
        self.is_registered = True
        self.is_training = True
        #
        if self.disable_buffer:
            print("Warning: Saliency Buffer is Disabled")
        self.check_required_args(print_args=True)

    # --------------------------------------------------------------------------- #
    #                         Training Specific Functions                         #
    # --------------------------------------------------------------------------- #

    # the main function to be called for data augmentation
    def __call__(self, obs_dict, buffer_ids, epoch_idx, batch_idx):
        self.is_training = self.model.training
        self.epoch_idx, self.batch_idx = epoch_idx, batch_idx
        is_turned_off = self.is_training and self.disable_during_training
        if is_turned_off or epoch_idx < self.disable_first_n_epochs:
            self.unregister_hooks()
            return obs_dict
        self.register_hooks()
        obs_dict, obs_meta = self.prepare_obs_dict(obs_dict)
        self.model.eval()  # required for saliency computation
        if self.is_training and not self.disable_during_training:
            self.step_augmentation_scheduler()
            update_dict = self.update_saliency_buffer(buffer_ids, obs_dict, obs_meta)
            obs_dict = self.saliency_guided_augmentation(
                obs_dict, buffer_ids, obs_meta, update_dict
            )
        elif not self.is_training:
            self.save_debug_images(obs_dict, obs_meta)
        self.model.train() if self.is_training else self.model.eval()
        # n_updated = torch.sum(self.buffer_watcher[obs_meta["visual_modalities"][0]] > 0).item()
        # print(f"Updated {n_updated/self.buffer_depth*100:.2f}% of buffer")
        return self.restore_obs_dict_shape(obs_dict, obs_meta)

    def step_augmentation_scheduler(self):
        if self.augment_scheduler is None or self.augment_scheduler["mode"] == "constant":
            return
        self.past_augmentation_ratio = self.augmentation_ratio
        mode = self.augment_scheduler["mode"]
        end_epoch = self.augment_scheduler["end_epoch"]
        assert mode in ["linear", "cosine"], "Invalid scheduler mode"

        if mode == "linear":
            lambda_ = min(self.epoch_idx / end_epoch, 1)
        elif mode == "cosine":
            lambda_ = (
                0.5 * (1 - np.cos(self.epoch_idx * np.pi / end_epoch))
                if self.epoch_idx < end_epoch
                else 1
            )
        self.augmentation_ratio = lambda_ * self.augment_scheduler["end_ratio"]
        if self.augmentation_ratio != self.past_augmentation_ratio:
            print(f"Augmentation Ratio: {self.augmentation_ratio:.2f}")

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
            rand_bg_idx = random.sample(range(self.background_images.shape[0]), len(aug_inds))
            bg = obs_meta["randomisers"][i].forward_in(self.background_images[rand_bg_idx])
            x = obs_dict[obs_key][aug_inds]
            bg = self.normalizer[obs_key].normalize(bg) if self.normalizer is not None else bg
            x_aug = x * smaps + bg * (1 - smaps)
            obs_dict[obs_key][aug_inds] = x_aug
            if self.batch_idx % 50 == 0:
                idx = 0
                x_vis, x_aug_vis = x[idx], x[idx]
                vis_smap, bg_vis = torch.ones_like(smaps[idx]), torch.zeros_like(bg[idx])
                if idx in aug_inds:
                    vis_smap, x_aug_vis, bg_vis = smaps[idx], x_aug[idx], bg[idx]
                vis_ims_ = [x_vis, x_aug_vis, bg_vis]
                vis_ims_ = [self.denormalize_image(im, obs_key) for im in vis_ims_]
                vis_ims.append(self.compose_saga_images(vis_ims_, vis_smap))
        if len(vis_ims) >= 1:
            cv2.imwrite("augmentation_vis.jpg", self.vstack_images(vis_ims))
        return obs_dict

    def frequency_based_sampling(self, n_samples, buffer_ids, obs_key):
        n_augs = int(n_samples * self.augmentation_ratio)
        n_updates = int(n_samples * self.update_ratio_per_batch)
        n_updates = n_augs if n_updates > n_augs else n_updates  # ensure n_updates <= n_augs
        if n_updates == 0:
            return torch.tensor([]), torch.tensor([])
        n_retrivals = n_augs - n_updates
        # randomly permute the buffer ids
        batch_inds = torch.randperm(buffer_ids.shape[0])
        buffer_ids = buffer_ids[batch_inds]
        update_freq = self.buffer_watcher[obs_key][buffer_ids]
        _, sorted_inds = torch.sort(update_freq)
        # augmentation batch indices should be a mix of updates and buffer retrivals
        # by including the least the most recent updates and the latest updates from the buffer
        update_batch_inds = batch_inds[sorted_inds[:n_updates]]
        aug_batch_inds = update_batch_inds
        if n_retrivals > 0:
            aug_batch_inds = torch.cat((aug_batch_inds, batch_inds[sorted_inds[-n_retrivals:]]))
        return update_batch_inds, aug_batch_inds

    def update_saliency_buffer(self, buffer_ids, obs_dict, obs_meta):
        if self.disable_during_training or not self.is_training:
            return {}
        if self.mode == "full_policy":
            assert not obs_meta["has_temporal_dim"], "full policy with temporal dim not supported"
        self.model.eval()
        n_samples = obs_meta["n_samples"]
        # get update frequency
        shared_update_inds = None
        shared_aug_inds = None
        out = {}
        for k in obs_meta["visual_modalities"]:
            if shared_update_inds is None or not self.augment_obs_pairs:
                update_inds, aug_inds = self.frequency_based_sampling(n_samples, buffer_ids, k)
                if shared_update_inds is None:
                    shared_update_inds, shared_aug_inds = update_inds, aug_inds
            else:
                update_inds, aug_inds = shared_update_inds, shared_aug_inds  #
            if len(update_inds) == 0:
                self.unregister_hooks()
                continue
            net_input_dict = None
            image_for_update = obs_dict[k][update_inds]  # batch indices
            if self.mode == "full_policy":
                net_input_dict = {k: obs_dict[k][update_inds] for k in obs_dict.keys()}
            smaps = self.extractors[k].saliency(image_for_update, net_input_dict).detach()
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
        updated_s_map = s_map
        if self.momentum > 0:
            updated_s_map = self.m * map_from_buffer + (1 - self.m) * s_map
        self.buffer[obs_key][buffer_ids] = updated_s_map.to(torch.uint8)

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
            "momentum",
            "update_ratio_per_batch",
            "buffer_shape",
            "background_images",
            "save_dir",
            "disable_during_training",
            "disable_first_n_epochs",
            "disable_buffer",
            "augment_obs_pairs",
            "augmentation_ratio",
            "augment_scheduler",
        ]
        if self.augment_scheduler is not None and "end_ratio" in self.augment_scheduler:
            print(
                "[SaGA Warning] Property 'augmentation_ratio' is overwriten by the augment_scheduler"
            )
        if self.augment_scheduler is None and self.augmentation_ratio is None:
            raise ValueError("augmentation_ratio is required if augment_scheduler is None")

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
