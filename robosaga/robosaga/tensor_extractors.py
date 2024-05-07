# Modified based on the original code from Fullgrad repository: written by Suraj Srinivas <suraj.srinivas@idiap.ch>

import torch
import torch.nn as nn

from robosaga.custom_group_norm import CustomGroupNorm


class FullGradExtractorBase:
    def __init__(self, model):

        self.model = model
        self.feature_grads = []
        self.grad_handles = []
        self.target_modules = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, CustomGroupNorm]

    def unregister_hooks(self):
        for handle in self.grad_handles:
            handle.remove()
        self.grad_handles = []

    def _extract_layer_grads(self, m, in_grad, out_grad):
        if m.bias is not None:
            self.feature_grads.append(out_grad[0])

    def _register_module_hooks(self, m):
        if not any([isinstance(m, M) for M in self.target_modules]):
            return
        handle_g = m.register_full_backward_hook(self._extract_layer_grads)
        self.grad_handles.append(handle_g)

    def _extract_layer_bias(self, m):
        if not any([isinstance(m, M) for M in self.target_modules]):
            return None
        # extract bias of each layer
        # for batchnorm, the overall "bias" is different
        # from batchnorm bias parameter.
        # Let m -> running mean, s -> running std
        # Let w -> BN weights, b -> BN bias
        # Then, ((x - m)/s)*w + b = x*w/s + (- m*w/s + b)
        # Thus (-m*w/s + b) is the effective bias of batchnorm
        if isinstance(m, nn.BatchNorm2d):
            b = -(m.running_mean * m.weight / torch.sqrt(m.running_var + m.eps)) + m.bias
            return b.data
        elif isinstance(m, CustomGroupNorm):
            return m.fullgrad_bias.data
        elif m.bias is None:
            return None
        else:
            return m.bias.data

    def getFeatureGrads(self, x, output_scalar):

        # Empty feature grads list
        self.feature_grads = []

        self.model.zero_grad()
        # Gradients w.r.t. input
        input_gradients = torch.autograd.grad(outputs=output_scalar, inputs=x)[0]
        return input_gradients, self.feature_grads

    def register_hooks(self):
        raise NotImplementedError

    def get_biases(self):
        raise NotImplementedError


class EncoderOnly(FullGradExtractorBase):
    # unused obs_key for api compatibility
    def __init__(self, model, obs_key):
        super().__init__(model)
        self.register_hooks()

    def register_hooks(self):
        for m in self.model.modules():
            self._register_module_hooks(m)

    def get_biases(self):
        biases = []
        for m in self.model.modules():
            b = self._extract_layer_bias(m)
            if b is not None:
                biases.append(b)
        return biases


class FullPolicy(FullGradExtractorBase):
    def __init__(self, model, obs_key):

        assert obs_key in model.nets["encoder"].nets["obs"].obs_nets.keys()
        assert "mlp" in model.nets.keys()
        assert "decoder" in model.nets.keys()

        super().__init__(model)

        self.encoder = self.model.nets["encoder"].nets["obs"].obs_nets[obs_key]
        self.mlp = self.model.nets["mlp"]
        self.decoder = self.model.nets["decoder"]

        self.register_hooks()

    def register_hooks(self):
        for sub_net in [self.encoder, self.mlp, self.decoder]:
            for m in sub_net.modules():
                self._register_module_hooks(m)

    def get_biases(self):
        biases = []
        for sub_net in [self.encoder, self.mlp, self.decoder]:
            for m in sub_net.modules():
                b = self._extract_layer_bias(m)
                if b is not None:
                    biases.append(b)
        return biases
