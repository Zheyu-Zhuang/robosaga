# Modified based on the original code from Fullgrad repository: written by Suraj Srinivas <suraj.srinivas@idiap.ch


from math import isclose

import torch
import torch.nn.functional as F


class FullGrad:
    """
    Compute FullGrad saliency map and full gradient decomposition
    """

    def __init__(self, model, obs_key, Extractor):
        self.model = model
        self.model_ext = Extractor(model, obs_key)

    def unregister_hooks(self):
        self.model_ext.unregister_hooks()

    def register_hooks(self):
        if len(self.model_ext.grad_handles) == 0:
            self.model_ext.register_hooks()

    def fullGradientDecompose(self, x, net_input=None):
        """
        Compute full-gradient decomposition for an x
        """
        self.model.eval()
        if net_input is None:
            net_input = x

        x = x.requires_grad_()

        out = self.model(net_input)

        # Get biases, Not the most efficient way when the model has no weights updates
        biases = self.model_ext.get_biases()

        output_scalar = self._raw_output_to_scalar(out)

        input_gradient, feature_gradients = self.model_ext.getFeatureGrads(x, output_scalar)

        # Compute feature-gradients \times bias
        bias_times_gradients = []

        L = len(biases)

        for i in range(L):

            # feature gradients are indexed backwards
            # because of backprop
            g = feature_gradients[L - 1 - i]

            # reshape bias dimensionality to match gradients
            b = biases[i]
            if not len(g.size()) == len(b.size()):
                bias_size = [1] * len(g.size())
                bias_size[1] = biases[i].size(0)
                b = biases[i].view(tuple(bias_size))

            bias_times_gradients.append(g * b)

        return input_gradient, bias_times_gradients

    def saliency(self, image, net_input=None):
        # FullGrad saliency
        self.model.eval()
        input_grad, bias_grad = self.fullGradientDecompose(image, net_input)

        with torch.no_grad():
            # Input-gradient * image
            grd = input_grad * image
            gradient = self._postProcess(grd).sum(1, keepdim=True)
            cam = gradient

            im_size = image.size()

            # Aggregate Bias-gradients
            for i in range(len(bias_grad)):
                # Select only Conv layers
                if len(bias_grad[i].size()) == len(im_size):
                    temp = self._postProcess(bias_grad[i])
                    gradient = F.interpolate(
                        temp, size=(im_size[2], im_size[3]), mode="bilinear", align_corners=True
                    )
                    cam += gradient.sum(1, keepdim=True)
        self.model.zero_grad()
        return cam

    def _postProcess(self, layer_grad, eps=1e-6):
        # Absolute value
        layer_grad = abs(layer_grad)

        # Rescale operations to ensure gradients lie between 0 and 1
        flatin = layer_grad.reshape((layer_grad.size(0), -1))
        temp, _ = flatin.min(1, keepdim=True)
        layer_grad = layer_grad - temp.unsqueeze(1).unsqueeze(1)

        flatin = layer_grad.reshape((layer_grad.size(0), -1))
        temp, _ = flatin.max(1, keepdim=True)
        layer_grad = layer_grad / (temp.unsqueeze(1).unsqueeze(1) + eps)
        return layer_grad

    def _raw_output_to_scalar(self, raw_output):
        return raw_output.sum()

    def checkCompleteness(self, x):
        """
        Check if completeness property is satisfied. If not, it usually means that
        some bias gradients are not computed (e.g.: implicit biases of non-linearities).

        """

        # Random input image

        # Get raw outputs
        self.model.eval()
        raw_output = self._raw_output_to_scalar(self.model(x))

        # Compute full-gradients and add them up
        input_grad, bias_grad = self.fullGradientDecompose(x)

        fullgradient_sum = (input_grad * x).sum()
        for i in range(len(bias_grad)):
            fullgradient_sum += bias_grad[i].sum()

        # Compare raw output and full gradient sum
        err_message = "\nThis is due to incorrect computation of bias-gradients."
        err_string = (
            "Completeness test failed! Raw output = "
            + str(raw_output.max().item())
            + " Full-gradient sum = "
            + str(fullgradient_sum.item())
        )
        assert isclose(raw_output.item(), fullgradient_sum.item(), rel_tol=1e-4), (
            err_string + err_message
        )
        print("Completeness test passed for FullGrad.")
