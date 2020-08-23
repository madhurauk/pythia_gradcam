import sys
sys.path.append('purva-vqa-maskrcnn-benchmark')

print("in pythia_gradcam : vqa-maskrcnn-benchmark appended to sys")
#print(sys.path)

from maskrcnn_benchmark.structures.bounding_box import BoxList
from collections import OrderedDict, Sequence
from torch.nn import functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
import copy


class _BaseWrapper(object):
    """Please modify forward() and backward() according to your task."""

    def __init__(self, model):
        print("in pythia_gradcam : init")
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        """Simple classification."""
        self.model.zero_grad()
        self.logits = self.model.forward(image)
        print("in pgc: ",self.model(image).size())
        #self.logits = self.model(image)["scores"]
        #return self.logits
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """Class-specific backpropagation.

        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """
        # print("backward")
        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """Remove all the forward/backward hook functions."""
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        image["resnet_img"].requires_grad_(True)
        image["detectron_img"].requires_grad_(True)
        image["detectron_scale"].requires_grad_(True)
        self.image = image    # .requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """Striving for Simplicity: the All Convolutional Net.

    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))
            # pass


class Deconvnet(BackPropagation):
    """Striving for Simplicity: the All Convolutional Net.

    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))

def detach_output(output, depth):
    if not isinstance(output, torch.Tensor) and not isinstance(output, dict) and not isinstance(output, BoxList):
        tuple_list = []
        for item in output:
            if isinstance(output, torch.Tensor):
                # print("output: ", output.shape)
                tuple_list.append(item.detach())
            else:
                tuple_list.append(detach_output(item, depth+1))
        return tuple_list
    elif isinstance(output, dict) or isinstance(output, BoxList):
        # print("Dict keys: ", output.keys())
        return output
    # print("output: ", output.shape)
    return output.detach()


class GradCAM(_BaseWrapper):
    """Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # print("forward_hook_:")
                # Save featuremaps
                print("key forward: ", key)
                # output.register_hook(backward_hook(key))
                output = detach_output(output, 0)
                if not isinstance(output, dict) or not isinstance(output, BoxList):
                    self.fmap_pool[key] = output

            return forward_hook_

        def backward_hook(key):
            # print("key2: ", key)
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                print("key backward: ", key)
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # print(model.resnet152_model._modules.get("0"))
        # model.resnet152_model.layer4[0].conv2.register_backward_hook(backward_hook)

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                # print("name: ", name)
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                # print("name: ", name)
                # module.retain_grad()
                # module.require_grad = True
                # print(name)
                self.handlers.append(module.register_backward_hook(backward_hook(name)))
                # print("self.handlers: ", self.handlers)
        # self.handlers.append(model.resnet152_model._modules.get("0").register_backward_hook(backward_hook('resnet152_model.0')))
        # my_layer = model.resnet152_model._modules.get("7")
        # my_layer_name = 'detection_model.backbone.fpn.fpn_layer4'
        # self.handlers.append(my_layer.register_backward_hook(backward_hook(my_layer_name)))

    def _find(self, pool, target_layer):
        # print(pool.keys())
        # print(list(self.model.named_modules()))
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, sample_list, image_shape):
        self.image_shape = image_shape
        return super(GradCAM, self).forward(sample_list)

    # def select_highest_layer(self):
    #     fmap_list, weight_list = [], []
    #     module_names = []
    #     for name, _ in self.model.named_modules():
    #         module_names.append(name)
    #     module_names.reverse()
    #
    #     for i in range(self.logits.shape[0]):
    #         counter = 0
    #         for layer in module_names:
    #             try:
    #                 print("Testing layer: {}".format(layer))
    #                 fmaps = self._find(self.fmap_pool, layer)
    #                 print("1")
    #                 np.shape(fmaps) # Throws error without this line, I have no idea why...
    #                 print("2")
    #                 fmaps = fmaps[i]
    #                 print("3")
    #                 grads = self._find(self.grad_pool, layer)[i]
    #                 print("4")
    #                 import array_check
    #                 array_check.check(fmaps)
    #                 array_check.check(grads)
    #                 # print("counter: {}".format(counter))
    #                 # print("fmaps shape: {}".format(np.shape(fmaps)))
    #                 # print("grads shape: {}".format(np.shape(grads)))
    #                 nonzeros = np.count_nonzero(grads.detach().cpu().numpy())
    #                 # if True: #counter < 100:
    #                 #     print("counter: {}".format(counter))
    #                 #     #print("fmaps: {}".format(fmaps))
    #                 #     print("nonzeros: {}".format(nonzeros))
    #                 #     print("fmaps shape: {}".format(np.shape(fmaps)))
    #                 #     print("grads shape: {}".format(np.shape(grads)))
    #                 self._compute_grad_weights(grads)
    #                 if nonzeros == 0 or not isinstance(fmaps, torch.Tensor) or not isinstance(grads, torch.Tensor):
    #                     counter += 1
    #                     print("Skipped layer: {}".format(layer))
    #                     continue
    #                 print("Dismissed the last {} module layers (Note: This number can be inflated if the model contains many nested module layers)".format(counter))
    #                 print("Selected module layer: {}".format(layer))
    #                 fmap_list.append(self._find(self.fmap_pool, layer)[i])
    #                 grads = self._find(self.grad_pool, layer)[i]
    #                 weight_list.append(self._compute_grad_weights(grads))
    #                 break
    #             except ValueError:
    #                 counter += 1
    #             except RuntimeError:
    #                 counter += 1
    #
    #     return fmap_list, weight_list
    #
    # def generate_helper(self, fmaps, weights):
    #     gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    #     gcam = F.relu(gcam)
    #
    #     gcam = F.interpolate(
    #         gcam, self.image_shape, mode="bilinear", align_corners=False
    #     )
    #
    #     B, C, H, W = gcam.shape
    #     gcam = gcam.view(B, -1)
    #     gcam -= gcam.min(dim=1, keepdim=True)[0]
    #     gcam /= gcam.max(dim=1, keepdim=True)[0]
    #     gcam = gcam.view(B, C, H, W)
    #
    #     return gcam
    #
    # def generate(self, target_layer, dim=2):
    #     if target_layer == "auto":
    #         fmaps, weights = self.select_highest_layer()
    #         gcam = []
    #         for i in range(self.logits.shape[0]):
    #             gcam.append(self.generate_helper(fmaps[i].unsqueeze(0), weights[i].unsqueeze(0)))
    #     else:
    #         fmaps = self._find(self.fmap_pool, target_layer)
    #         grads = self._find(self.grad_pool, target_layer)
    #         weights = self._compute_grad_weights(grads)
    #         gcam_tensor = self.generate_helper(fmaps, weights)
    #         gcam = []
    #         for i in range(self.logits.shape[0]):
    #             tmp = gcam_tensor[i].unsqueeze(0)
    #             gcam.append(tmp)
    #     return gcam

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        # fmaps = fmaps.unsqueeze(2).unsqueeze(3)
        # grads = grads.unsqueeze(2).unsqueeze(3)
        # if len(fmaps.shape) == 2:
        #     fmaps = fmaps.unsqueeze(2)
        #     grads = grads.unsqueeze(2)
        # if len(fmaps.shape) == 3:
        #     fmaps = fmaps.unsqueeze(3)
        #     grads = grads.unsqueeze(3)
        # print("fmaps.shape: ", fmaps.shape)
        # print("grads.shape: ", grads.shape)
        weights = self._compute_grad_weights(grads)
        # print("weights.shape: ", weights.shape)
        # print("fmaps.shape: ", fmaps.shape)
        # print("grads.shape: ", grads.shape)
        # print("weights.shape: ", weights.shape)
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        # print("gcam.shape: ", gcam.shape)
        gcam = F.relu(gcam)
        # print("gcam.shape: ", gcam.shape)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )
        # print("gcam.shape: ", gcam.shape)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def occlusion_sensitivity(model, batch, ids, mean=None, patch=35, stride=1, n_batches=128):
    """Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17

    Originally proposed in:
    "Visualizing and Understanding Convolutional Networks"
    https://arxiv.org/abs/1311.2901
    """
    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    batch["raw_image"] = F.pad(batch["raw_image"], (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = batch["raw_image"].shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(batch).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        # print("Test 1")
        # batches = []
        # batch_ids = []
        for grid_h, grid_w in tqdm(anchors[i: i + n_batches]):
            # print("Test 2")
            batch_ = _batch_clone(batch)    # batch.clone()
            batch_["raw_image"][..., grid_h: grid_h + patch_H, grid_w: grid_w + patch_W] = mean
            score = model(batch_).detach().gather(1, ids)
            scoremaps.append(score)
        #     batches.append(batch_)
        #     batch_ids.append(ids)
        # batches = _batch_cat(batches) #torch.cat(batches, dim=0)
        # batch_ids = torch.cat(batch_ids, dim=0)
        # scores = model(batches).detach().gather(1, batch_ids)
        # scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps


def _batch_clone(batch):
    clone = {}
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            clone[key] = batch[key].clone()
        else:
            clone[key] = copy.deepcopy(batch[key])
    return clone


def _batch_cat(batch_list):
    cat_batch = {}
    for key in batch_list[0].keys():
        cat_batch[key] = [batch[key] for batch in batch_list]
        if isinstance(batch_list[0][key], torch.Tensor):
            cat_batch[key] = torch.cat(cat_batch[key], dim=0)
    return cat_batch
