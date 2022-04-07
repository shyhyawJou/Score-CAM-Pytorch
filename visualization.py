import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
import cv2 as cv
from matplotlib import cm
import numpy as np



class ScoreCAM:
    """
    #### Args:
        layer_name: module name (not child name), if None, 
                    will use the last layer before average pooling
                    , default is None
    """
    def __init__(self, model, device, layer_name=None):
        if layer_name is None:
            layer_name = self.get_layer_name(model)
        
        if layer_name is None:
            raise ValueError(
                "There is no global average pooling layer, plz specify 'layer_name'"
            )
                
        for name, layer in model.named_modules():
            if name == layer_name:
                layer.register_forward_hook(self.forward_hook)
                break
        
        model = model.to(device)
        self.model = model
        self.device = device
        self.feature_maps = {}

    def get_heatmap(self, img, img_tensor):
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # outputs shape = 1x2
            outputs = self.model(img_tensor)
            feature_maps = self.feature_maps["output"]
            feature_maps = TF.resize(
                feature_maps, 
                img_tensor.shape[-2:], 
                InterpolationMode.BILINEAR
            )

            min_value = feature_maps.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
            max_value = feature_maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
            norm_feature_map = (feature_maps - min_value) / (max_value - min_value) 

            # (b, c, 1, h, w)
            norm_feature_map = norm_feature_map[:, :, None, ...]
            # (b, 1, c, h, w)
            img_tensor = img_tensor[:, None, ...]
            
            inputs = norm_feature_map * img_tensor
            inputs = inputs.view(*inputs.shape[1:])
            nc_for_plot = int(0.2 * inputs.shape[0])
            logit = self.model(inputs[:nc_for_plot])
            pred_label = logit.max(1)[1]

            score = F.softmax(logit, 1)[torch.arange(pred_label.numel()), pred_label]
            cam = (score.view(1, -1, 1, 1) * feature_maps[:, :nc_for_plot, :, :]).sum(1)

            F.relu(cam, True)
            cam = cam / cam.max() * 255
            cam = cam.to(dtype=torch.uint8, device="cpu")
            cam = cam.numpy().transpose(1,2,0)
            cam = cv.resize(cam, img.size[:2], interpolation=4)
            cam = np.uint8(255 * cm.get_cmap("jet")(cam.squeeze()))

            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
            img_size = img.shape[:2][::-1] # w, h

            overlay = np.uint8(0.6*img + 0.4 * cam[:,:,:3])
            overlay = Image.fromarray(overlay)
            if overlay.size != img_size:
                overlay = overlay.resize(img_size, Image.BILINEAR)
            
        return outputs.detach(), overlay

    def get_layer_name(self, model):
        layer_name = None
        for n, m in model.named_modules():
            if isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                layer_name = tmp
            tmp = n
        return layer_name

    def forward_hook(self, module, x, y):
        #self.feature_maps["input"] = x
        self.feature_maps["output"] = y

