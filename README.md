# Overview
The implementation of [ScoreCAM](https://arxiv.org/abs/1910.01279) for getting the attention map of CNN

# Usage
My code is very easy to use

### step 1: create the EigenCAM object and model

```
model = your_pytorch_model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
scorecam = ScoreCAM(model, device, layer_name=None)
```  

### step 2: get the heatmap
```
preprocess = your_preprocess
img = Image.open(img_path)  
img_tensor = preprocess(img).unsqueeze_(0).to(device)  
outputs, overlay = scorecam.get_heatmap(img, img_tensor)
overlay.show() # show the heatmap
```

# Complete Example
```
from PIL import Image

import torch
from torchvision import transfoms as T

from visualization import EigenCAM


class_name = ['Class A', 'Class B']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
preprocess = T.Compose([
                        T.ToTensor()
                       ])  

# create the ScoreCAM object 
scorecam = ScoreCAM(model, device)  

img = Image.open(img_path)  
img_tensor = preprocess(img).unsqueeze_(0).to(device)  
outputs, overlay = scorecam.get_heatmap(img, img_tensor)
_, pred_label = outputs.max(1)
pred_class = class_name[pred_label.item()]
conf = F.softmax(outputs, 1).squeeze()[pred_label]

print("Result:", pred_class)
print("Confidence:", conf)

# show the heatmap
overlay.show() 
```

# Reference
Original paper:  
[Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](https://arxiv.org/abs/2008.00299)  
