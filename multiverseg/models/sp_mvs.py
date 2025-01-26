"""
Combine pre-trained ScribblePrompt and MultiverSeg network
"""
import os
import torch
import torch.nn as nn
import warnings
import pathlib
from typing import Optional

from pylot.util import ShapeChecker

from scribbleprompt.models.unet import ScribblePromptUNet, prepare_inputs
from multiverseg.models.network import MultiverSegNet

checkpoint_dir = pathlib.Path(os.path.realpath(__file__)).parent.parent.parent / "checkpoints"

class MultiverSeg(nn.Module):

    weights = {
        "v0": checkpoint_dir / "MultiverSeg_v0_nf256_res128.pt"
    }

    def __init__(self, version: str = "v0", min_context: int = 1, device = None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.version = version
        self.device = device
        self.weights = self.weights[version]
        self.min_context = min_context

        self.multiverseg = MultiverSegNet(
            in_channels=[5, 2],
            encoder_blocks=[256, 256, 256, 256],
            block_kws=dict(conv_kws=dict(norm="layer")),
            cross_relu=True
        ).to(self.device)
        self.multiverseg.load_state_dict(
            torch.load(self.weights)["model"]
        )

        self.scribbleprompt = ScribblePromptUNet(version='v1', device=self.device)


    def to(self, device):
        self.multiverseg.to(device)
        self.scribbleprompt.to(device)
        self.device = device
        
    def forward(self, target_image, context_images = None, context_labels = None):
        
        sc = ShapeChecker()
        sc.check(target_image, "B 5 H W")
        
        if context_images is not None:
            sc.check(context_images, "B S 1 H W")
        if context_labels is not None:
            sc.check(context_labels, "B S 1 H W")

        given_context = not (context_images is None or context_labels is None)
        given_prompts = (target_image[:,:, 1:-1].sum() > 0)
        given_previous_prediction = (target_image[:,:, -1:].abs().sum() > 0)

        if given_context:
            if context_images.shape[1] < self.min_context:
                # Ignore the context!
                given_context = False

        if given_context:
            target_image = target_image.unsqueeze(1) # Shape: B x 1 x 5 x H x W
            return self.multiverseg(target_image, context_images, context_labels)
        else:
            return self.scribbleprompt.model(target_image) # Shape: 1 x 1 x H x W

    @torch.no_grad()
    def predict(self,
                img: torch.Tensor, # B x 1 x H x W
                # In-Context Inputs 
                context_images: Optional[torch.Tensor] = None, # B x n x 1 x H x W
                context_labels: Optional[torch.Tensor] = None, # B x n x 1 x H x W
                # Interactive Inputs
                point_coords: Optional[torch.Tensor] = None, # B x n x 2
                point_labels: Optional[torch.Tensor] = None, # B x n 
                scribbles: Optional[torch.Tensor] = None, # B x 2 x H x W
                box: Optional[torch.Tensor] = None, # B x 1 x 4
                mask_input: Optional[torch.Tensor] = None, # B x 1 x H x W
                # misc. 
                return_logits: bool = False):
        
        prompts = {
            'img': img,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'scribbles': scribbles,
            'box': box,
            'mask_input': mask_input,
        }

        # Prepare target image inputs (B x 5 x H x W)
        x = prepare_inputs(prompts).float().to(self.device)

        # Make prediction
        yhat = self.forward(x, context_images, context_labels)

        # B x 1 x H x W
        if return_logits:
            return yhat
        else:
            return torch.sigmoid(yhat)

