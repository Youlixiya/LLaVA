import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .sam import build_sam_vit_b

class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

class BaseImageProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.image_mean = mean
        self.image_std = std
        self.normalize = transforms.Normalize(mean, std)

class SimpleCLIPImageProcessor(BaseImageProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        self.crop_size = {'height':image_size,
                          'width': image_size}
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, images, return_tensors='pt'):
        return self.preprocess(images, return_tensors)
    
    def preprocess(self, images, return_tensors='pt'):
        images_tensor = []
        if type(images) == list:
            for image in images:
                images_tensor.append(self.transform(image))
            images_tensor = torch.stack(images_tensor)
        else:
            images_tensor = self.transform(images)[None]
        return {'pixel_values': images_tensor}

class CLIPSAMVisionTower(nn.Module):
    sam_checkpoint='./ckpts/sam/sam_vision_tower.pt'
    clip_checkpoint='./ckpts/clip-vit-large-patch14'
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.clip_checkpoint)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        image_processor = CLIPImageProcessor.from_pretrained(self.clip_checkpoint)
        image_processor = SimpleCLIPImageProcessor(image_processor.crop_size['height'])
        image_processor_high = SimpleCLIPImageProcessor(1024)
        self.image_processor = [image_processor, image_processor_high]
        self.vision_tower = CLIPVisionModel.from_pretrained(self.clip_checkpoint)
        self.vision_tower_high = build_sam_vit_b()
        self.vision_tower_high.load_state_dict(torch.load(self.sam_checkpoint))
        self.vision_tower.requires_grad_(False)
        self.vision_tower_high.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                tmp_image_feature = []
                # print(image[0])
                image_forward_out = self.vision_tower(image[0].to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(self.dtype)
                tmp_image_feature.append(image_feature)
                tmp_image_feature.append(self.vision_tower_high(image[1].to(device=self.device, dtype=self.dtype).unsqueeze(0)).flatten(2).permute(0, 2, 1))
                # image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(torch.cat(tmp_image_feature, dim=-1))
            image_features = torch.cat(image_features)
        else:
            tmp_image_feature = []
            image_forward_outs = self.vision_tower(images[0].to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_feature = self.feature_select(image_forward_outs).to(self.dtype)
            # image_feature.append(self.vision_tower(images[0].to(device=self.device, dtype=self.dtype), True)[0])
            tmp_image_feature.append(image_feature)
            image_feature.append(self.vision_tower_high(images[1].to(device=self.device, dtype=self.dtype)).flatten(2).permute(0, 2, 1))
            image_features.append(torch.cat(tmp_image_feature, dim=-1))

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size + self.vision_tower_high.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
