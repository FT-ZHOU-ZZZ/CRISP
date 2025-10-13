import timm
import torch
from timm.layers import SwiGLUPacked
from peft import LoraConfig, get_peft_model
from torchvision import transforms


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def get_trans():
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            transforms.CenterCrop(224),
            MaybeToTensor(),
            transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ]
    )
    return transform


def get_model(device, ckpt_path):
    base = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        reg_tokens=4,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        mlp_ratio=6832 / 1280,
        init_values=1e-5,
        global_pool="",
    ).to(device)
    # insert lora layers into attention layers
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "attn.qkv",
            "attn.proj",
        ],
        lora_dropout=0.1,
    )
    model = get_peft_model(base, config)
    print(f"loading CRISP model from {ckpt_path}")
    # load the model
    pth = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    error = model.load_state_dict(pth, strict=True)
    print(error)
    model.eval()

    def func(image):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(image)
        class_token = output[:, 0]
        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding

    return func


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dummy input
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    img = img.convert("RGB")
    # feature extractor
    trans = get_trans()
    model = get_model(device, "CRISP.pth")
    img = trans(img).unsqueeze(0).to(device)
    print(img.shape)
    feature = model(img)
    print(feature.shape)
