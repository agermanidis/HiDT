import runway

import torch
from torchvision import transforms

from hidt.networks.enhancement.RRDBNet_arch import RRDBNet
from hidt.style_transformer import StyleTransformer
from hidt.utils.preprocessing import GridCrop, enhancement_preprocessing


@runway.setup
def setup():
    transformer = StyleTransformer(
      './configs/daytime.yaml', 
      './trained_models/generator/daytime.pt',
      inference_size=256,
      device='cuda'
    )
    enhancer = RRDBNet(in_nc=48,
                    out_nc=3,
                    nf=64,
                    nb=5,
                    gc=32).to(torch.device('cuda'))
    enhancer.load_state_dict(torch.load('./trained_models/enhancer/enhancer.pth'))
    return transformer, enhancer


@runway.command('translate', inputs={'content': runway.image, 'style': runway.image}, outputs={'image': runway.image})
def translate(model, inputs):
    transformer, enhancer = model
    with torch.no_grad():
        style = transformer.get_style(inputs['style'])[0]
        crop_transform = GridCrop(4, 1, hires_size=256 * 4)
        crops = [img for img in crop_transform(inputs['content'])]
        out = transformer.transfer_images_to_styles(crops, [style], batch_size=4, return_pil=False)
        padded_stack = enhancement_preprocessing(out[0])
        out = enhancer(padded_stack)
        return transforms.ToPILImage()((out[0].cpu().clamp(-1, 1) + 1.) / 2.)


if __name__ == '__main__':
    runway.run()

