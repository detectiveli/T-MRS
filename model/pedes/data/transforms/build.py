from . import transforms as T
from PIL import Image
import random, math
from .RandomErasing import RandomErasing

def build_transforms(cfg, mode='train'):
    assert mode in ['train', 'test', 'val']
    min_size = cfg.SCALES[0]
    max_size = cfg.SCALES[1]
    assert min_size <= max_size

    if mode == 'train':
        flip_prob = cfg.TRAIN.FLIP_PROB
    elif mode == 'test':
        flip_prob = cfg.TEST.FLIP_PROB
    else:
        flip_prob = cfg.VAL.FLIP_PROB

    to_bgr255 = True

    normalize_transform = T.Normalize(
        mean=cfg.NETWORK.PIXEL_MEANS, std=cfg.NETWORK.PIXEL_STDS, to_bgr255=to_bgr255
    )

    # transform = T.Compose(
    #     [
    #         T.Resize(min_size, max_size),
    #         T.RandomHorizontalFlip(flip_prob),
    #         T.ToTensor(),
    #         normalize_transform,
    #         T.FixPadding(min_size, max_size, pad=0)
    #     ]
    # )
    if mode == 'train':
        transform = T.Compose(
            [
                T.Resize(min_size, max_size),
                #RandomSizedRectCrop(min_size, max_size),
                T.RandomHorizontalFlip(flip_prob),
                T.ColorChange(),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=flip_prob, mean=cfg.NETWORK.PIXEL_MEANS),
            ]
        )   
    else:
            transform = T.Compose(
            [
                T.Resize(min_size, max_size),
                #RectScale(min_size, max_size),
                T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )

    return transform


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation
    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                return img.resize((self.width, self.height), self.interpolation)

        scale = RectScale(self.height, self.width,interpolation=self.interpolation)
        return scale(img)

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)
