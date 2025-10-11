import random

import albumentations as A
import cv2
import numpy as np
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.crops.functional import crop


class NoiseAugmentation:
    def __init__(self, loc_num=4, resolution=384):
        self.keypoint = ['left_edge', 'right_edge', 'top_edge', 'left_eyebrow', 'right_eyebrow', 'nose', 'left_eye',
                         'right_eye',
                         'mouth']
        self.loc_num = loc_num
        self.landmark_loc_index = {
            'left_edge': (0, 8),
            'right_edge': (8, 17),
            'top_edge': (68, 81),
            'left_eyebrow': (17, 22),
            'right_eyebrow': (22, 27),
            'nose': (27, 36),
            'left_eye': (36, 42),
            'right_eye': (42, 48),
            'mouth': (48, 68)
        }
        self.resolution = resolution
        self.aug = self.init_data_aug_method()

    def init_data_aug_method(self):
        trans = A.Compose([
            A.OneOf([
                A.GaussianBlur(),
                A.MedianBlur(),
                A.GaussNoise(),
                A.ISONoise()
            ], p=0.5),
            A.OneOf([
                IsotropicResize(max_side=self.resolution, interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.resolution, interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.resolution, interpolation_down=cv2.INTER_LINEAR,
                                interpolation_up=cv2.INTER_LINEAR),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.1],
                                           contrast_limit=[-0.1, 0.1]),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=40,
                               quality_upper=100, p=1)
        ])
        return trans

    def noise_aug(self, image, landmark):
        # return image, np.zeros((384,384,1))
        N = random.randint(1, self.loc_num)
        mask = np.zeros((image.shape[0], image.shape[1]))
        selected = random.sample(self.keypoint, N)
        selected_landmark = self.get_face_landmark_dict(landmark, selected) / 256 * image.shape[0]
        mask = self.get_mask(mask, selected_landmark)
        mask = np.expand_dims(mask, axis=2)
        image_aug = (1 - mask) * image + mask * self.aug(image=image)['image']
        return image_aug.astype(np.uint8), mask

    def get_mask(self, mask, landmark):
        t = cv2.fillConvexPoly(mask, np.int32(cv2.convexHull(landmark)), color=255) / 255
        return t

    def get_face_landmark_dict(self, landmark, loc="all"):
        if isinstance(loc, list) or isinstance(loc, tuple):
            return np.concatenate([landmark[self.landmark_loc_index[i][0]:self.landmark_loc_index[i][1]] for i in loc])
        else:
            return landmark[self.landmark_loc_index[loc][0]:self.landmark_loc_index[loc][1]]

    # def aug(self, image):
    #     pass


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.p = p
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class Resize4xAndBack(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h),
                         interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists, self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1]))
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"
