import albumentations as A
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPet(Dataset):
    def __init__(
        self,
        imgs_file,
        masks_file,
        height=240,
        width=240,
        transform_img=None,
        transform_mask=None,
    ):
        self.imgs_file = imgs_file
        self.masks_file = masks_file
        self.height = height
        self.width = width
        self.transform_img = transform_img or self._default_img_transform()
        self.transform_mask = transform_mask or self._default_mask_transform()

    def __len__(self):
        return len(self.imgs_file)

    def __getitem__(self, idx):
        img = self._load_image(self.imgs_file[idx])
        mask = self._load_image(self.masks_file[idx], mode="L")

        transformed_img = self.transform_img(img)
        transformed_mask = self.transform_mask(mask)
        transformed_mask = transformed_mask * 255.0 - 1.0

        return {"image": transformed_img, "mask": transformed_mask}

    def show_img_mask(self, idx):
        img = self._load_image(self.imgs_file[idx])
        mask = self._load_image(self.masks_file[idx])

        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax2.imshow(mask)
        plt.show()

    @staticmethod
    def _load_image(path, mode="RGB"):
        return Image.open(path).convert(mode)

    def _default_img_transform(self):
        return A.Compose(
            [
                A.Resize(self.height, self.width),
                ToTensorV2(),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def _default_mask_transform(self):
        return A.Compose([A.Resize(self.height, self.width), ToTensorV2()])
