import albumentations as A
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class OxfordIIITPet(Dataset):
    """
    Dataset class for Oxford-IIIT Pet dataset.

    Args:
        imgs_file (list): List of file paths to the images.
        masks_file (list): List of file paths to the corresponding masks.
        height (int): Height of the images and masks (default: 240).
        width (int): Width of the images and masks (default: 240).
        transform_img (callable): Optional transform to be applied to the images (default: None).
        transform_mask (callable): Optional transform to be applied to the masks (default: None).

    Attributes:
        imgs_file (list): List of file paths to the images.
        masks_file (list): List of file paths to the corresponding masks.
        height (int): Height of the images and masks.
        width (int): Width of the images and masks.
        transform_img (callable): Transform to be applied to the images.
        transform_mask (callable): Transform to be applied to the masks.
    """

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
        """
        Display the image and mask corresponding to the given index.

        Parameters:
        - idx (int): The index of the image and mask to display.

        Returns:
        None
        """

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
