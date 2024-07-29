
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet as Dataset
from torchvision import transforms

class OxfordIIITPet(Dataset):
    """
    Dataset class for the Oxford-IIIT Pet dataset.

    Args:
    - config (dict): Configuration dictionary containing the dataset parameters.
    - split (str): Split of the dataset to use (default: "trainval").
    - transform_img (callable): Optional image transformation function (default: None).
    - transform_mask (callable): Optional mask transformation function (default: None).

    Attributes:
    - config (dict): Configuration dictionary containing the dataset parameters.
    - height (int): Height of the images and masks.
    - width (int): Width of the images and masks.
    - transform_img (callable): Image transformation function.
    - transform_mask (callable): Mask transformation function.
    """

    def __init__(
        self,
        config,
        split="trainval",
        transform_img=None,
        transform_mask=None,
    ):
        super().__init__(
            root=config.root_dir,
            split=split,
            target_types="segmentation",
            download=True,
        )
        self.config = config
        self.height = self.config.height
        self.width = self.config.width
        self.transform_img = transform_img or self._default_transform_img()
        self.transform_mask = transform_mask or self._default_transform_mask()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img = self._load_image(self._images[idx])
        mask = self._load_image(self._segs[idx], mode="L")

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

        img = self._load_image(self._images[idx])
        mask = self._load_image(self._segs[idx])

        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax2.imshow(mask)
        plt.show()

    @staticmethod
    def _load_image(path, mode="RGB"):
        return Image.open(path).convert(mode)

    def _default_transform_img(self):
        return transforms.Compose(
            [
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        
    def _default_transform_mask(self):
        return transforms.Compose(
            [
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
            ]
        )
