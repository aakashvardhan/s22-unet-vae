import random
import matplotlib.pyplot as plt


def plot_test_example(val_loader):
    images, _ = next(iter(val_loader))
    image = images[random.randint(0, len(images) - 1)]
    image = image.numpy().transpose((1, 2, 0))
    plt.imshow(image)
    plt.title("Sample Image")
    plt.show()