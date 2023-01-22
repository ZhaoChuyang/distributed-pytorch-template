import os
from ...utils import check_path_is_image
from ..catalog import DATASET_CATALOG


def load_cifar_image_dataset(root, split: str, **kwargs):
    """
    Load the cifar dataset in a list of dataset dicts. Each dict contains:
        * image_path: path to the input image
        * target: class label of the image.

    Args:
        root (str): root path of the dataset.
        split (str): split of the dataset, should be one of ['train', 'test'].

    Returns:
        dataset (list): a list of dataset dicts.

    NOTE: this function does not read image, it only returns the paths to the images
    and leave the read and transforms to the wrapper dataset.
    """
    assert split in ['train', 'test']
    dataset = []
    split_dir = os.path.join(root, split)
    label_dict = {label: label_id for label_id, label in enumerate(sorted(os.listdir(split_dir)))}
    for label in label_dict:
        for filename in os.listdir(os.path.join(split_dir, label)):
            image_path = os.path.join(split_dir, label, filename)
            if not check_path_is_image(image_path):
                continue
            dataset.append(dict(image_path=image_path, target=label_dict[label], **kwargs))
    return dataset


def register_cifar_image_dataset(name: str, root: str, split: str, **kwargs):
    """
    Register the cifar-image dataset in DATASET_CATALOG.
    """
    DATASET_CATALOG.register(name, lambda: load_cifar_image_dataset(root, split, **kwargs))


if __name__ == '__main__':
    # test
    dataset = load_cifar_image_dataset('/home/zhaochuyang/Workspace/datasets/CIFAR-10-images', 'train')
    print(dataset[0:10])
