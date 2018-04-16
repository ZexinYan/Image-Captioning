import torch
import torch.utils.data as data
import os
from PIL import Image
from build_vocab import Vocabulary, JsonReader


class COCODataSet(data.Dataset):
    def __init__(self, image_dir: str, caption_json: str, images_json: str, vocabulary: Vocabulary, transform=None):
        """
        :param image_dir: Image Directory
        :param caption_json: caption Json file path
        :param images_json: image Json file path
        :param vocabulary: vocabulary class
        :param transform: image transformer
        """
        self.image_dir = image_dir
        self.images = JsonReader(images_json)
        self.caption = JsonReader(caption_json)
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self, index):
        image_id = self.images[index]
        caption = self.caption[image_id]

        image = Image.open(os.path.join(self.image_dir, image_id)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        target = list()
        target.append(self.vocabulary('<start>'))
        target.extend([self.vocabulary(token) for token in caption])
        target.append(self.vocabulary('<end>'))
        target = torch.Tensor(target)
        return image, target, image_id

    def __len__(self):
        return len(self.images)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, X, X).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, X, X).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, image_id = zip(*data)

    # Merge Image(from tuple of 3D to 4D)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, image_id, lengths


def get_loader(image_dir, caption_json, images_json, vocabulary, transform, batch_size, shuffle, num_workers):
    cocodataset = COCODataSet(image_dir=image_dir,
                              caption_json=caption_json,
                              images_json=images_json,
                              vocabulary=vocabulary,
                              transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=cocodataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
