import os
import nltk
from PIL import Image
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

#  Vocabulary class that uses vocab.pkl file to map words to indices and vice versa
class Vocabulary:
    def __init__(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def get_word(self, idx):
        return self.idx2word.get(idx, "<unk>")

    def __len__(self):
        return len(self.word2idx)


# COCO Caption Dataset
class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_path, vocab, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        image_id = self.coco.anns[ann_id]['image_id']
        image_info = self.coco.loadImgs(image_id)[0]
        path = image_info['file_name']

        # Load and transform image
        image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Tokenize and numericalize caption
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption_ids = [self.vocab("<start>")] + [self.vocab(token) for token in tokens] + [self.vocab("<end>")]
        caption_tensor = torch.tensor(caption_ids)

        return image, caption_tensor

    def __len__(self):
        return len(self.ids)


# Collate Function for DataLoader
def collate_fn(batch):
    images, captions = zip(*batch)

    # Pad captions to same length
    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions