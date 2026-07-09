import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
            
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes))
            else:
                self.all_categories = unseen_classes
        self.all_categories = sorted(self.all_categories)
        self.category_to_idx = {category: idx for idx, category in enumerate(self.all_categories)}

        self.all_sketches_path = []
        self.all_photos_path = {}
        self.photo_id_to_path = {}
        self.sketch_to_photo = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png')))
            photo_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                photo_paths.extend(glob.glob(os.path.join(self.opts.data_dir, 'photo', category, ext)))
            self.all_photos_path[category] = photo_paths
            self.photo_id_to_path[category] = {
                self._instance_id(path): path for path in photo_paths
            }

        self._validate_data()
        if self.opts.retrieval_level == 'fine_grain':
            self._build_fine_grain_pairs()

    def __len__(self):
        return len(self.all_sketches_path)

    @staticmethod
    def _instance_id(path):
        return os.path.splitext(os.path.basename(path))[0]

    @classmethod
    def _candidate_photo_ids(cls, sketch_path):
        stem = cls._instance_id(sketch_path)
        candidates = [stem]
        for sep in ('-', '_'):
            if sep in stem:
                candidates.append(stem.rsplit(sep, 1)[0])
        candidates.append(stem.split('-')[0])

        seen = set()
        return [item for item in candidates if not (item in seen or seen.add(item))]

    def _validate_data(self):
        missing = [cat for cat in self.all_categories if len(self.all_photos_path[cat]) == 0]
        if missing:
            raise RuntimeError(
                'No photos found for categories: %s. Expected files under %s'
                % (missing[:10], os.path.join(self.opts.data_dir, 'photo')))
        if len(self.all_sketches_path) == 0:
            raise RuntimeError(
                'No sketches found. Expected PNG files under %s'
                % os.path.join(self.opts.data_dir, 'sketch'))

    def _build_fine_grain_pairs(self):
        unmatched = []
        for sketch_path in self.all_sketches_path:
            category = sketch_path.split(os.path.sep)[-2]
            photo_lookup = self.photo_id_to_path[category]
            matched_id = None
            for candidate in self._candidate_photo_ids(sketch_path):
                if candidate in photo_lookup:
                    matched_id = candidate
                    break

            if matched_id is None:
                unmatched.append(sketch_path)
            else:
                self.sketch_to_photo[sketch_path] = photo_lookup[matched_id]

        if unmatched:
            examples = '\n'.join(unmatched[:10])
            raise RuntimeError(
                'Could not infer fine-grained photo pairs for %d sketches. '
                'Expected names like sketch xxx-1.png -> photo xxx.jpg. Examples:\n%s'
                % (len(unmatched), examples))
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        
        sk_path  = filepath
        if self.opts.retrieval_level == 'fine_grain':
            img_path = self.sketch_to_photo[sk_path]
            same_category_neg = [
                path for path in self.all_photos_path[category]
                if self._instance_id(path) != self._instance_id(img_path)
            ]
            if not same_category_neg:
                raise RuntimeError(
                    'Fine-grained training needs at least two photos per category. '
                    'Category %s only has one usable photo.' % category)
            neg_path = np.random.choice(same_category_neg)
        else:
            neg_classes = self.all_categories.copy()
            neg_classes.remove(category)
            img_path = np.random.choice(self.all_photos_path[category])
            neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])
        target_id = self._instance_id(img_path)
        class_idx = self.category_to_idx[category]

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename, target_id, class_idx,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename, target_id, class_idx)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
