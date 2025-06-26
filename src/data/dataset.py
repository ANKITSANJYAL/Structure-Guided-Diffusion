"""
Dataset module for Oxford Flowers 102 and other datasets.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import torchvision.transforms as transforms


class OxfordFlowersDataset(Dataset):
    """
    Oxford Flowers 102 dataset loader.
    
    This dataset contains 8,189 images of 102 flower categories.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
    ):
        """
        Initialize the Oxford Flowers dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transforms to apply
            download: Whether to download the dataset
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load dataset metadata
        self._load_metadata()
        
        # Download if requested
        if download:
            self._download_dataset()
    
    def _load_metadata(self):
        """Load dataset metadata and splits."""
        # Load processed data
        processed_dir = os.path.join(os.path.dirname(self.root_dir), 'processed')
        
        # Load image labels and splits
        image_labels_path = os.path.join(processed_dir, 'image_labels.npy')
        train_ids_path = os.path.join(processed_dir, 'train_ids.npy')
        val_ids_path = os.path.join(processed_dir, 'val_ids.npy')
        test_ids_path = os.path.join(processed_dir, 'test_ids.npy')
        
        if not all(os.path.exists(p) for p in [image_labels_path, train_ids_path, val_ids_path, test_ids_path]):
            raise FileNotFoundError(
                "Dataset metadata not found. Please run the data preparation script first."
            )
        
        # Load data
        self.image_labels = np.load(image_labels_path)
        self.train_ids = np.load(train_ids_path)
        self.val_ids = np.load(val_ids_path)
        self.test_ids = np.load(test_ids_path)
        
        # Set split indices
        if self.split == 'train':
            self.indices = self.train_ids
        elif self.split == 'val':
            self.indices = self.val_ids
        elif self.split == 'test':
            self.indices = self.test_ids
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Get image paths
        self.image_paths = []
        for idx in self.indices:
            # Oxford Flowers images are named as image_00001.jpg, image_00002.jpg, etc.
            image_name = f"image_{idx+1:05d}.jpg"
            image_path = os.path.join(self.root_dir, image_name)
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
            else:
                print(f"Warning: Image not found: {image_path}")
        
        # Get labels for this split
        self.labels = self.image_labels[self.indices]
        
        print(f"Loaded {self.split} split: {len(self.image_paths)} images")
    
    def _download_dataset(self):
        """Download the Oxford Flowers 102 dataset."""
        import urllib.request
        import tarfile
        import scipy.io
        
        # Create directories
        os.makedirs(self.root_dir, exist_ok=True)
        processed_dir = os.path.join(os.path.dirname(self.root_dir), 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Download URLs
        base_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
        files = {
            "102flowers.tgz": "images",
            "imagelabels.mat": "labels",
            "setid.mat": "splits"
        }
        
        for filename, file_type in files.items():
            url = base_url + filename
            output_path = os.path.join(self.root_dir, filename)
            
            if not os.path.exists(output_path):
                print(f"Downloading {file_type}...")
                urllib.request.urlretrieve(url, output_path)
            else:
                print(f"{file_type} already exists")
        
        # Extract images
        images_dir = os.path.join(self.root_dir, "jpg")
        if not os.path.exists(images_dir):
            print("Extracting images...")
            with tarfile.open(os.path.join(self.root_dir, "102flowers.tgz"), 'r:gz') as tar:
                tar.extractall(self.root_dir)
        
        # Process annotations
        print("Processing annotations...")
        labels = scipy.io.loadmat(os.path.join(self.root_dir, "imagelabels.mat"))
        splits = scipy.io.loadmat(os.path.join(self.root_dir, "setid.mat"))
        
        # Process data
        image_labels = labels['labels'].flatten() - 1  # Convert to 0-indexed
        train_ids = splits['trnid'].flatten() - 1
        val_ids = splits['valid'].flatten() - 1
        test_ids = splits['tstid'].flatten() - 1
        
        # Save processed data
        np.save(os.path.join(processed_dir, 'image_labels.npy'), image_labels)
        np.save(os.path.join(processed_dir, 'train_ids.npy'), train_ids)
        np.save(os.path.join(processed_dir, 'val_ids.npy'), val_ids)
        np.save(os.path.join(processed_dir, 'test_ids.npy'), test_ids)
        
        print("Dataset download and processing complete!")
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single image and its label.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
            "sweet pea", "english marigold", "tiger lily", "moon orchid",
            "bird of paradise", "monkshood", "globe thistle", "snapdragon",
            "colt's foot", "king protea", "spear thistle", "yellow iris",
            "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
            "giant white arum lily", "fire lily", "pincushion flower",
            "fritillary", "red ginger", "grape hyacinth", "corn poppy",
            "prince of wales feathers", "stemless gentian", "artichoke",
            "sweet william", "carnation", "garden phlox", "love in the mist",
            "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
            "cape flower", "great masterwort", "siam tulip", "lenten rose",
            "barbeton daisy", "daffodil", "sword lily", "poinsettia",
            "bolero deep blue", "wallflower", "marigold", "buttercup",
            "oxeye daisy", "common dandelion", "petunia", "wild pansy",
            "primula", "sunflower", "pelargonium", "bishop of llandaff",
            "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
            "cautleya spicata", "japanese anemone", "black-eyed susan",
            "silverbush", "californian poppy", "osteospermum", "spring crocus",
            "bearded iris", "windflower", "tree poppy", "gazania",
            "azalea", "water lily", "rose", "thorn apple", "morning glory",
            "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
            "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
            "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
            "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
            "mallow", "mexican petunia", "bromelia", "blanket flower",
            "trumpet creeper", "blackberry lily", "common tulip", "wild rose"
        ]


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars dataset loader.
    
    This dataset contains 16,185 images of 196 car categories.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the Stanford Cars dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'test')
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load dataset metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata."""
        import scipy.io
        
        # Load annotations
        annotations_path = os.path.join(self.root_dir, 'cars_annos.mat')
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        
        # Load data
        annotations = scipy.io.loadmat(annotations_path)
        
        # Extract information
        image_paths = annotations['annotations'][0]
        labels = annotations['class'][0]
        
        # Filter by split
        self.image_paths = []
        self.labels = []
        
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Stanford Cars uses 0-indexed labels
            label = label[0][0] - 1  # Convert to 0-indexed
            
            # Determine split based on label (first 98 classes are training)
            if self.split == 'train' and label < 98:
                self.image_paths.append(path[0])
                self.labels.append(label)
            elif self.split == 'test' and label >= 98:
                self.image_paths.append(path[0])
                self.labels.append(label)
        
        # Convert to full paths
        self.image_paths = [os.path.join(self.root_dir, path) for path in self.image_paths]
        
        print(f"Loaded {self.split} split: {len(self.image_paths)} images")
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single image and its label.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class CUB200Dataset(Dataset):
    """
    CUB-200-2011 dataset loader.
    
    This dataset contains 11,788 images of 200 bird categories.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the CUB-200-2011 dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'test')
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load dataset metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata."""
        # Load image paths and labels
        images_path = os.path.join(self.root_dir, 'images.txt')
        labels_path = os.path.join(self.root_dir, 'image_class_labels.txt')
        splits_path = os.path.join(self.root_dir, 'train_test_split.txt')
        
        if not all(os.path.exists(p) for p in [images_path, labels_path, splits_path]):
            raise FileNotFoundError("Dataset files not found")
        
        # Load data
        with open(images_path, 'r') as f:
            image_data = [line.strip().split() for line in f.readlines()]
        
        with open(labels_path, 'r') as f:
            label_data = [line.strip().split() for line in f.readlines()]
        
        with open(splits_path, 'r') as f:
            split_data = [line.strip().split() for line in f.readlines()]
        
        # Create mappings
        image_to_label = {int(item[0]): int(item[1]) - 1 for item in label_data}  # 0-indexed
        image_to_split = {int(item[0]): int(item[1]) for item in split_data}
        
        # Filter by split
        self.image_paths = []
        self.labels = []
        
        for image_id, image_path in image_data:
            image_id = int(image_id)
            split_id = image_to_split[image_id]
            label = image_to_label[image_id]
            
            # 1 = training, 0 = test
            if (self.split == 'train' and split_id == 1) or (self.split == 'test' and split_id == 0):
                full_path = os.path.join(self.root_dir, 'images', image_path)
                if os.path.exists(full_path):
                    self.image_paths.append(full_path)
                    self.labels.append(label)
        
        print(f"Loaded {self.split} split: {len(self.image_paths)} images")
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single image and its label.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


def create_dataset(
    dataset_name: str,
    root_dir: str,
    split: str = 'train',
    transform: Optional[transforms.Compose] = None,
    **kwargs
) -> Dataset:
    """
    Create a dataset instance.
    
    Args:
        dataset_name: Name of the dataset ('oxford_flowers_102', 'stanford_cars', 'cub_200')
        root_dir: Root directory of the dataset
        split: Dataset split
        transform: Optional transforms
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    if dataset_name == 'oxford_flowers_102':
        return OxfordFlowersDataset(root_dir, split, transform, **kwargs)
    elif dataset_name == 'stanford_cars':
        return StanfordCarsDataset(root_dir, split, transform)
    elif dataset_name == 'cub_200':
        return CUB200Dataset(root_dir, split, transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}") 