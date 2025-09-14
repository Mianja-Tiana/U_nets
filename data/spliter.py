from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from data.datasets import BiologyDataset

def create_dataloaders(data_path, class_weight, batch_size=1):
    
    dataset = BiologyDataset(data_path)
    idx = range(len(dataset))
    
    train_idx, test_idx, train_cat, _ = train_test_split(
        idx, dataset.category, test_size=0.2, stratify=dataset.category
    )
    
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, stratify=train_cat
    )
    
    train_dataset = Subset(
        BiologyDataset(data_path, class_weight=class_weight, transforms=get_train_transforms((572, 572), (388, 388))),
        train_idx
    )
    
    val_dataset = Subset(
        BiologyDataset(data_path, class_weight=class_weight, transforms=get_val_transforms((572, 572), (388, 388))),
        val_idx
    )
    
    test_dataset = Subset(
        BiologyDataset(data_path, class_weight=class_weight, transforms=get_val_transforms((572, 572), (388, 388))),
        test_idx
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
