import torch

from .camvid import CamVid

def camvid_loaders(path, batch_size, num_workers, transform_train, transform_test, 
                shuffle_train=True, joint_transform=None, ft_joint_transform=None, ft_batch_size=1, **kwargs):

    #load training and finetuning datasets
    print(path)
    train_set = CamVid(root=path, split='train', joint_transform=joint_transform, transform=transform_train, **kwargs)
    ft_train_set = CamVid(root=path, split='train', joint_transform=ft_joint_transform, transform=transform_train, **kwargs)

    val_set = CamVid(root=path, split='val', joint_transform=None, transform=transform_test, **kwargs)
    test_set = CamVid(root=path, split='test', joint_transform=None, transform=transform_test, **kwargs)

    num_classes = 11 # hard coded labels ehre
    
    return {'train': torch.utils.data.DataLoader(
                        train_set, 
                        batch_size=batch_size, 
                        shuffle=shuffle_train, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'fine_tune': torch.utils.data.DataLoader(
                        ft_train_set, 
                        batch_size=ft_batch_size, 
                        shuffle=shuffle_train, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'val': torch.utils.data.DataLoader(
                        val_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True
                ),
            'test': torch.utils.data.DataLoader(
                        test_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True
                )}, num_classes
