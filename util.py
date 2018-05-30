import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def two_set_sampler(dataset, split=0.9, shuffle=True):
    N = len(dataset)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    splitpoint = int(split * N)

    # Create samplers based on the splitted data
    set1_idx, set2_idx = indices[:splitpoint], indices[splitpoint:]
    set1_sampler = SubsetRandomSampler(set1_idx)
    set2_sampler = SubsetRandomSampler(set2_idx)

    return set1_sampler, set2_sampler


def create_dataloaders(dataset, batch_size, split=0.9, num_workers=2):
    # Apparently the pin_memory parameter is important when using
    # CUDA
    train_sampler, valid_sampler = two_set_sampler(dataset, split)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers
    )
    return train_loader, valid_loader
