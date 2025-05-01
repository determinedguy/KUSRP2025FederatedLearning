
import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
import os
from options import args_parser



'''def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_dir = 'data/train/'
    test_dir = 'data/test/'

    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(train_dir, train=True, download=True,
                                       transform=apply_transform)

    test_dataset = datasets.MNIST(test_dir, train=False, download=True,
                                    transform=apply_transform)

    # To chechk the size of the dataset
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = mnist_iid(train_dataset, args.num_users)
        #user_groups = mnist_iid(train_dataset, 10)
    else:
        # Sample Non-IID user data from Mnist
        if args.unequal:
            # Chose uneuqal splits for every user
            user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            #user_groups = mnist_noniid_unequal(train_dataset, 10)
        else:
            # Chose euqal splits for every user
            user_groups = mnist_noniid(train_dataset, args.num_users)
            #user_groups = mnist_noniid(train_dataset, 10)

    return train_dataset, test_dataset, user_groups
'''

def get_dataset(dataset_path, image_size=(224, 224)):
    """
    Loads train and test datasets from a directory structure like:
    dataset_path/
        train/
            class1/
            class2/
            ...
        test/
            class1/
            class2/
            ...
    
    Args:
        dataset_path (str): Root path to dataset (should contain 'train' and 'test' folders).
        image_size (tuple): Desired image size (default is 224x224 for pretrained models).
    
    Returns:
        train_dataset, test_dataset, num_classes
    """
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    # Transform: normalize to ImageNet statistics (standard for pretrained models)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    num_classes = len(train_dataset.classes)
    print(f"Loaded dataset from: {dataset_path}")
    print(f" - Number of classes: {num_classes}")
    print(f" - Training samples: {len(train_dataset)}")
    print(f" - Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset, num_classes

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return