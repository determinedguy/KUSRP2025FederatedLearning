
import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_dir = '/data/train/'
    test_dir = '/data/test/'

    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])


    train_dataset = datasets.MNIST(train_dir, train=True, download=True,
                                       transform=apply_transform)

    test_dataset = datasets.MNIST(test_dir, train=False, download=True,
                                    transform=apply_transform)

    # Veri setinin boyutunu kontrol etme
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = mnist_iid(train_dataset, args.num_users)
    else:
        # Sample Non-IID user data from Mnist
        if args.unequal:
            # Chose uneuqal splits for every user
            user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
        else:
            # Chose euqal splits for every user
            user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups

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


class Args:
    iid = True

args = Args()
train_dataset, test_dataset = get_dataset(args)
