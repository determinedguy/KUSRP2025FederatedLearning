import numpy as np
from torchvision import datasets, transforms
import random
from typing import Dict, Set
import torch
from torch.utils.data import Dataset



def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


'''def mnist_noniid(dataset, num_users, shards_per_user=2):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    ''# 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset))   # artık 5245 olacak
    
    #labels = dataset.train_labels.numpy()

    # eğer torchvision>=0.2 ise .targets kullanabiliyoruz
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        # fallback: .samples içinden 2. elemanı al
        labels = np.array([s[1] for s in dataset.samples])


    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users  ''
    idxs   = np.arange(len(dataset))
    labels = np.array(dataset.targets) if hasattr(dataset, 'targets') \
             else np.array([s[1] for s in dataset.samples])

    # 2) label’a göre sırala
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs        = idxs_labels[0, :].astype(int)

    # 3) shard ve kullanıcı sayısını tanımla
    num_shards = num_users * shards_per_user
    shard_size = len(idxs) // num_shards

    # 4) shard’ları oluştur
    shard_idxs = [idxs[i*shard_size:(i+1)*shard_size] 
                  for i in range(num_shards)]

    # 5) shard’ları karıştır ve kullanıcıya ata
    random.shuffle(shard_idxs)
    user_groups = {}
    for i in range(num_users):
        user_shards = shard_idxs[i*shards_per_user:(i+1)*shards_per_user]
        user_groups[i] = set(np.concatenate(user_shards))

    return user_groups'''


def dirichlet_non_iid(dataset, num_users: int, alpha: float):
    """
    Sample non-I.I.D client data using a Dirichlet distribution.
    
    :param dataset: torchvision Dataset (has `targets` veya `samples`)
    :param num_users: kaç client’a bölüneceği
    :param alpha: Dirichlet hyperparametresi (>0). Küçük α ⇒ daha skewed.
    :return: dict kullanıcı_id → set(indeksler)
    """
    # 1) Etiketleri al
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([s[1] for s in dataset.samples])

    # 2) Sınıf sayısını bul
    classes = np.unique(labels)
    K = len(classes)

    # 3) Her client için boş küme hazırla
    dict_users: Dict[int, Set[int]] = {i: set() for i in range(num_users)}

    # 4) Her sınıf k için:
    for k in classes:
        # 4a) O sınıfa ait tüm örnek indekslerini al ve karıştır
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # 4b) Client’lara düşen oranları Dirichlet(α) ile çiz
        proportions = np.random.dirichlet(alpha * np.ones(num_users))

        # 4c) Her client’a kaç örnek düşeceğini sayıya çevir
        counts = (proportions * len(idx_k)).astype(int)

        # 4d) Yuvarlama kaynaklı kayıp/ekseneden dolayı toplam tutmuyorsa düzelt
        #    (eksik kalıyorsa en büyük orana, fazla kalıyorsa en küçük orana ekle/çıkar)
        diff = len(idx_k) - counts.sum()
        if diff > 0:
            counts[np.argmax(proportions)] += diff
        elif diff < 0:
            counts[np.argmin(proportions)] += diff  # diff negatif ⇒ çıkar

        # 4e) İndeksleri client kümelerine dağıt
        start = 0
        for i in range(num_users):
            c = counts[i]
            if c > 0:
                dict_users[i].update(idx_k[start:start+c])
            start += c


    return dict_users

def dirichlet_non_iid_unequal(
    dataset,
    num_users: int,
    alpha: float,
    beta: float,
    seed: int = None
) -> Dict[int, Set[int]]:
    """
    Two‐stage Dirichlet non‐IID partitioning with unequal sample sizes per client.

    Stage A: split total |D| across clients via Dirichlet(beta)
    Stage B: for each client i, split its n_i samples across classes via Dirichlet(alpha)

    :param dataset: torchvision Dataset (has .targets or .samples)
    :param num_users: number of clients
    :param alpha: Dirichlet concentration for class division (smaller ⇒ more skew)
    :param beta: Dirichlet concentration for total‐size division (smaller ⇒ larger size variance)
    :param seed: optional random seed for reproducibility
    :return: dict mapping client_id → set of sample indices
    """
    if seed is not None:
        np.random.seed(seed)

    # extract labels and total size
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([s[1] for s in dataset.samples])
    total_samples = len(labels)

    # Stage A: split total samples among clients
    psi = np.random.dirichlet(beta * np.ones(num_users))
    sizes = (psi * total_samples).astype(int)
    diff = total_samples - sizes.sum()
    if diff > 0:
        sizes[np.argmax(psi)] += diff
    elif diff < 0:
        sizes[np.argmin(psi)] += diff

    # prepare per‐class index pools
    classes = np.unique(labels)
    idx_by_class = {k: list(np.where(labels == k)[0]) for k in classes}

    # container for each client's indices
    user_data: Dict[int, Set[int]] = {i: set() for i in range(num_users)}

    # Stage B: for each client, split its share across classes
    for i in range(num_users):
        # draw class proportions for client i
        theta = np.random.dirichlet(alpha * np.ones(len(classes)))
        counts = (theta * sizes[i]).astype(int)
        diff_i = sizes[i] - counts.sum()
        if diff_i > 0:
            counts[np.argmax(theta)] += diff_i
        elif diff_i < 0:
            counts[np.argmin(theta)] += diff_i

        # assign samples
        for cls_idx, cls in enumerate(classes):
            take = counts[cls_idx]
            if take > 0:
                chosen = idx_by_class[cls][:take]
                user_data[i].update(chosen)
                # remove from pool
                idx_by_class[cls] = idx_by_class[cls][take:]

    return user_data




'''def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    #labels = dataset.train_labels.numpy()

    # eğer torchvision>=0.2 ise .targets kullanabiliyoruz
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        # fallback: .samples içinden 2. elemanı al
        labels = np.array([s[1] for s in dataset.samples])


    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users'''


'''if __name__ == '__main__':
    dataset_train = datasets.MNIST('data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)'''