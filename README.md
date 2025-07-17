# Federated Learning

## Usage

Before running, make sure to go inside the `src` folder: `cd src`

There are two files inside the project: `main.py` for the (centralized) machine learning and `federated_main.py` for the federated learning.

### Parameters

| Parameter    | Case               | Description                                                                                                                       | Example                                          |
|--------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| model        | All                | Defines the model to be used when running the program.<br>Available options: cnn, resnet18, resnet50, mobilenet_v2, shufflenet_v2 | --model mobilenet_v2                             |
| dataset_path | All                | Defines the dataset path to be used by the program.                                                                               | --dataset_path /teamspace/uploads/malimg_dataset |
| num_channels | All                | Defines the number of channels to be used by the model.<br>Example: 1 for grayscale and 3 for RGB.                                | --num_channels 1                                 |
| dataset      | All                | Defines the name of the used dataset.                                                                                             | --dataset mallimg                                |
| gpu          | All (Optional)     | The program uses CPU by default. Add this option to use GPU (use 0-based indexing for device).                                                                      | --gpu 0                                          |
| epoch        | All                | Defines the amount of epoch (or communication round) to be executed.                                                              | --epoch 20                                       |
| local_ep     | Federated Learning | Defines the amount of local epoch to be executed.                                                                                 | --local_ep 1                                     |
| num_users    | Federated Learning | Defines the amount of clients (users) for the training.                                                                           | --num_users 50                                   |
| iid          | Federated Learning | Defines the usage of IID in the data. Use 0 or 1 to define the value.                                                             | --iid 0                                          |
| unequal      | Federated Learning | Defines the split of the data (balanced/imbalanced). Use 0 or 1 to define the value.                                              | --unequal 1                                      |

To explore the full available parameters, please refer to the `options.py` file.

### Examples

#### (Centralized) Machine Learning

```bash
python main.py --model mobilenet_v2 --dataset_path /teamspace/studios/this_studio/malnet_resized_32x256 --num_channels 1 --dataset malimg --gpu 0 --epoch 20
```

#### Federated Learning

```bash
python federated_main.py --model mobilenet_v2 --dataset_path /teamspace/studios/this_studio/malnet_resized_32x256 --num_channels 1 --dataset malimg --gpu 0 --epoch 20 --local_ep 3 --num_users 50 --iid 1
```
