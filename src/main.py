
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import os

from utils import get_dataset
from options import args_parser
from update import test_inference
from model import CNN


if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    global_model = CNN(args=args)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    # Define the path for the log file
    log_file_path = 'save/main_training_log.txt'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Open the log file in write mode
    with open(log_file_path, 'w') as log_file:

        # Training loop
        for epoch in tqdm(range(args.epochs)):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    log_message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item())
                    print(log_message)
                    log_file.write(log_message + '\n')
                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)
            log_file.write('\nTrain loss: {}\n'.format(loss_avg))
            epoch_loss.append(loss_avg)

        # Plot loss
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')
        plt.savefig('save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                    args.epochs))

        # testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print("*"*32)
        print("*"*10, "Test Results", "*"*10)
        print('Test on', len(test_dataset), 'samples')
        print("Test Accuracy: {:.2f}%".format(100*test_acc))
        log_file.write("*"*32 + '\n')
        log_file.write("*"*10 + "Test Results" + "*"*10 + '\n')
        log_file.write('Test on {} samples\n'.format(len(test_dataset)))
        log_file.write("Test Accuracy: {:.2f}%\n".format(100*test_acc))

    # Save the model
    model_path = 'save/models/main_model_{}_{}_{}.pth'.format(
    args.dataset, args.model, args.epochs)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model
    torch.save(global_model.state_dict(), model_path)

    print(f'Model successfully saved to {model_path}')

    # load the model
    # model_path = 'save/models/main_model_{}_{}_{}.pth'.format(
    #     args.dataset, args.model, args.epochs)

    # global_model.load_state_dict(torch.load(model_path))
    # global_model.to(device)
    # print(f'Model loaded from {model_path}')