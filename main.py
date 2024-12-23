from libauc.losses import AUCMLoss
from libauc.utils import ImbalancedDataGenerator
from libauc.sampler import DualSampler
from libauc.metrics import auc_roc_score

import torch 
import numpy as np
from optimizer import PrivateDiff
from torchvision.datasets import MNIST
import math
import argparse
from utils import weights_init, set_all_seeds, ImageDataset, NeuralNetwork



def train(args):
    epsilon, delta = args.epsilon, args.delta
    c1, c2, cy = args.c1, args.c2, args.cy
    lr, lr_alpha = args.lr, args.lr_alpha
    t, t2 = args.T, args.T2
    imratio = args.imratio
    seed = args.seed
    batch_size = args.batch_size
    total_epochs = args.total_epochs
    set_all_seeds(seed)

    # NOTE: load data as numpy arrays
    d = MNIST(root="./data", train=True, download=True)
    train_data, train_targets = d.data, d.targets
    d = MNIST(root="./data", train=False)
    test_data, test_targets = d.data, d.targets

    # NOTE: get dataset size
    if imratio == 0.5: n=50000
    else: n=33995

    # NOTE: generate imbalanced data
    generator = ImbalancedDataGenerator(verbose=True, random_seed=0)
    (train_images, train_labels) = generator.transform(train_data, train_targets, imratio=imratio)
    (test_images, test_labels) = generator.transform(test_data, test_targets, imratio=0.5)

    # NOTE: data augmentations
    trainSet = ImageDataset(train_images, train_labels)
    testSet = ImageDataset(test_images, test_labels, mode="test")

    # NOTE: dataloaders
    sampler = DualSampler(trainSet, batch_size, sampling_rate=imratio)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, sampler=sampler, num_workers=2)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=2)

    # NOTE: network
    model = NeuralNetwork()
    weights_init(model)
    model = model.cuda()

    # NOTE: optimizer
    loss_fn = AUCMLoss(imratio=imratio)
    sigma_1 = math.sqrt(total_epochs / t * math.log(1 / delta)) / n / epsilon
    sigma_2 = math.sqrt(math.log(1 / delta)) / n / epsilon
    sigma_alpha = math.sqrt(math.log(1 / delta)) / n / epsilon
    optimizer = PrivateDiff(
        model.parameters(),
        loss_fn=loss_fn,
        lr = lr,
        lr_alpha=lr_alpha,
        c1=c1,
        c2=c2,
        c_y=cy,
        sigma1=sigma_1,
        sigma2=sigma_2,
        sigma_alpha=sigma_alpha,
        inner_iters=t2,
        T = t,
    )

    test_log = []
    r = 0
    for epoch in range(total_epochs):

        model.train()
        for i, (data, targets) in enumerate(trainloader):
            data, targets = data.cuda(), targets.cuda()

            def closure():
                loss = loss_fn(torch.sigmoid(model(data)), targets)
                loss.backward()
                return loss

            optimizer.zero_grad()
            loss = loss_fn(torch.sigmoid(model(data)), targets)
            loss.backward()
            optimizer.step(r=r, closure=closure)
            optimizer.zero_grad()
            r += 1

        # NOTE: evaluation on train & test sets
        model.eval()

        test_pred_list = []
        test_true_list = []
        for test_data, test_targets in testloader:
            test_data = test_data.cuda()
            test_pred = model(test_data)
            test_pred_list.append(test_pred.cpu().detach().numpy())
            test_true_list.append(test_targets.numpy())
        test_true = np.concatenate(test_true_list)
        test_pred = np.concatenate(test_pred_list)
        val_auc = auc_roc_score(test_true, test_pred)
        model.train()

        # NOTE: print results
        print("epoch: %s, test_auc: %.4f, lr: %.4f"% (epoch, val_auc, optimizer.lr))
        test_log.append(val_auc)

    print("best auc: %.4f" % max(np.array(test_log)))


parser = argparse.ArgumentParser(description='auc maximization.')
parser.add_argument('--epsilon', default=0.5, type=float, help='Param of differential privacy')
parser.add_argument("--c1", default=1, type=float, help="value of gradient clip")
parser.add_argument("--c2", default=1, type=float, help="value of gradient clip")
parser.add_argument("--cy", default=1, type=float, help="value of gradient clip")
parser.add_argument("--lr", default=0.2, type=float, help="scale the lr")
parser.add_argument("--lr_alpha", default=0.2, type=float, help="scale the lr")
parser.add_argument("--T2", default="3", type=int, help="T2")
parser.add_argument("--T", default="2", type=int, help="T")
parser.add_argument("--imratio", default=0.1, type=float, help="imratio")
parser.add_argument("--batch_size", default=2048, type=float, help="batch size")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument("--delta", default=1e-4, type=float, help="delta")
parser.add_argument("--total_epochs", default=80, type=int, help="epoch")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    train(args)
