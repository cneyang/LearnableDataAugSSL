import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from tqdm import tqdm
import pandas as pd
from datetime import datetime

from models import Encoder, Classifier, Policy
from diffaug import Augmenter
import dataset


def train_val(model, augmenter, dataloader, optimizer=None):
    data_bar = tqdm(dataloader)
    criterion = F.cross_entropy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    total_num = 0
    if optimizer is not None:
        total_clf_loss, total_d_loss, total_adv_loss, total_policy_clf_loss = 0, 0, 0, 0
        model["encoder"].train(), model["classifier"].train(), model["policy"].train()
        augmenter.train()
        for step, (x, y) in enumerate(data_bar):
            b = x.size(0) // 2
            x, y = x.to(device), y.to(device)

            a_x, a_y = x[:b], y[:b]
            n_x, n_y = normalize(x[b:]), y[b:]

            # train encoder & critic
            ones = n_x.new_tensor(1.0)
            model["encoder"].requires_grad_(True)
            model["classifier"].requires_grad_(True)
            optimizer["main"].zero_grad()

            ## real images
            out, n_out = model["classifier"](model["encoder"](n_x))
            loss = criterion(out, n_y)
            loss.backward(retain_graph=True)
            real_loss = F.mse_loss(n_out.sigmoid(), torch.zeros_like(n_out))
            real_loss.backward()

            ## augmented images
            with torch.no_grad():
                mag = model["policy"](a_x)
                augmented = augmenter(a_x, mag)

            out, a_out = model["classifier"](model["encoder"](augmented))
            _loss = criterion(out, a_y)
            _loss.backward(retain_graph=True)
            fake_loss = F.mse_loss(a_out.sigmoid(), mag.detach().sigmoid())
            fake_loss.backward()

            # train policy
            if step % args.d_steps == 0:
                model["encoder"].requires_grad_(False)
                model["classifier"].requires_grad_(False)
                optimizer["policy"].zero_grad()

                mag = model["policy"](a_x)
                augmented = augmenter(a_x, mag)
                _out, a_out = model["classifier"](model["encoder"](augmented))
                policy_loss = criterion(_out, a_y)
                policy_loss.backward(retain_graph=True)
                adv_loss = F.mse_loss(mag.sigmoid(), a_out.clone().detach().sigmoid())
                adv_loss.backward(-ones)
                optimizer["policy"].step()

            total_num += b
            total_clf_loss += (loss.item() + _loss.item()) * b
            total_d_loss += (real_loss.item() + fake_loss.item()) * b
            total_adv_loss += adv_loss.item() * b
            total_policy_clf_loss += policy_loss.item() * b
            data_bar.set_description(
                "Epoch: [{}/{}] CLF Loss: {:.4f} D Loss: {:.4f} ADV Loss {:.4f}".format(
                    epoch,
                    args.epochs,
                    total_clf_loss / total_num,
                    total_d_loss / total_num,
                    total_adv_loss / total_num,
                )
            )
        return (
            total_clf_loss / total_num,
            total_d_loss / total_num,
            total_adv_loss / total_num,
            total_policy_clf_loss / total_num,
        )
    else:
        n_acc, a_acc = 0, 0
        with torch.no_grad():
            model["encoder"].eval(), model["classifier"].eval(), model["policy"].eval()
            augmenter.eval()
            for x, y in data_bar:
                b = x.size(0) // 2
                x, y = x.to(device), y.to(device)

                a_x, a_y = x[:b], y[:b]
                n_x, n_y = normalize(x[b:]), y[b:]

                ## real images
                n_out, _ = model["classifier"](model["encoder"](n_x))

                ## augmented images
                mag = model["policy"](a_x)
                augmented = augmenter(a_x, mag)
                a_out, _ = model["classifier"](model["encoder"](augmented))

                total_num += b
                n_acc += (n_out.argmax(dim=1) == n_y).float().sum().item()
                a_acc += (a_out.argmax(dim=1) == a_y).float().sum().item()
                data_bar.set_description(
                    "Epoch: [{}/{}] Real Acc: {:.4f} Aug Acc: {:.4f}".format(
                        epoch, args.epochs, n_acc / total_num, a_acc / total_num
                    )
                )
        return n_acc / total_num, a_acc / total_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--d_steps", type=int, default=1)
    args = parser.parse_args()

    # config
    model_name = "batch{}_lr{}_dstep{}_{}".format(
        args.batch_size, args.lr, args.d_steps, datetime.now().strftime("%m%d%H%M%S"),
    )
    print("Model Name: {}".format(model_name))

    result_path = "./results/pretext/" + model_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    writer = SummaryWriter("runs/search/pretext/" + model_name)

    # load data
    train_loader, valid_loader = dataset.create_loaders(args.batch_size)

    augmenter = Augmenter(
        after_operations=transforms.Compose(
            [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
        ),
    ).cuda()

    encoder = Encoder().cuda()
    classifier = Classifier(
        in_features=encoder.out_features,
        num_classes=10,
        num_ops=len(augmenter.operations),
    ).cuda()
    policy = Policy(encoder=encoder, num_ops=len(augmenter.operations)).cuda()

    model = {"encoder": encoder, "classifier": classifier, "policy": policy}

    optimizer = {
        "main": optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=args.lr,
            betas=(0, 0.999),
        ),
        "policy": optim.Adam(policy.parameters(), lr=args.lr, betas=(0, 0.999)),
    }

    results = {
        "train_clf_loss": [],
        "train_d_loss": [],
        "train_adv_loss": [],
        "train_policy_clf_loss": [],
        "real_acc": [],
        "aug_acc": [],
    }
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_clf_loss, train_d_loss, train_adv_loss, train_policy_clf_loss = train_val(
            model, augmenter, train_loader, optimizer
        )
        real_acc, aug_acc = train_val(model, augmenter, valid_loader)

        writer.add_scalar("train_clf_loss", train_clf_loss, epoch)
        writer.add_scalar("train_d_loss", train_d_loss, epoch)
        writer.add_scalar("train_adv_loss", train_adv_loss, epoch)
        writer.add_scalar("train_policy_clf_loss", train_policy_clf_loss, epoch)
        writer.add_scalar("real_acc", real_acc, epoch)
        writer.add_scalar("aug_acc", aug_acc, epoch)
        results["train_clf_loss"].append(train_clf_loss)
        results["train_d_loss"].append(train_d_loss)
        results["train_adv_loss"].append(train_adv_loss)
        results["train_policy_clf_loss"].append(train_policy_clf_loss)
        results["real_acc"].append(real_acc)
        results["aug_acc"].append(aug_acc)

        df = pd.DataFrame(data=results, index=range(1, epoch + 1))
        df.to_csv(result_path + f"/search_statistics.csv", index_label="epoch")

        if epoch % 20 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "policy": policy.state_dict(),
                    "augmenter": augmenter.state_dict(),
                },
                result_path + f"/epoch{epoch}_model.pth",
            )

        if aug_acc > best_acc:
            best_acc = aug_acc
            torch.save(
                {
                    "epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "policy": policy.state_dict(),
                    "augmenter": augmenter.state_dict(),
                },
                result_path + f"/best_model.pth",
            )

