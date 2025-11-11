import torch
import numpy as np

import wandb
import datetime
import os

from src.agents.supervised_laerned_agent.data_loader import DataLoader, DNN_INPUT_NUM
from src.agents.supervised_laerned_agent.model_loader import save_torch_model
from src.common.sample_mlp_model import Dnn


def training(
    model: torch.nn.Module,
    data_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.functional = torch.nn.CrossEntropyLoss(),
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    checkpoint_epoch: int = 100,
):
    """
    Args:
        - model (torch.nn.Module): 学習させるモデル
        - data_loader (DataLoader): 学習データローダー
        - epochs (int): エポック数
        - optimizer: ただし、model.parameters()がセットされていること
        - loss_fn: 損失関数 | default: CrossEntropyLoss
        - device: 学習に使用するデバイス default: cuda
    """
    wandb.init(entity="ygo-ai", project="U-Ni-Yo", group="SupervisedLearning")
    print(f"Training on device: {device}")
    model.to(device)
    model.train()

    # モデルの保存先を決定
    start_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = f"trained_models/{start_datetime}"
    save_dir = os.path.join(current_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        losses = []
        batch_data = data_loader.get_train_batch_data()
        while batch_data is not None:
            # === impl here ===============
            # note: now... -> unable to batch learning
            (X_batch, y_batch, hash_batch) = batch_data
            for X, y, hash in zip(X_batch, y_batch, hash_batch):
                input_tensor = torch.tensor(X).to(device)
                label_tensor = torch.tensor(y).to(device)

                optimizer.zero_grad()
                output = torch.softmax(model(input_tensor).reshape((hash,)), dim=0)
                # breakpoint()
                loss = loss_fn(output, label_tensor)
                loss.backward()
                optimizer.step()
                # breakpoint()  # lossの値を確認 mean()するかしないか

                losses.append(loss.item())

            # =============================
            batch_data = data_loader.get_train_batch_data()
        print(f"Epoch: {epoch} | AveLoss: {np.mean(losses)}")
        wandb.log(
            {"average_loss": np.mean(losses) if len(losses) > 0 else 0}, step=epoch
        )
        if epoch % checkpoint_epoch == 0:
            top_k_accuracy_dict = evaluate(
                model=model, data_loader=data_loader, device=device
            )
            save_torch_model(
                model=model,
                save_dir=save_dir,
                model_name=f"epoch{epoch + 1}.pth",
            )
            wandb.log(top_k_accuracy_dict, step=epoch)
        data_loader.reset()

    wandb.finish()


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.functional = torch.nn.CrossEntropyLoss(),
) -> dict:
    model.eval()
    X, y = data_loader.get_test_data()
    top_ks = [1, 2, 3]
    top_k_accuracy_dict = {k: 0 for k in top_ks}
    # top_kの値よりも長いラベル数の時だけtop_k accuracyを計算する
    total_samples = {k: len([label for label in y if len(label) > k]) for k in top_ks}
    losses = []
    for x, label in zip(X, y):
        input_tensor = torch.tensor(x).to(device)
        label_tensor = torch.tensor(label).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor).reshape((len(label),)), dim=0)
            for top_k in top_ks:
                # top_kの値よりも長いラベル数の時だけtop_k accuracyを計算する
                if len(probs) <= top_k:
                    continue
                topk_indices = torch.topk(probs, k=top_k).indices.cpu().numpy()
                if label_tensor[topk_indices].any() == 1.0:
                    top_k_accuracy_dict[top_k] += 1

            loss = loss_fn(probs, label_tensor)
            losses.append(loss.item())

    top_k_accuracy_dict = {
        f"top_{k}_accuracy": v / total_samples[k] if total_samples[k] > 0 else 0
        for k, v in top_k_accuracy_dict.items()
    }
    loss_dict = {"evaluation_loss": np.mean(losses) if len(losses) > 0 else 0}
    print(top_k_accuracy_dict)
    return top_k_accuracy_dict | loss_dict


if __name__ == "__main__":
    model = Dnn(input_size=DNN_INPUT_NUM, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    data_loader = DataLoader()
    training(
        model=model,
        data_loader=data_loader,
        epochs=int(1e5),
        optimizer=optimizer,
        checkpoint_epoch=100,
    )
