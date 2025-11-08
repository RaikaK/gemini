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
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
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
    dir_name = "trained_models"
    save_dir = os.path.join(current_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
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

                total_loss += loss.item()

            # =============================
            batch_data = data_loader.get_train_batch_data()
        print(f"Epoch: {epoch} | TotalLoss: {total_loss}")
        wandb.log({"total_loss": total_loss}, step=epoch)
        if epoch % checkpoint_epoch == 0:
            save_torch_model(
                model=model,
                save_dir=save_dir,
                model_name=start_datetime + f"_epoch{epoch + 1}.pth",
            )
        data_loader.reset()

    wandb.finish()


def evaluate(model: torch.nn.Module, data_loader: DataLoader) -> dict:
    model.eval()
    test_data = data_loader.



if __name__ == "__main__":
    model = Dnn(input_size=DNN_INPUT_NUM, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    data_loader = DataLoader()
    training(
        model=model,
        data_loader=data_loader,
        epochs=int(1e6),
        optimizer=optimizer,
    )
