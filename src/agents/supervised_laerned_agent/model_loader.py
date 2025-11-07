import torch
import os


def save_torch_model(model: torch.nn.Module, save_dir: str, model_name: str):
    torch.save(
        model,
        os.path.join(save_dir, model_name),
    )


def load_torch_model(model_path: str) -> torch.nn.Module:
    model = torch.load(model_path, weights_only=False)
    return model
