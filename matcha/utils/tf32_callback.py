import torch
from lightning import Callback


class TF32Callback(Callback):
    """Enable TensorFloat-32 matrix multiplications on Ampere/Hopper GPUs."""

    def on_train_start(self, trainer, pl_module):  # pylint: disable=unused-argument
        torch.set_float32_matmul_precision("high")
