
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, List, Union

from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from torch import Tensor as T

from InnerEye.SSL.utils import SSLModule
from InnerEye.SSL.byol.byol_models import SSLEncoder

SingleBatchType = Tuple[List, T]
BatchType = Union[Dict[SSLModule, SingleBatchType], SingleBatchType]

class _Projection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLRInnerEye(SimCLR):
    def __init__(self, encoder_name: str, dataset_name: str, **kwargs: Any) -> None:
        """
        Returns SimCLR pytorch-lightning module, based on lightning-bolts implementation.
        :param encoder_name [str] Image encoder name (predefined models)
        :param dataset_name [str] Image dataset name (e.g. cifar10, kaggle, etc.)
        """
        if "dataset" not in kwargs:  # needed for the new version of lightning-bolts
            kwargs.update({"dataset": dataset_name})
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.encoder = SSLEncoder(encoder_name, dataset_name)
        self.projection = _Projection(input_dim=self.encoder.get_output_feature_dim(), hidden_dim=2048, output_dim=128)

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch: BatchType) -> T:
        batch = batch[SSLModule.ENCODER] if isinstance(batch, dict) else batch

        (img1, img2), y = batch

        # get h representations, bolts resnet returns a list
        h1, h2 = self(img1), self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        return loss