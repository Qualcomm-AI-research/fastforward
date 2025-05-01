# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import torch


class AutoquantTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upcast = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1)
        self.l1 = AutoquantTestBlock(
            channels_in=12, channels_out=12, hidden_dim=24, activate_output=True
        )
        self.l2 = AutoquantTestBlock(
            channels_in=12, channels_out=12, hidden_dim=24, activate_output=False
        )
        self.downcast = torch.nn.Conv2d(
            in_channels=12, out_channels=4, kernel_size=4, stride=(4, 4)
        )
        self.classifier = AutoquantClassifier(4096, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expected input shape: (B, 3, 128, 128)."""
        h = self.upcast(x)  # Transform to 'compute shape'
        h = self.l1(h)
        h = self.l2(h)

        # 'Pooling' and downcast
        h = self.downcast(h)

        output = self.classifier(h)

        return output


class AutoquantTestBlock(torch.nn.Module):
    def __init__(
        self, channels_in: int, channels_out: int, hidden_dim: int, activate_output: bool
    ) -> None:
        super().__init__()
        self.l1 = torch.nn.Conv2d(channels_in, hidden_dim, kernel_size=3, padding=1)
        self.l2 = torch.nn.Conv2d(hidden_dim, channels_out, kernel_size=3, padding=1)
        self.activate_output = activate_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l1(x)
        h = torch.relu(h)
        h = self.l2(h)

        # Residual
        h = h + x

        # Conditional activation
        if self.activate_output:
            h = torch.sigmoid(h)

        return h


class AutoquantClassifier(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.l1 = torch.nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform from 3d to 1d representation.
        h = x.reshape((-1, self.num_features))
        h = self.l1(h)

        # Compute probabilities
        h = torch.softmax(h, 1)
        return h


def get_model() -> torch.nn.Module:
    return AutoquantTestModel()
