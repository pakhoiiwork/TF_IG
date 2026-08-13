import torch
from torch import nn
from torchinfo import summary

from neural_methods.models.baseline import BaselineModel
from neural_methods.models.baseline_transformer import BaselineTransformer
from neural_methods.models.baseline_2branch import Baseline2Branch
from neural_methods.models.base_i2branch import Improved2Branch

models = {
    # "base": BaselineModel,
    # "base_trans": BaselineTransformer,
    # "base_2branch": Baseline2Branch,
    "base_i2branch": Improved2Branch,
}

for model_name, model_class in models.items():
    print(f"Model: {model_name}")
    model = model_class()
    model_summary = summary(
        model, 
        input_size=(80, 2000),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        col_width=20
    )
    print(model_summary)
    print("=" * 80)
