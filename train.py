
from transformers import AutoImageProcessor, DPTForDepthEstimation
from src.models import DPTMultiviewDepth
import wandb

import torch

from PIL import Image
import requests

from src.models import DPTMultiviewDepth

EPOCHES = 1000
LR = 1e-4

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="multiview-robustness-cs-503",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LR,
    "architecture": "DPT",
    "dataset": "",
    "epochs": EPOCHES,
    }
)

random_tensor = torch.rand((3, 384, 384))
dummy_image = Image.open('data/0.png')
dataloader_train = [[dummy_image]]

dummy_image_val = dummy_image
dataloader_val = [[dummy_image_val]]

# define the model
model = DPTMultiviewDepth.from_pretrained("Intel/dpt-large")
image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

step = 0
for _ in range(EPOCHES):
    for batch in dataloader_train:
        #load data
        inputs = image_processor(images=batch, return_tensors="pt")

        outputs = model(**inputs)
        predicted_depth = outputs["predicted_depth"]

        # calculate loss
        random_noise = torch.randn(predicted_depth.shape)
        loss_train = criterion(predicted_depth, random_tensor)

        # backpropagate loss
        loss_train.backward()
        optimizer.step()

        # log metrics
        wandb.log({"train loss": loss_train})

        #eval every 10 steps
        if step % 10 == 0:
            # load data
            inputs = image_processor(images=batch, return_tensors="pt")

            outputs = model(**inputs)
            predicted_depth = outputs["predicted_depth"]

            # calculate loss
            loss_val = criterion(predicted_depth, random_tensor)

            # log metrics
            wandb.log({"val loss": loss_val})

        step += 1
