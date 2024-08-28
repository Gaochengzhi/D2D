import wandb
import random

wandb.init(
    project="TPAMI",
    config={
    "learning_rate": 0.02,
    "epochs": 1000,
    }
)

epochs = 100
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()


