import torch
import torch.nn as nn
import numpy as np
from data_loader import data_load
from unet_pytorch import build_unet
from ramps import get_current_consistency_weight, update_ema_variables
from glob import glob
import tensorflow as tf
from time import time
from datetime import datetime
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    AsDiscrete,
    EnsureType,
)
import os


def load_dataset(batch, folder, label=True, buffer_size=1000):
    features = BANDS + TARGETS if label else BANDS
    tf_files = glob(f"{folder}/*.gz")
    columns = [
        tf.io.FixedLenFeature(
            shape=KERNEL_SHAPE if label else BUFFERED_SHAPE, dtype=tf.float32
        )
        for _feature in features
    ]
    description = dict(zip(features, columns))
    data_func = data_load(
        tf_files,
        BANDS,
        description,
        response=TARGETS,
        batch_size=batch,
        buffer_size=buffer_size,
    )
    data = (
        data_func.get_training_dataset()
        if label
        else data_func.get_pridiction_dataset()
    )
    # print(tf_files)
    return data


BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "nirv"]
KERNEL_SHAPE = [256, 256]
KERNEL_BUFFER = [128, 128]
X_BUFFER, Y_BUFFER = [buffer // 2 for buffer in KERNEL_BUFFER]
X_BUFFERED, Y_BUFFERED = (X_BUFFER + KERNEL_SHAPE[0]), (Y_BUFFER + KERNEL_SHAPE[1])
BUFFERED_SHAPE = [
    kernel + buffer for kernel, buffer in zip(KERNEL_SHAPE, KERNEL_BUFFER)
]
TARGETS = ["cropland"]
NCLASS = 2
model_folder = f"/bess23/huaize/semi-supervised/models/"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
run_time = datetime.today().strftime("%m_%d_%H_%M_%S")
batch = 128


dataset = load_dataset(
    batch, "/bess23/huaize/semi-supervised/data/labeled/train/", label=True
)
val_dataset = load_dataset(
    batch, "/bess23/huaize/semi-supervised/data/labeled/valid", label=True
)
test_dataset = load_dataset(
    batch, "/bess23/huaize/semi-supervised/data/unlabeled", label=False
)

model = build_unet(len(BANDS), NCLASS).cuda()
ema_model = build_unet(len(BANDS), NCLASS).cuda()
model = nn.DataParallel(model)
ema_model = nn.DataParallel(ema_model)
model.to(device)
ema_model.to(device)

max_epochs = 1000
MeanTeacherEpoch = 50
lr = 3e-4
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# %% train
max_epochs = 1000
MeanTeacherEpoch = 50
val_interval = 1
best_metric = -1
best_metric_epoch = -1
iter_num = 0
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=NCLASS)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=NCLASS)])

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    start_time = time()
    model.train()
    epoch_loss = 0
    step = 0
    train_loader = dataset.as_numpy_iterator()
    val_loader = val_dataset.as_numpy_iterator()
    unlabeled_train_loader = test_dataset.as_numpy_iterator()
    for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_train_loader):
        step += 1
        labeled_inputs, labels = (
            torch.tensor(labeled_batch[0]).to(device),
            torch.tensor(labeled_batch[1]).to(device),
        )
        unlabeled_batch = unlabeled_batch[
            slice(None), slice(None), X_BUFFER:X_BUFFERED, Y_BUFFER:Y_BUFFERED
        ]
        unlabeled_inputs = torch.tensor(unlabeled_batch).to(device)
        opt.zero_grad()
        noise_labeled = torch.clamp(torch.randn_like(labeled_inputs) * 0.1, -0.2, 0.2)
        noise_unlabeled = torch.clamp(
            torch.randn_like(unlabeled_inputs) * 0.1, -0.2, 0.2
        )
        noise_labeled_inputs = labeled_inputs + noise_labeled
        noise_unlabeled_inputs = unlabeled_inputs + noise_unlabeled

        outputs = model(labeled_inputs)
        with torch.no_grad():
            soft_out = torch.softmax(outputs, dim=1)
            outputs_unlabeled = model(unlabeled_inputs)
            soft_unlabeled = torch.softmax(outputs_unlabeled, dim=1)
            outputs_aug = ema_model(noise_labeled_inputs)
            soft_aug = torch.softmax(outputs_aug, dim=1)
            outputs_unlabeled_aug = ema_model(noise_unlabeled_inputs)
            soft_unlabeled_aug = torch.softmax(outputs_unlabeled_aug, dim=1)

        supervised_loss = loss_function(outputs, labels)
        if epoch < MeanTeacherEpoch:
            consistency_loss = 0.0
        else:
            consistency_loss = torch.mean((soft_out - soft_aug) ** 2) + torch.mean(
                (soft_unlabeled - soft_unlabeled_aug) ** 2
            )
        consistency_weight = get_current_consistency_weight(iter_num // 150)
        iter_num += 1
        loss = supervised_loss + consistency_weight * consistency_loss
        loss.backward()
        opt.step()
        update_ema_variables(model, ema_model, 0.99, iter_num)
        epoch_loss += loss.item()
        print(
            # f"{step}/{len(unlabeled_train_ds) // unlabeled_train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}"
        )
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    torch.tensor(val_data[0]).to(device),
                    torch.tensor(val_data[1]).to(device),
                )
                val_outputs = model(val_inputs)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            print(f"val dice: {metric}")
            # reset the status for next validation round
            dice_metric.reset()

        metric_values.append(metric)
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(
                model.module.state_dict(),
                os.path.join(model_folder, f"best_{run_time}.pth"),
            )
            print("saved new best metric model")
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f"\nbest mean dice: {best_metric:.4f} "
            f"at epoch: {best_metric_epoch}"
        )
    print(f"epoch time = {time() - start_time}")