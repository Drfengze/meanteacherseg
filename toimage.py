import numpy as np
import torch
from glob import glob
import torch.utils.data
from data_loader import data_load
import tensorflow as tf
from unet_pytorch import build_unet
import torch.nn as nn
import json
import xarray as xr
import rioxarray
from copy import deepcopy
import argparse
from affine import Affine


def to_image(img, mixer, raster_dir):
    f = open(*mixer)
    mixer = json.load(f)
    f.close()
    doubleMatrix = mixer["projection"]["affine"]["doubleMatrix"]
    patchesPerRow = mixer["totalPatches"] / mixer["patchesPerRow"]
    img = np.split(img, patchesPerRow)
    img = [np.hstack(l) for l in img]
    img = np.vstack(img).astype("int8")
    ds = xr.DataArray(img)
    img = []
    ds = ds.rio.write_crs(4326, inplace=True)
    transform = Affine(*doubleMatrix)
    ds.rio.write_transform(transform, inplace=True)
    ds.spatial_ref.GeoTransform
    ds.rio.set_spatial_dims("dim_1", "dim_0")
    ds.rio.to_raster(raster_dir)


def doPrediction(
    tf_files,
    model,
    bands,
    description,
    batch_size,
    device,
    x_buffer,
    y_buffer,
    kernel_shape,
):
    # print(tf_files)
    file_ls = np.append(
        np.arange(0, 70) * int(len(list(tf_files)) / 70), len(list(tf_files))
    )
    patches = 0
    np_ls = []
    _ = 1
    for i in range(len(file_ls) - 1):
        print(i)
        data = (
            data_load(
                tf_files[file_ls[i] : file_ls[i + 1]],
                bands,
                description,
            )
            .get_pridiction_dataset()
            .batch(batch_size)
        )

        ls = []
        for batch in data.as_numpy_iterator():
            # print(batch.shape)
            predictions = torch.tensor(batch).to(device)
            with torch.no_grad():
                predictions = model(predictions).cpu().numpy()  # .tolist()
            # print(predictions.shape)
            # print(predictions)
            predictions = predictions.argmax(axis=1).squeeze()[
                :,
                x_buffer : x_buffer + kernel_shape[0],
                y_buffer : y_buffer + kernel_shape[1],
            ]
            print("Writing patch " + str(_ * 64) + "...", flush=True)
            _ = _ + 1
            ls.append(predictions)
        np_ls.append(np.vstack(ls).astype("int8"))
    return np_ls
    print("finished")


def main(args):
    bands = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "nirv"]
    targets = ["cropland"]
    features = bands  # + targets
    nclass = 2
    batch_size = 256
    class_names = ["Others", "Wheat"]
    kernel_shape = [256, 256]
    kernel_buffer = [128, 128]
    # Get set up for prediction.
    x_buffer = int(kernel_buffer[0] / 2)
    y_buffer = int(kernel_buffer[1] / 2)

    buffered_shape = [
        kernel_shape[0] + kernel_buffer[0],
        kernel_shape[1] + kernel_buffer[1],
    ]
    columns = [
        tf.io.FixedLenFeature(shape=buffered_shape, dtype=tf.float32) for k in features
    ]
    description = dict(zip(features, columns))
    device = "cuda"
    folder = "prediction"
    model = (
        build_unet(len(bands), nclass).cuda()
        if device == "cuda"
        else build_unet(len(bands), nclass)
    )

    # Load the state_dict
    loaded_state_dict = torch.load(args.ckpt_path)

    # Try to remove the "module." prefix from the keys and load the state_dict
    try:
        loaded_state_dict = {
            k.replace("module.", ""): v for k, v in loaded_state_dict.items()
        }
        model.load_state_dict(loaded_state_dict)
    except Exception as e:
        print(f"Error loading state_dict: {e}")

    # Wrap the model with DataParallel
    model = nn.DataParallel(model)
    model.eval()
    folder = args.pred_folder
    tf_files = glob(folder + "/*.gz")
    tf_files.sort()
    json_file = glob(folder + "/*.json")
    out_image_file = args.tif_filename
    prediction = doPrediction(
        tf_files,
        model,
        bands,
        description,
        batch_size,
        device,
        x_buffer,
        y_buffer,
        kernel_shape,
    )
    img = np.vstack(prediction)
    # np.save("/content/drive/MyDrive/RDA/geotiff_result/{}.npy".format(state), img)
    to_image(img, json_file, out_image_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pred_folder", type=str, required=True)
    parser.add_argument("--tif_filename", type=str, required=True)
    args = parser.parse_args()
    main(args)
