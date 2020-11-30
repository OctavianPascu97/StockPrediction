import argparse
import os
import os.path
import pdb
import sys
import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from tqdm import tqdm
import torchvision
from models import MODELS
import src.dataset

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])
  print(labels)
  return np.array(data), np.array(labels)



BATCH_SIZE = 8
DEVICE = "cuda"

ROOT = os.environ.get("ROOT", "")
def collate_fn(batches):
    return batches

def predict(args):
    Dataset = src.dataset.Stockpred_Dataset("")

    past_history = 30
    future_target = 9
    STEP = 1
    np.random.seed(3)

    TRAIN_SPLIT = np.int(len(Dataset) * 0.9)

    X_train, y_train = multivariate_data(Dataset, Dataset[:, 3], 0,
                                         TRAIN_SPLIT, past_history,
                                         future_target, STEP,
                                         single_step=True)
    X_test, y_test = multivariate_data(Dataset, Dataset[:, 3],
                                       TRAIN_SPLIT, None, past_history,
                                       future_target, STEP,
                                       single_step=True)

    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    batch_size = TRAIN_SPLIT
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )
    model = MODELS[args.model_type]()
    model_name = f"{args.model_type}"
    model_path = f"output/models/{model_name}.pth"
    model.load_state_dict(torch.load(model_path))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    mse = nn.L1Loss()

    test_loss = 0

    with torch.no_grad():
        for i, loader_data in enumerate(tqdm(loader)):
            pdb.set_trace()
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = mse(output, target)
            np.save(f"output/rez/{model_name}{i}.npy", output.cpu().numpy())
            test_loss += loss.item()

    test_loss = test_loss / len(loader)
    print(test_loss)


def main():
    parser = argparse.ArgumentParser(description="Test a given model")
    parser.add_argument("--model-type",
                        type=str,
                        required=True,
                        choices=MODELS,
                        help="which model type to use")
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()

