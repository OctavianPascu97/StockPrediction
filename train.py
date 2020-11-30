import argparse
import os
import os.path
import pdb
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
import ignite.engine as engine
import ignite.handlers
import torchvision

import src.dataset

from models import MODELS

ROOT = os.environ.get("ROOT", "")

SEED = 1337
MAX_EPOCHS = 10000
PATIENCE = 20
LR_REDUCE_PARAMS = {
    "factor": 0.2,
    "patience": 4,
}

def collate_fn(batches):
    return batches


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a given model")
    parser.add_argument("--model-type",
                            type=str,
                            required=True,
                            choices=MODELS,
                            help="which model type to train")
    parser.add_argument("-model",
                        type=str,
                        required=False,
                        choices=MODELS,
                        help="which model to load")
    parser.add_argument("-v",
                            "--verbose",
                            action="count",
                            help="verbosity level")
    args = parser.parse_args()

    model = MODELS[args.model_type]()
    if args.model is not None:
        model_path = args.model
        model_name = os.path.basename(args.model)
        model.load(model_path)
    else:
        model_name = f"{args.model_type}"
        model_path = f"output/models/{model_name}.pth"

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

    train_dataset = torch.utils.data.TensorDataset(X_train,y_train)
    valid_dataset = torch.utils.data.TensorDataset(X_test,y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False)
    train_loss = []
    valid_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss = nn.L1Loss()
    device = 'cuda'

    trainer = engine.create_supervised_trainer(model, optimizer, loss, device=device)

    evaluator = engine.create_supervised_evaluator(
        model,
        metrics={'loss': ignite.metrics.Loss(loss)},
        device=device,
    )

    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch {:3d} Train loss: {:8.6f}".format(trainer.state.epoch,
                                                       trainer.state.output))


    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_validation_loss(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print("Epoch {:3d} Valid loss: {:8.6f} ←".format(
            trainer.state.epoch, metrics['loss']))
        valid_loss.append(metrics['loss'])

    lr_reduce = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               verbose=args.verbose,
                                               **LR_REDUCE_PARAMS)

    @evaluator.on(engine.Events.COMPLETED)
    def update_lr_reduce(engine):
        loss = engine.state.metrics['loss']
        lr_reduce.step(loss)

    def score_function(engine):
        return -engine.state.metrics['loss']

    early_stopping_handler = ignite.handlers.EarlyStopping(
        patience=PATIENCE, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED,
                                early_stopping_handler)

    checkpoint_handler = ignite.handlers.ModelCheckpoint(
        "output/models/checkpoints",
        model_name,
        score_function=score_function,
        n_saved=1,
        require_empty=False,
        create_dir=True)

    evaluator.add_event_handler(engine.Events.EPOCH_COMPLETED,
                                checkpoint_handler, {"model": model})

    trainer.run(train_loader, max_epochs=MAX_EPOCHS)
    torch.save(model.state_dict(), model_path)
    print("Model saved at:", model_path)
    np.save('train_loss', np.asarray(train_loss))
    np.save('valid_loss', np.asarray(valid_loss))


if __name__ == "__main__":
    main()
