import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import math
import time
import random
from contextlib import contextmanager
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import warnings

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchsummary import summary

from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

# Set seed for random generators
algorithm_globals.random_seed = 42

OUTPUT_DIR = "./"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

import torch


class CFG:
    print_freq = 100
    num_workers = 4
    model_name = "resnext50_32x4d"
    dim = (128, 281)
    scheduler = "CosineAnnealingWarmRestarts"
    epochs = 15
    #lr=1e-4
    lr = 0.0005
    T_0 = 10  # for CosineAnnealingWarmRestarts
    min_lr = 5e-7  # for CosineAnnealingWarmRestarts
    batch_size = 32
    weight_decay = 1e-6
    max_grad_norm = 1000
    seed = 42
    target_size = 2
    target_col = "hasbird"
    n_fold = 5
    pretrained = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train = pd.read_csv("../input/rich_metadata.csv")
train.loc[train["hasbird"] == 0, "filepath"] = (
    "../input/nocall/"
    + train.query("hasbird==0")["filename"]
    + ".npy"
)
train.loc[train["hasbird"] == 1, "filepath"] = (
    "../input/bird/"
    + train.query("hasbird==1")["filename"]
    + ".npy"
)

train = train.dropna().reset_index(drop=True)

folds = train.copy()
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)
print(folds.groupby(["fold", CFG.target_col]).size())

warnings.filterwarnings("ignore")


def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def init_logger(log_file=OUTPUT_DIR + "train.log"):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=CFG.seed)


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_paths = df["filepath"].values
        self.labels = df["hasbird"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_paths[idx]
        file_path = file_name
        image = np.load(file_path)
        image = image.transpose(1, 2, 0)
        image = np.squeeze(image)
        image = np.stack((image,) * 3, -1)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        label = torch.tensor(self.labels[idx]).long()
        return image, label


def get_transforms(*, data):
    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.dim[0], CFG.dim[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.augmentations.transforms.JpegCompression(p=0.5),
                A.augmentations.transforms.ImageCompression(
                    p=0.5, compression_type=A.augmentations.transforms.ImageCompression.ImageCompressionType.WEBP
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(CFG.dim[0], CFG.dim[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CustomResNext(nn.Module):
    def __init__(self, model_name="resnext50_32x4d", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


# we decompose the circuit for the QNN to avoid additional data copying
# Define and create QNN
nq = 1


def create_qnn():
    feature_map = ZFeatureMap(nq, reps=2)
    ansatz = RealAmplitudes(nq, reps=1)
    qc = QuantumCircuit(nq)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )

    return qnn


qnn = create_qnn()


class QuantumCustomResNext(nn.Module):
    def __init__(self, model_name="resnext50_32x4d", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.fc = nn.Linear(2048, nq)
        self.qnn = TorchConnector(qnn)
        # Remove fully connected layer and last two blocks
        self.model.fc = nn.Identity()
        # self.model.layer4 = nn.Identity()
        # self.model.layer3 = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        x = self.qnn(x)

        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(images)
        loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  ".format(
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    grad_norm=grad_norm,
                )
            )
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(1).to("cpu").numpy())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step + 1,
                    len(valid_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state["model"])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to("cpu").numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


def train_loop(train_folds, valid_folds):
    LOGGER.info(f"========== training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_dataset = TrainDataset(train_folds, transform=get_transforms(data="train"))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data="valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    def get_scheduler(optimizer):
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = QuantumCustomResNext(CFG.model_name, pretrained=True)
    # model = CustomResNext(CFG.model_name, pretrained=True)
    model.to(CFG.device)
    model.train()

    #optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=True)
    optimizer = Adam(model.parameters(), lr=CFG.lr)

    #scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    best_score = 0.0
    best_loss = np.inf

    scores = []

    for epoch in range(CFG.epochs):
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, optimizer, CFG.device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, CFG.device)
        valid_labels = valid_folds[CFG.target_col].values

        #scheduler.step()
        optimizer.step()
        
        # scoring
        score = get_score(valid_labels, preds.argmax(1))

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Accuracy: {score}")

        scores.append(score)

        if score > best_score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save({"model": model.state_dict(), "preds": preds}, OUTPUT_DIR + f"{CFG.model_name}_best.pth")
            
        # Pause the training for 60 sec
        time.sleep(120)

    check_point = torch.load(OUTPUT_DIR + f"{CFG.model_name}_best.pth")
    valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point["preds"]
    valid_folds["preds"] = check_point["preds"].argmax(1)

    return valid_folds, scores


def main(fold, num_iters=1):
    for i in range(num_iters):
        def get_result(result_df):
            preds = result_df["preds"].values
            labels = result_df[CFG.target_col].values
            score = get_score(labels, preds)
            LOGGER.info(f"Score: {score:<.5f}")

        def get_result2(result_df):
            preds = result_df["preds"].values
            labels = result_df[CFG.target_col].values
            matrix = get_confusion_matrix(labels, preds)
            print("TN", matrix[0, 0])
            print("FP", matrix[0, 1])
            print("FN", matrix[1, 0])
            print("TP", matrix[1, 1])

        # train
        training_runtime = []
        started = time.time()
        train_folds = folds.query(f"fold!={fold}").reset_index(drop=True)
        valid_folds = folds.query(f"fold=={fold}").reset_index(drop=False)
        oof_df, scores = train_loop(train_folds, valid_folds)
        ended = time.time()
        training_runtime.append(ended - started)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        get_result2(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR + "oof_df.csv", index=False)
        
        with open("training_hybrid_v2.csv", "a") as f:
            f.write(f"{fold},{i},{scores[-1]},{training_runtime[-1]}\n")
        
        time.sleep(120)
        
        """
        plt.plot([i for i in range(CFG.epochs)], scores)
        plt.title("valid score")
        plt.show()
        """


if __name__ == "__main__":
    main(0)