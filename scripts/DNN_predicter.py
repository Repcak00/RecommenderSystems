import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from tqdm.autonotebook import tqdm
import pandas as pd

cwd = os.getcwd()
parent_directory = os.path.abspath(os.path.join(cwd, os.pardir))
data_directory = os.path.join(parent_directory, "data")
df_train = pd.read_json(
    os.path.join(data_directory, "limited_10_3_sweaters_reviews_sentiment_train.json"),
    orient="split",
)


class ResnetEmbeddings(nn.Module):
    def __init__(self):
        super(ResnetEmbeddings, self).__init__()
        resnet18 = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, img):
        return self.feature_extractor(img)


class PredictionModel(nn.Module):
    def __init__(
        self,
        n_users=1000,
        user_embedding_dim=100,
        picture_embedding_dim=512,
        linear_embedding_dim=300,
        output_pred_dim=5,
        df_train=df_train,
    ):
        super(PredictionModel, self).__init__()

        self.resnet = ResnetEmbeddings()
        self.user_embeddings = nn.Embedding(
            num_embeddings=n_users, embedding_dim=user_embedding_dim
        )
        self.df_train = df_train
        self.concat_dim = user_embedding_dim + picture_embedding_dim
        self.linear_1 = nn.Linear(
            in_features=picture_embedding_dim, out_features=picture_embedding_dim
        )
        self.linear_2 = nn.Linear(
            in_features=picture_embedding_dim, out_features=picture_embedding_dim
        )
        self.linear_3 = nn.Linear(
            in_features=self.concat_dim, out_features=linear_embedding_dim
        )
        self.linear_4 = nn.Linear(
            in_features=linear_embedding_dim, out_features=output_pred_dim
        )

    def forward(self, user_id, image):
        with torch.no_grad():
            image_embedding = self.resnet(image)
        user_embedding = self.user_embeddings(user_id)  # Get trainable user embeddings
        image_embedding = F.relu(
            self.linear_1(image_embedding.view(image_embedding.size(0), -1))
        )
        image_embedding = F.relu(self.linear_2(image_embedding))
        concatenated_embeddings = torch.cat(
            [user_embedding, image_embedding], dim=1
        )  # concat user and image embedding
        output = F.relu(self.linear_3(concatenated_embeddings))
        output = F.sigmoid(self.linear_4(output))
        return output


class TrainDataset(Dataset):
    def __init__(self, df, pictures_dir, transfrom, image_size):
        self.df = df
        self.pictures_dir = pictures_dir
        self.transform = transfrom
        self.image_size = image_size
        # Create a mapping from user names to indices
        self.user_to_index = {
            user: idx for idx, user in enumerate(df["user_id"].unique())
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # zmiana nazwa_usera -> index w df
        user_id = row.user_id
        user_index = self.user_to_index[user_id]

        dict_path = os.path.join(self.pictures_dir, row.item_id)
        images = os.listdir(dict_path)
        if images:
            image_path = os.path.join(dict_path, images[0])
            image = Image.open(image_path)
        else:
            image = Image.fromarray(
                np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            )
        image = self.transform(image)
        return (user_index, image), row.sentiment
        # return (row.user_id, image), row.sentiment

def count_correct(
    y_pred: torch.Tensor, y_batch: torch.Tensor, device
) -> torch.Tensor:
    out = (y_pred.to(device) > 0.5).float() * 1
    acc = (out.to(device) == y_batch).float().mean()
    return acc

def validate(
    model: nn.Module,
    loss_fn: torch.nn.BCELoss,
    dataloader: DataLoader,
    device
):
    correct = 0
    accuracy_array = []
    loss_array = []
    for X_batch, y_batch in tqdm(dataloader, desc="Testing", unit="Batch", colour="green", leave=False):
        (
            user_id_batch,
            image_batch,
        ) = X_batch
        user_id_batch, image_batch, y_batch = (
            user_id_batch.to(device),
            image_batch.to(device),
            y_batch.float().to(device),
        )
        y_pred = model(user_id_batch, image_batch).squeeze().float()
        loss_array.append(loss_fn(y_pred, y_batch).sum().detach().cpu())
        accuracy_array.append(count_correct(y_pred, y_batch, device).cpu())
    return np.array(loss_array).mean(), np.array(correct).mean()


def fit(model, train_dl, n_epochs=30, lr=0.001, test_dl=None):
    loss_fn = nn.BCELoss()  # maybe change to nn.CrossEntropy(
    optimizer = optim.Adam(model.parameters(), lr=lr)
    acc_train = []
    loss_train = []
    acc_test = []
    loss_test = []
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available!")
    else:
        device = torch.device("cpu")
        print("CUDA is not available! Using CPU")

    for epoch in tqdm(
        range(n_epochs),
        desc="Training epoch",
        unit="Epoch",
        total=n_epochs,
        colour="cyan",
    ):
        
        acc_batch = []
        loss_array = []
        for X_batch, y_batch in tqdm(
            iter(train_dl), desc="Training batch", unit="Batch", colour="magenta", leave=False
        ):
            model.train()
            # forward
            (
                user_id_batch,
                image_batch,
            ) = X_batch
            user_id_batch, image_batch, y_batch = (
                user_id_batch.to(device),
                image_batch.to(device),
                y_batch.float().to(device),
            )
            y_pred = model(user_id_batch, image_batch).squeeze().float()
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            out = (y_pred.to(device) > 0.5).float() * 1
            acc = (out.to(device) == y_batch).float().mean()
            acc_batch.append(acc.cpu())
            loss_array.append(loss.detach().cpu().numpy())
            
        if test_dl is not None:
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = validate(model, loss_fn, test_dl, device)
                print(f"After epoch mean test loss: {test_loss}")
                print(f"After epoch mean test acc: {test_acc}")
                acc_test.append(test_acc)
                loss_test.append(test_loss)
            
        print(f"After epoch mean loss: {np.array(loss_array).mean()}")
        print(f"After epoch acc: {np.array(acc_batch).mean()}")
        acc_train.append(np.array(acc_batch).mean())
        loss_train.append(np.array(loss_array).mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_train, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Over Epochs")
    plt.savefig(os.path.join(data_directory, f"training_loss_over_epoch_{n_epochs}.png"))
    plt.show()
    

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(acc_train, label="Training Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy Over Epochs")
    plt.savefig(os.path.join(data_directory, f"training_accuracy_over_epoch_{n_epochs}.png"))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_test, label="Testing Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Test Loss Over Epochs")
    plt.savefig(os.path.join(data_directory, f"testing_loss_over_epoch_{n_epochs}.png"))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(acc_test, label="Testing Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy Over Epochs")
    plt.savefig(os.path.join(data_directory, f"testing_accuracy_over_epoch_{n_epochs}.png"))
    plt.show()

    pickle.dump(
        model, open(os.path.join(data_directory, f"model_{n_epochs}_{lr}.pkl"), "wb")
    )

    return loss_train, acc_train
