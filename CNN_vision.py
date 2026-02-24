# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="RsBQYwRlGw1c"
# # Implementación de una *Convolutional Neural Net, CNN*

# + [markdown] id="2-gVQ_pdH_Sr"
# **Diego Alberto Barriga Martínez**

# + [markdown] id="gSNMi2G3OW5q"
# ## Dataset
#
# Se utilizará el dataset MNIST

# + id="3Cgj-YqWFkNd"
# !pip install -U torch torchvision

# + id="HF1wEi1CDKbc"
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader

# + id="IkrxOHvvObki"
IMAGE_PATH = "./"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20

# + colab={"base_uri": "https://localhost:8080/"} id="R1LHTEGWOjvU" outputId="159eb42d-9edf-4d35-905b-94f28026f932"
transform  = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root=IMAGE_PATH, train=True, transform=transform, download=True)

# + [markdown] id="5MB1l3jUPLT6"
# ### Train, test, val

# + id="k8tHVbmwOqBM"
mnist_validation = Subset(mnist_dataset, torch.arange(10000))
mnist_train = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
# Ya debe estar descargado por la celda anterior
mnist_test = datasets.MNIST(root=IMAGE_PATH, train=False, transform=transform, download=False)

# + [markdown] id="gKm0BtHnPR64"
# ### Cargador de datos

# + id="bIelG_65POLs"
batch_size = 64
train_dataloader = DataLoader(mnist_train, batch_size=batch_size)
validation_dataloader = DataLoader(mnist_validation, batch_size=batch_size)


# + [markdown] id="qmeEIBmgPnnf"
# ## Implementación de la red

# + id="aiurhSTDPjO8"
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        # Primera parte: Convolucional y Pooling
        # Feature extractor: 6 capas
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Capa para aplanar matriz
        self.flatten = nn.Flatten()

        # Segunda parte: Clasificador
        # 4 capas
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=1024),
            nn.ReLU(),
            # Usamos Dropout
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        # Extrayendo features
        features = self.feature(x)
        # Aplicando aplanado para la capa lineal del clasificador
        features_flat = self.flatten(features)
        # Aplicando clasificador
        out = self.classifier(features_flat)
        return out


# + colab={"base_uri": "https://localhost:8080/"} id="F0jc3QuOsFay" outputId="099192e4-eaaa-4886-9a50-c7df3174e40a"
print(f"Usando el dispositivo: {DEVICE}")
model = MyCNN().to(DEVICE)

# + [markdown] id="hlpfN_yAsuhW"
# ### Optimizadores y función de perdida

# + id="IT0tSv9OsslM"
# Función de pérdida: Entropía Cruzada
loss_fn = nn.CrossEntropyLoss()
# Optimizador (Gradiente): Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# + [markdown] id="Q4fggeNgs-Nf"
# ## Función de entrenamiento

# + id="0MWcWhaos5Rj"
def calculate_validation_loss(model, validation_data, loss_hist, acc_hist, epoch):
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in validation_data:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss_hist[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
            acc_hist[epoch] += is_correct.sum()
        loss_hist[epoch] /= len(validation_data.dataset)
        acc_hist[epoch] /= len(validation_data.dataset)
    return loss_hist, acc_hist

def train(model, train_data, valid_data, epochs):
    loss_hist_train = [0] * epochs
    accuracy_hist_train = [0] * epochs

    loss_hist_valid = [0] * epochs
    accuracy_hist_valid = [0] * epochs

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        for x_batch, y_batch in train_data:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # Realiza la predicción
            y_pred = model(x_batch)
            # Calcular pérdida
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Guardando historial de loss
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()

        loss_hist_train[epoch] /= len(train_data.dataset)
        accuracy_hist_train[epoch] /= len(train_data.dataset)

        # Validación
        loss_hist_valid, accuracy_hist_valid = calculate_validation_loss(
            model,
            valid_data,
            loss_hist_valid,
            accuracy_hist_valid,
            epoch
        )
        print(f"Epoch: {epoch} - Accuracy Train: {accuracy_hist_train[epoch]:.4f} - Accuracy Validation: {accuracy_hist_valid[epoch]:.4f}")

    return loss_hist_train, accuracy_hist_train, loss_hist_valid, accuracy_hist_valid


# + colab={"base_uri": "https://localhost:8080/"} id="-6IpywKtw3WR" outputId="0f15b97c-80e9-4061-be9c-f52302ab4b8d"
# %%time
history = train(model, train_dataloader, validation_dataloader, EPOCHS)

# + [markdown] id="zhfrVrRIyL6N"
# ### Mostrando el *accuracy* y la perdida a lo largo del entrenamiento

# + colab={"base_uri": "https://localhost:8080/", "height": 394} id="ymcOXeKmxEVi" outputId="f89e8806-499c-47b5-8000-b458c1b35f2a"
x_arr = np.arange(len(history[0])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, history[0], '-o', label='Train loss')
ax.plot(x_arr, history[2], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, [h.cpu() for h in history[1]], '-o', label='Train acc.')
ax.plot(x_arr, [h.cpu() for h in history[3]], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.show()

# + [markdown] id="QB_Q04TBysgz"
# ## Probando el modelo

# + colab={"base_uri": "https://localhost:8080/"} id="dhTbMQPaytpP" outputId="66ab83f6-c1db-462b-dd99-693d82b69832"
predictions = model(mnist_test.data.unsqueeze(1).float().to(DEVICE) / 255.)
is_correct = (torch.argmax(predictions, dim=1) == mnist_test.targets.to(DEVICE)).float()
print(f"Test Accuracy: {is_correct.mean():.4f}")

# + colab={"base_uri": "https://localhost:8080/", "height": 337} id="I1w4nw4CzDMf" outputId="cd9d93ed-87d7-4d69-be6b-49ad7993d182"
fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    img = mnist_test[i][0][0, :, :]
    # Move the image tensor to the same device as the model
    pred = model(img.unsqueeze(0).unsqueeze(1).to(DEVICE))
    y_pred = torch.argmax(pred)
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, y_pred.item(),
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.show()

# + id="dvRobwiiH0F0"

print("Hola mundo")