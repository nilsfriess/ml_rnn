import torch
from torch.utils.data import DataLoader

from pathlib import Path

from utils import RNN, device, SampleMetroDataset, State, writer

# Nombre de stations utilisé
CLASSES = 4
# Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
# Taille du batch
BATCH_SIZE = 16

PATH = "../../data/"

with open(PATH+"hzdataset.pch", "rb") as file:
    matrix_train, matrix_test = torch.load(file)

ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT],
                              length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT],
                             length=LENGTH,
                             stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

savepath = Path("model.pch")
if False and savepath.is_file():
    with savepath.open("rb") as file:
        state = torch.load(file)
else:
    model = RNN(DIM_INPUT, 64, CLASSES, device)
    model = model.to(device)
    # optim = torch.optim.SGD(model.parameters(), lr=0.00001)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # state = State(model, optim)

# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.CrossEntropyLoss()
model.train(True)
NUM_EPOCHS = 200
iteration = 0
for epoch in range(NUM_EPOCHS):
    for x, y in data_train:
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad()

        yhat = model(x)
        loss = loss_fn(yhat, y)

        loss.backward()
        optim.step()

        iteration += 1

    # with savepath.open("wb") as file:
    #     state.epoch = epoch + 1
    #     torch.save(state, file)

    if (epoch+1)%5 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], loss: {loss.item():.6f}")
    writer.add_scalar('Loss/train', loss.item(), epoch)

writer.flush()

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for x, y in data_test:
        x = x.to(device)
        y = y.to(device)

        yhat = model(x)
        yhat = torch.nn.functional.softmax(yhat, dim=0)
        yhat = torch.argmax(yhat, dim=1)

        correct += (yhat == y).sum().item()
        total += y.size(0)

    print(f"Test Accuracy of the model on the {total} test data: {100 * correct / total:.2f}%") 
