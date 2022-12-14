import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)


batch_size = 64

train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y in test_dataloader:
    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)
    print(f"Shape of X [N, C, H, W]: {X.shape}, {train_batches, test_batches}")
    print(f"Shape of y [N, C, H, W]: {y.shape}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} divce")

#Define a model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#optimizing the model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3) #the model.parameters() will be optimized


#train model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        if (batch == 0):
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y [N, C, H, W]: {y.shape}")

        #compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss: > 7f} [{current: > 5d}/{size: >5d}]")
        

#evelation
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    save_model = True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if save_model == True:
                save_model = False

                print("X: ", X)
                jit_model = torch.jit.trace(model, (X))
                print("torchscript model: \n", jit_model)
                jit_model.save("mnist_model.pt")

        test_loss /= num_batches
        correct /= size
        print(f"Test ERROR: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss: >8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n -------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("done")

torch.save(model.state_dict(), "model.pth")
print("Saved Pytorch Model State to model.pth")


