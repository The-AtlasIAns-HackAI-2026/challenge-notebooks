import sys
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLIENT_ID = int(sys.argv[1])
NUM_CLIENTS = 3

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def load_data(client_id):
    transform = transforms.ToTensor()

    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_indices = list(range(client_id, len(trainset), NUM_CLIENTS))
    test_indices = list(range(client_id, len(testset), NUM_CLIENTS))

    trainloader = DataLoader(
        Subset(trainset, train_indices),
        batch_size=32,
        shuffle=True
    )

    testloader = DataLoader(
        Subset(testset, test_indices),
        batch_size=32,
        shuffle=False
    )

    return trainloader, testloader

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        print(f"Client {CLIENT_ID}: sending parameters", flush=True)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"Client {CLIENT_ID}: training started", flush=True)

        self.set_parameters(parameters)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        for epoch in range(1):
            for data, target in self.trainloader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        print(f"Client {CLIENT_ID}: training finished", flush=True)

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"Client {CLIENT_ID}: evaluation started", flush=True)

        self.set_parameters(parameters)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in self.testloader:
                output = self.model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        accuracy = correct / len(self.testloader.dataset)

        print(f"Client {CLIENT_ID}: accuracy = {accuracy:.4f}", flush=True)

        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

model = Net()
trainloader, testloader = load_data(CLIENT_ID)

fl.client.start_client(
    server_address="127.0.0.1:9090",
    client=FlowerClient(model, trainloader, testloader).to_client()
)
