import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def generate_dataset(num_samples, test_size):
    rng = np.random.default_rng(seed=42)
    x = rng.random((num_samples, 2), dtype=np.float32) * 0.5
    y = x.sum(axis=1, keepdims=True)
    return train_test_split(x, y, test_size=test_size, random_state=42)


# build model
class SumMLP(nn.Module):
    def __init__(self, hidden_dim=5):
        super().__init__()
        # build a model: 2 -> 5 -> 1
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# train model
def train(
    model, dataloader, optimizer, loss_fn, epochs=100, device=torch.device("cpu")
):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        # calculate the average loss per batch
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # clear the gradientsts
            optimizer.zero_grad(set_to_none=True)

            # forward propagation
            preds = model(x_batch)

            # mse loss
            loss = loss_fn(preds, y_batch)

            # backward propagation
            loss.backward()

            # compute the gradient descent to update the parameters
            optimizer.step()

            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    print("========================")
    print("Training Finished!\n")


# evaluate model
def evaluate(model, x_test, y_test, loss_fn, device=torch.device("cpu")):
    model.to(device)
    model.eval()

    with torch.inference_mode():
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        preds = model(x_test)
        # calculate the MSE loss from the entire test set
        loss = loss_fn(preds, y_test)

    eval_loss = loss.item()
    print(f"Evaluation MSE Loss: {eval_loss:.6f}\n")

    return eval_loss


if __name__ == "__main__":
    # accelerator check
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # data
    x_train_np, x_test_np, y_train_np, y_test_np = generate_dataset(5000, 0.2)
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_test = torch.from_numpy(x_test_np)
    y_test = torch.from_numpy(y_test_np)

    # dataloader
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model, optimizer, loss
    model = SumMLP(hidden_dim=5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    # train
    train(model, dataloader, optimizer, loss_fn, epochs=100, device=device)
    # evaluate
    evaluate(model, x_test, y_test, loss_fn, device=device)

    # predictions
    data = torch.tensor([[0.1, 0.2], [0.2, 0.2]], dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        predictions = model(data)

    data_cpu = data.cpu()
    predictions_cpu = predictions.cpu()

    print("Predictions:")
    for d, p in zip(data_cpu, predictions_cpu):
        expected = d[0].item() + d[1].item()
        print(
            f"{d[0].item():.1f} + {d[1].item():.1f} = {p[0].item():.4f} (Expected: {expected:.1f})"
        )
