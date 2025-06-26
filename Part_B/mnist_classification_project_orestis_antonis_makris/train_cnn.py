import torch, torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import get_dataloaders
from cnn_model import CNN

import torch.nn as nn

def train(epochs=10, lr=0.01, batch_size=64):
    train_loader, test_loader = get_dataloaders(batch_size)
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    history = {'train_loss':[], 'test_loss':[], 'test_acc':[]}  # record
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x,y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        # eval
        model.eval()

     

        total_loss, correct = 0, 0
        with torch.no_grad():
            for x,y in test_loader:
                out = model(x)
                total_loss += criterion(out,y).item()
                pred = out.argmax(dim=1)
                correct += pred.eq(y).sum().item()
        history['test_loss'].append(total_loss/len(test_loader))
        history['test_acc'].append(correct/len(test_loader.dataset))
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, test_loss={history['test_loss'][-1]:.4f}, test_acc={history['test_acc'][-1]:.4f}")
    # plots
    plt.figure(); plt.plot(history['test_acc'], label='Accuracy'); plt.xlabel('Epoch'); plt.legend(); plt.show()
    plt.figure(); plt.plot(history['test_loss'], label='Loss'); plt.xlabel('Epoch'); plt.legend(); plt.show()
    torch.save(model.state_dict(), 'cnn_mnist.pth')

if __name__=='__main__':
    train()