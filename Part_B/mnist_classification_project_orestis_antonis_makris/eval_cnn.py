import torch
from cnn_model import CNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

def conf_matrix():
    model=CNN(); model.load_state_dict(torch.load('cnn_mnist.pth'))
    test_ds=datasets.MNIST(root='./data', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]))
    loader=DataLoader(test_ds, batch_size=1000)
    all_preds, all_labels = [],[]
    with torch.no_grad():
        for x,y in loader:
            all_labels.extend(y.numpy())
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__=='__main__':
    conf_matrix()