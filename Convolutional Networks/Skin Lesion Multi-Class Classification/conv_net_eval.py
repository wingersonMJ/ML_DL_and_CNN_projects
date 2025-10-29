import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.metrics import confusion_matrix

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import medmnist
from medmnist import INFO

from conv_net_model import MyNetwork

# get the data 
nChannels = INFO['dermamnist']['n_channels']
nClasses = len(INFO['dermamnist']['label'])
DataClass = medmnist.DermaMNIST

# send images to tensors and normalize w/ mean 0 and sd 1
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1])
])

# split into train, val, test
trainingData = DataClass(split='train', transform=data_transform, download=True)
validationData = DataClass(split='val', transform=data_transform, download=True)
testData = DataClass(split='test', transform=data_transform, download=True)

# print number of samples of each class in training data 
all_labels = [int(label) if isinstance(label, (np.ndarray, torch.Tensor)) else label for _, label in trainingData]
class_counts = Counter(all_labels)
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} samples")

#####################
# batch size and data loader
batchSize = 128
trainLoader = data.DataLoader(dataset=trainingData, batch_size=batchSize, shuffle=True)
validationLoader = data.DataLoader(dataset=validationData, batch_size=batchSize, shuffle=True)
testLoader = data.DataLoader(dataset=testData, batch_size=batchSize, shuffle=False)

# eval function
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in loader:
            device = next(model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device).to(torch.long).view(targets.size(0))

            outputs = model(inputs)
            total_loss += loss_fn(outputs, targets).item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def compute_metrics(loader, model):
    model.eval()
    device = next(model.parameters()).device

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).to(torch.long)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    sensitivity = cm.diagonal() / cm.sum(axis=1)
    specificity = (cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + cm.diagonal()) / (cm.sum() - cm.sum(axis=1))
    return sensitivity, specificity, all_targets, all_preds


model = MyNetwork(nChannels, nClasses)
model.trainModel(trainLoader, validationLoader, evaluate)


# Evaluate on test set
test_loss, test_accuracy = evaluate(testLoader, model)
sensitivity, specificity, all_targets, all_preds = compute_metrics(testLoader, model)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Sensitivity by Class: {sensitivity}")
print(f"Specificity by Class: {specificity}")

# plot
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Class {i}' for i in range(nClasses)],
            yticklabels=[f'Class {i}' for i in range(nClasses)])
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('Convolutional Networks/Skin Lesion Multi-Class Classification/figs/test_confusion_matrix.png')
plt.show()

# Plot
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=[f'Class {i}' for i in range(nClasses)],
            yticklabels=[f'Class {i}' for i in range(nClasses)])
plt.title('Confusion Matrix (Percent per True Class)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('Convolutional Networks/Skin Lesion Multi-Class Classification/figs/test_confusion_matrix_percent.png')
plt.show()

# plot
x = np.arange(nClasses)
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, sensitivity, width, label='Sensitivity')
plt.bar(x + width/2, specificity, width, label='Specificity')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Per-Class Sensitivity and Specificity')
plt.xticks(x, [f'Class {i}' for i in range(nClasses)])
plt.legend()
plt.tight_layout()
plt.savefig('Convolutional Networks/Skin Lesion Multi-Class Classification/figs/test_class_metrics.png')
plt.show()
