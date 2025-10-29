import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc
from neural_net_model import MyNetwork
import torch

# Load in data, already split into test/val/train
train_df = pd.read_excel("Convolutional Networks/Skin Lesion Multi-Class Classification/Data/GeneExpressionCancer_training.xlsx").to_numpy()
val_df = pd.read_excel("Convolutional Networks/Skin Lesion Multi-Class Classification/Data/GeneExpressionCancer_validation.xlsx").to_numpy()
test_df = pd.read_excel("Convolutional Networks/Skin Lesion Multi-Class Classification/Data/GeneExpressionCancer_test.xlsx").to_numpy()

# shapes
train_df.shape
val_df.shape
test_df.shape 

# Last col is target, so separate it out
trainX, trainY = train_df[:, :-1], train_df[:, -1]
valX, valY = val_df[:, :-1], val_df[:, -1]
testX, testY = test_df[:, :-1], test_df[:, -1]

# standardize everything 
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
valX = scaler.transform(valX)
testX = scaler.transform(testX)

# train
model = MyNetwork(nFeatures=trainX.shape[1], hidden_sizes=[50, 10])
model.trainModel(trainX, trainY, valX, valY, './logs')

# eval
model.evaluate(testX, testY)

# plot
def plot_roc_curve(model, X, y, save_path):
    x_t = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_prob = model(x_t).cpu().numpy().ravel()
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

plot_roc_curve(model, testX, testY, save_path="Neural Networks/Cancer Prediction/figs/roc_curve.png")