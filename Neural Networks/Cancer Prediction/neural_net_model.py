import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class MyNetwork(torch.nn.Module):
    # initialize
    def __init__(self, nFeatures, hidden_sizes=[50, 50], lr=0.01, momentum=0.5, nEpochs=2500, batchSize=500):
        super().__init__()

        self.nFeatures = nFeatures
        self.learningRate = lr
        self.momentum = momentum
        self.nEpochs = nEpochs
        self.batchSize = batchSize

        # Feed forward part
        layers = []
        in_size = nFeatures
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, h))
            layers.append(torch.nn.Sigmoid())
            in_size = h
        layers.append(torch.nn.Linear(in_size, 1))
        layers.append(torch.nn.Sigmoid())
        self.fc = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
    
    def compute_metrics(self, y_true, y_pred):
        y_pred_bin = torch.round(y_pred)
        TP = ((y_true == 1) & (y_pred_bin == 1)).sum().item()
        TN = ((y_true == 0) & (y_pred_bin == 0)).sum().item()
        FP = ((y_true == 0) & (y_pred_bin == 1)).sum().item()
        FN = ((y_true == 1) & (y_pred_bin == 0)).sum().item()

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        return accuracy, sensitivity, specificity

    def plot_distributions(self, probs, labels, dataset_name, save_path):
        pos = probs[labels == 1]
        neg = probs[labels == 0]
        plt.figure(figsize=(8, 6))
        plt.hist(pos, bins=50, alpha=0.5, label=f'{dataset_name}: Positive')
        plt.hist(neg, bins=50, alpha=0.5, label=f'{dataset_name}: Negative')
        plt.title(f'{dataset_name} Probability Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(save_path)
        plt.show()

    def trainModel(self, trainX, trainY, valX, valY, logPath='./logs'):
        writer = SummaryWriter(logPath)
        device = torch.device('cpu')
        self.to(device=device)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate, momentum=self.momentum)
        loss_fn = torch.nn.BCELoss()

        nTrain = trainX.shape[0]
        nVal = valX.shape[0]
        nTrainBatches = -(-nTrain // self.batchSize)
        nValBatches = -(-nVal // self.batchSize)

        for epoch in range(self.nEpochs):
            self.train()
            epochLoss, all_probs, all_labels = 0, [], []
            for b in range(nTrainBatches):
                optimizer.zero_grad()
                x = torch.tensor(trainX[b*self.batchSize:(b+1)*self.batchSize, :], dtype=torch.float32)
                y = torch.tensor(trainY[b*self.batchSize:(b+1)*self.batchSize].reshape(-1, 1), dtype=torch.float32)
                y_pred = self(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                epochLoss += loss.item() * x.shape[0] / nTrain
                all_probs.append(y_pred.detach())
                all_labels.append(y)
            all_probs, all_labels = torch.cat(all_probs), torch.cat(all_labels)
            acc, sens, spec = self.compute_metrics(all_labels, all_probs)

            # Validation
            self.eval()
            valLoss, v_probs, v_labels = 0, [], []
            with torch.no_grad():
                for b in range(nValBatches):
                    x = torch.tensor(valX[b*self.batchSize:(b+1)*self.batchSize, :], dtype=torch.float32)
                    y = torch.tensor(valY[b*self.batchSize:(b+1)*self.batchSize].reshape(-1, 1), dtype=torch.float32)
                    y_pred = self(x)
                    valLoss += loss_fn(y_pred, y).item() * x.shape[0] / nVal
                    v_probs.append(y_pred)
                    v_labels.append(y)
            v_probs, v_labels = torch.cat(v_probs), torch.cat(v_labels)
            vacc, vsens, vspec = self.compute_metrics(v_labels, v_probs)

            print(f'Epoch {epoch:04d} | Train Loss: {epochLoss:.4f}, Acc: {acc:.3f} | Val Loss: {valLoss:.4f}, Acc: {vacc:.3f}')
            writer.add_scalar('Train/Loss', epochLoss, epoch)
            writer.add_scalar('Val/Loss', valLoss, epoch)
            writer.add_scalar('Train/Accuracy', acc, epoch)
            writer.add_scalar('Val/Accuracy', vacc, epoch)
        writer.close()

    def evaluate(self, testX, testY):
        device = torch.device('cpu')
        self.to(device=device)
        self.eval()
        x = torch.tensor(testX, dtype=torch.float32)
        y = torch.tensor(testY.reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            y_pred = self(x)
        acc, sens, spec = self.compute_metrics(y, y_pred)
        print(f'Test Accuracy: {acc:.3f}, Sensitivity: {sens:.3f}, Specificity: {spec:.3f}')
        self.plot_distributions(y_pred, y, 'Test', save_path="Neural Networks/Cancer Prediction/figs/distribution_plot.png")