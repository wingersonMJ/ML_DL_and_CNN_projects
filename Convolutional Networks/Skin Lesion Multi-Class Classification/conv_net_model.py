import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class MyNetwork(torch.nn.Module):

    def __init__(self, inputChannels, outputClasses):

        super().__init__()

        self.inputChannels = inputChannels
        self.outputClasses = outputClasses
        
        # hyperparams
        self.nEpochs = 200 
        self.learningRate = 0.001 

        # Convolution layers
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(self.inputChannels, 32, 3, padding='valid'),  # filters=32, kernel size=3, padding=yep
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # max vs mean pooling? try different kernel size and stride?
            torch.nn.Conv2d(32, 64, 3, padding='valid'), 
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), 
            torch.nn.Conv2d(64, 128, 3, padding='valid'), 
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        ) # three conv layers

        # Fully connected layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64), # adjustable
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(16, self.outputClasses)
        ) # flatten to 1d for 4 layers
        # Dropout included
        # No activation in final layer before passing to number of classes
        # Don't need the activation bc FocalLoss accounts for it

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def trainModel(self, trainLoader, validationLoader, evaluate):
        writer = SummaryWriter(log_dir='./logs')
        nTrainingSamples = len(trainLoader.dataset)

        device = torch.device('cpu') # need to get cuda set up
        self.to(device=device)

        class_counts = torch.zeros(self.outputClasses)
        for _, targets in trainLoader:
            for t in targets:
                class_counts[t] += 1
        
        total_samples = class_counts.sum().item()
        class_weights = total_samples / (self.outputClasses * class_counts)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(torch.float32).to('cpu')  

        print(f"Class Weights: {class_weights}")

        # Loss and optimization
        loss = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)

        # training loop
        for epoch in range(self.nEpochs):
            epochLoss = 0
            epochAccuracy = 0

            self.train()
            with tqdm(trainLoader, unit="batch", desc=f"Epoch {epoch + 1}/{self.nEpochs}") as tepoch:
                for inputs, targets in tepoch:
                    inputs = inputs.to(device)
                    targets = targets.to(device).to(torch.long).view(targets.size(0))

                    # forward, backward, optimizer steps
                    optimizer.zero_grad()
                    y_pred = self(inputs)
                    batchLoss = loss(y_pred, targets)
                    batchLoss.backward()
                    optimizer.step()

                    # get eval metrics
                    epochLoss += batchLoss.item() * inputs.size(0) / nTrainingSamples
                    labels_pred = torch.argmax(y_pred, dim=1)
                    correct = (targets == labels_pred).float().sum().item()
                    epochAccuracy += correct / nTrainingSamples

                    # progress bar!!
                    tepoch.set_postfix(loss=epochLoss, accuracy=100 * epochAccuracy)
            
            print(f' - Training epoch {epoch + 1}/{self.nEpochs}. Loss: {epochLoss:.5f}. Accuracy: {100 * epochAccuracy:.2f}%')

            val_loss, val_accuracy = evaluate(validationLoader, self)
            print(f' - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

            writer.add_scalar('Loss/Train', epochLoss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', epochAccuracy, epoch)
            writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        writer.close()

        pass

    def save(self, path):
        torch.save(self, path)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
