import torch
import pandas as pd
import numpy as np

"""
01 - Neutral
02 - Calm
03 - Happy
04 - Sad
05 - Angry
06 - Fearful
07 - Disgust

Capture filename:
Capture
"""

def load_data(p=0.1, batch_size=25):
    # Load/clean
    df = pd.read_csv('Features.csv')
    df['id'] = df['File_Path'].str.extract(r'(\d{2}-\d{2}-\d{2})')[0].str[3:5].astype(int) - 2
    df = df[df["id"] != -1]

    # Split into train/test dataframe
    train_df = df.sample(frac=1-p, random_state=42)
    test_df = df.drop(train_df.index)
    # Create trainloader
    y_train = train_df['id']
    X_train = train_df.drop(['File_Path', 'id'], axis=1)
    y_train = torch.from_numpy(y_train.to_numpy(dtype=np.long))
    X_train = torch.from_numpy(X_train.to_numpy(dtype=np.float32))
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    # Create testloader
    y_test = test_df['id']
    X_test = test_df.drop(['File_Path', 'id'], axis=1)
    y_test = torch.from_numpy(y_test.to_numpy(dtype=np.long))
    X_test = torch.from_numpy(X_test.to_numpy(dtype=np.float32))
    testset = torch.utils.data.TensorDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)

    return trainloader, testloader
    
    
# Basic training method, feel free to change the criterion/optimizer as necessary
def train_model(trainloader, model, epochs=10):
    
    # Set criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Main training loop
    for epoch in range(epochs):
        epoch_loss = 0
        correct=0
        total=0

        model.train()
        for X, Y in trainloader:
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, Y.long())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += float(loss.item())

            # Accuracy calculation
            prediction = torch.argmax(outputs, dim=1)
            correct += (prediction==Y).sum().item()
            total += Y.size(0)
        
        # Print epoch stats
        epoch_loss /= len(trainloader)
        accuracy = correct / total

        print(f'Epoch: [{epoch+1}/{epochs}]\tLoss: [{epoch_loss:.2f}]\tAccuracy: [{accuracy*100:.2f}%]')

# Basic testing method
def test_model(testloader, model):
    criterion = torch.nn.CrossEntropyLoss()
    epoch_loss = 0
    all_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for X, Y in testloader:
            outputs = model(X)
            loss = criterion(outputs, Y.long())
            epoch_loss += loss.item()

            all_labels.append(Y.cpu())
            all_predictions.append(torch.argmax(outputs, dim=1).cpu())
    
    # Calculate loss
    epoch_loss /= len(testloader)
    print(f"Test Loss: [{epoch_loss:.2f}]")

    # Get labels and predictions
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    # Calculate accuracy
    accuracy = (all_predictions == all_labels).float().mean().item()
    print(f'Test Accuracy: [{accuracy*100:.2f}%]')

    df = pd.DataFrame({'labels':all_labels.numpy(), 'predictions':all_predictions.numpy()})
    
    return df, epoch_loss, accuracy

# Basic classification model, tweak this later to get better accuracy
class EmotionPredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(70, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 6)

        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(64)

        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

if __name__ == '__main__':
    model = EmotionPredictionModel()
    trainloader, testloader = load_data()
    train_model(trainloader, model, 200)
    df, epoch_loss, accuracy = test_model(testloader, model)