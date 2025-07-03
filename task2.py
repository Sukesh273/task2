# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# # Define the CNN architecture
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3 input channels, 32 output channels
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 64 channels * 8x8 image after pooling
#         self.fc2 = nn.Linear(512, 10)  # 10 output classes
        
#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.25)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# # Data augmentation and normalization
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Load CIFAR-10 dataset
# train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
# test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Initialize model, loss function, and optimizer
# model = CNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Training function
# def train(model, train_loader, criterion, optimizer, epochs=10):
#     train_losses = []
#     for epoch in range(epochs):
#         running_loss = 0.0
#         model.train()
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        
#         epoch_loss = running_loss / len(train_loader)
#         train_losses.append(epoch_loss)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
#     return train_losses

# # Evaluation function
# def evaluate(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     accuracy = 100 * correct / total
#     print(f"Test Accuracy: {accuracy:.2f}%")
#     return accuracy

# # Train the model
# train_losses = train(model, train_loader, criterion, optimizer, epochs=15)

# # Evaluate the model
# test_accuracy = evaluate(model, test_loader)

# # Plot training loss
# plt.plot(train_losses)
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()