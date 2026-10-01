import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets
train_data = datasets.ImageFolder("dataset/train", transform=transform_train)
val_data   = datasets.ImageFolder("dataset/val", transform=transform_val)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data, batch_size=16)

# Model
model = get_model().to(device)

# Weighted loss (melanoma more important)
class_weights = torch.tensor([1.0, 2.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/5 - Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "melanoma_model.pth")
print("Model saved as melanoma_model.pth")
