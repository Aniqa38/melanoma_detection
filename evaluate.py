import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset
test_data = datasets.ImageFolder("dataset/test", transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

# Load model
model = get_model(pretrained=False).to(device)
model.load_state_dict(torch.load("melanoma_model.pth", map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=["Benign", "Melanoma"]))
