import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import ViTModel, ViTConfig
from sklearn.model_selection import train_test_split

# Define the custom dataset class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sentinel2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["agri", "barrenland", "grassland", "urban"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        # Loop through each class and collect images
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls, "s2")  # Only Sentinel-2 images
            if os.path.exists(class_dir):
                img_list = [f for f in os.listdir(class_dir) if f.endswith(".png")]
                if not img_list:
                    print(f"Warning: No images found in {class_dir}")
                for img_name in img_list:
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}. Check the folder structure.")

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        return image, label


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit model input
    transforms.ToTensor(),
])

# Load the dataset
dataset = Sentinel2Dataset(root_dir="v_2", transform=transform)

# Split dataset (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
'''
# Print dataset sizes
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# Example: Check first batch
for images, labels in train_loader:
    print(f"Batch size: {images.shape}, Labels: {labels[:10]}")
    break  # Only check first batch
'''
class MaskedAutoencoder(nn.Module):
    def __init__(self, vit_model, mask_ratio=0.75):
        super(MaskedAutoencoder, self).__init__()
        self.vit = vit_model
        self.mask_ratio = mask_ratio
        self.decoder = nn.Linear(vit_model.config.hidden_size, vit_model.config.patch_size ** 2 * 3)

    def forward(self, x, mask=True):
        batch_size, _, height, width = x.shape
        patch_size = self.vit.config.patch_size

        # Flatten image into patches
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, 3, -1, patch_size * patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(batch_size, -1, patch_size * patch_size * 3)

        # Apply masking
        if mask:
            num_patches = patches.shape[1]
            num_masked = int(self.mask_ratio * num_patches)
            mask_indices = torch.randperm(num_patches)[:num_masked]
            patches[:, mask_indices, :] = 0

        # Encode using ViT
        encoded = self.vit(pixel_values=x).last_hidden_state

        # Decode masked patches
        decoded = self.decoder(encoded[:, 1:, :])

        grid_size = int((decoded.shape[1]) ** 0.5)
        decoded = decoded.view(batch_size, grid_size, grid_size, patch_size, patch_size, 3)
        decoded = decoded.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed = decoded.view(batch_size, 3, height, width)

        return reconstructed
import torch.nn as nn

class ViTMAEClassification(nn.Module):
    def __init__(self, mae_model, num_classes, dropout_rate=0.3):
        super(ViTMAEClassification, self).__init__()
        self.mae = mae_model
        self.classifier = nn.Sequential(
            nn.Linear(mae_model.vit.config.hidden_size, 512),  # Reduce feature size
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout to prevent overfitting
            nn.Linear(512, num_classes)  # Final classification layer
        )

    def forward(self, x):
        encoded = self.mae.vit(pixel_values=x).pooler_output
        return self.classifier(encoded)

# Initialize Model
vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config=vit_config)
mae_model = MaskedAutoencoder(vit_model)
vitmae_cls_model = ViTMAEClassification(mae_model, num_classes=4).to(device)

# Optimizer and Loss
optimizer_mae = Adam(mae_model.parameters(), lr=1e-5)
criterion_mae = nn.MSELoss()
optimizer_cls = Adam(vitmae_cls_model.parameters(), lr=1e-5)
criterion_cls = CrossEntropyLoss()

def pretrain_mae(model, loader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()

            reconstructed = model(images, mask=True)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Pretrain Loss: {total_loss/len(loader):.4f}")

# Training Function
def train_classification(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100
    return total_loss / len(loader), accuracy

# Validation Function
def validate_classification(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100
    return total_loss / len(loader), accuracy

# Training Loop
#pretrain_mae(mae_model, train_loader, optimizer_mae, criterion_mae, device, epochs=5)
'''
# Fine-tuning
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_classification(vitmae_cls_model, train_loader, optimizer_cls, criterion_cls, device)
    val_loss, val_accuracy = validate_classification(vitmae_cls_model, val_loader, criterion_cls, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

# Save the Model
torch.save(vitmae_cls_model.state_dict(), "vit_sentinel2_model2.pth")
print("Model saved successfully!")
'''
import torch.nn.functional as F


