import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_dir)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.imgs[idx].replace('.jpg', '.txt'))
        
        img = Image.open(img_path).convert("RGB")
        
        # Read labels
        with open(label_path, 'r') as f:
            boxes = []
            labels = []
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.split())
                class_id = int(class_id)
                # Convert YOLO format to [x_min, y_min, x_max, y_max]
                img_width, img_height = img.size
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                x_max = (x_center + width / 2) * img_width
                y_max = (y_center + height / 2) * img_height
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_id)
                
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Define transformations
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the dataset
dataset = CustomDataset(img_dir="images", label_dir="labels", transforms=data_transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 21  # Assuming 20 classes + 1 background class
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    i = 0
    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        print(f"Iteration {i}/{len(data_loader)} of epoch {epoch}/{num_epochs}, Loss: {losses.item()}")
        i += 1

print("Training complete!")

# Save the model
torch.save(model.state_dict(), "fasterrcnn_resnet50_fpn_custom.pth")