import os
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import cv2
import numpy as np

def load_model(model_path, num_classes):
    # Create the model and load the trained weights
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"model_path: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(image_path, model, device, threshold=0.5):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        prediction = model(image_tensor)

    # Extract bounding boxes with score above threshold
    boxes = prediction[0]['boxes'][prediction[0]['scores'] > threshold].cpu().numpy()
    labels = prediction[0]['labels'][prediction[0]['scores'] > threshold].cpu().numpy()
    scores = prediction[0]['scores'][prediction[0]['scores'] > threshold].cpu().numpy()

    return boxes, labels, scores

def draw_boxes(image_path, boxes, labels, scores):
    image = cv2.imread(image_path)
    for box, label, score in zip(boxes, labels, scores):
        color = (0, 255, 0)
        cv2.rectangle(image, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), color, 2)
        text = f"Class: {label}, Score: {score:.2f}"
        cv2.putText(image, text, (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # cv2.imshow("Predictions", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("bounding_box_predicted.jpg", image)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load the model
    num_classes = 21  # Assuming 20 classes + 1 background class
    model_path = "fasterrcnn_resnet50_fpn_custom.pth"
    model = load_model(model_path, num_classes)
    model.to(device)
    
    # Provide path to your test image
    image_path = "images/000619_png_jpg.rf.75d5b762b01e8ea4c8dceb068e378522.jpg"
    
    # Get predictions
    boxes, labels, scores = predict(image_path, model, device)
    
    # Draw bounding boxes
    draw_boxes(image_path, boxes, labels, scores)