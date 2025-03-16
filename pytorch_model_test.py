import torch
import torchvision.transforms as transforms
from PIL import Image

# Define device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture (use the same model you trained)
from torchvision import models

model = models.resnet18(pretrained=False)  # Use the same model architecture
num_classes = 24  # Update this based on your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load saved weights
model.load_state_dict(torch.load("resnet18_fruit_freshness.pth", map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode
# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use ImageNet normalization
])
# Load your image
image_path = "image copy 2.png"  # Change to your image file
image = Image.open(image_path)

# Preprocess the image
input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
from time import perf_counter
t1=perf_counter()
# Run inference
with torch.no_grad():
    output = model(input_tensor)
    print(output)
    predicted_class = torch.argmax(output, dim=1).item()
t2=perf_counter()
print("Time taken for inference : ",(t2-t1))
# Define class labels (same order as your dataset)
class_labels = [
    'FreshApple', 'FreshBanana', 'FreshCucumber', 'FreshGrape', 'FreshGuava', 'FreshJujube', 'FreshOkra', 'FreshOrange', 
    'FreshPomegranate', 'FreshPotato', 'FreshStrawberry', 'FreshTomato', 'RottenApple', 'RottenBanana', 'RottenCucumber', 
    'RottenGrape', 'RottenGuava', 'RottenJujube', 'RottenOkra', 'RottenOrange', 'RottenPomegranate', 'RottenPotato', 
    'RottenStrawberry', 'RottenTomato'
]

# Print the predicted class
print(f"Predicted class: {class_labels[predicted_class]}")
