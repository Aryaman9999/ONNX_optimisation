import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load ONNX model
onnx_model_path = "resnet18_fruit_freshness.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CoreMLExecutionProvider'])

# Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image_path = "image.png"  # Change this to your image path
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Convert to NumPy array
inputs_np = input_tensor.numpy()
from time import perf_counter
t1=perf_counter()
# Run ONNX inference
outputs = ort_session.run(None, {"input": inputs_np})
predicted_class = np.argmax(outputs[0])
t2=perf_counter()
print("Time for inference : ",(t2-t1))
# Class labels
class_labels = [
    'FreshApple', 'FreshBanana', 'FreshCucumber', 'FreshGrape', 'FreshGuava', 'FreshJujube', 'FreshOkra', 'FreshOrange', 
    'FreshPomegranate', 'FreshPotato', 'FreshStrawberry', 'FreshTomato', 'RottenApple', 'RottenBanana', 'RottenCucumber', 
    'RottenGrape', 'RottenGuava', 'RottenJujube', 'RottenOkra', 'RottenOrange', 'RottenPomegranate', 'RottenPotato', 
    'RottenStrawberry', 'RottenTomato'
]

# Print result
print(f"Predicted class: {class_labels[predicted_class]}")
