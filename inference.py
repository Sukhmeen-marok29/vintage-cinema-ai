import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from models import CinematicAestheticClassifier, VGGStyleExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and pre-processing helpers
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    
    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return in_transform(image).unsqueeze(0).to(device)

def im_convert(tensor):
    """ Convert a PyTorch normalized tensor back into a standard numpy image """
    image = tensor.to("cpu").clone().detach().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return (image * 255).astype(np.uint8)

def get_gram_matrix(tensor):
    """ Compute the Gram Matrix to calculate style correlation """
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def run_style_transfer(content_img_path, style_img_path, steps=300):
    content = load_image(content_img_path)
    style = load_image(style_img_path)
    
    extractor = VGGStyleExtractor().to(device).eval()
    
    # Freeze VGG weights
    for param in extractor.parameters():
        param.requires_grad = False
        
    content_features = extractor(content)
    style_features = extractor(style)
    
    # Calculate target styles using Gram Matrices
    style_grams = [get_gram_matrix(sf) for sf in style_features]
    
    # Start target image as a mutable clone of our content image
    target = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)
    
    style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}
    content_weight = 1e4  # Keeps structural features intact
    style_weight = 1e2    # Blends color patterns and grains
    
    for i in range(1, steps + 1):
        target_features = extractor(target)
        
        # Compute Content Loss (comparing middle block layers)
        content_loss = torch.mean((target_features[3] - content_features[3])**2)
        
        # Compute Style Loss across all targeting layers
        style_loss = 0
        for layer_idx, layer_name in enumerate(['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']):
            target_gram = get_gram_matrix(target_features[layer_idx])
            layer_style_loss = style_weights[layer_name] * torch.mean((target_gram - style_grams[layer_idx])**2)
            style_loss += layer_style_loss / (target_features[layer_idx].shape[1] * target_features[layer_idx].shape[2] * target_features[layer_idx].shape[3])

        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    return im_convert(target)

def process_pipeline(input_image_path, classes=['70s_technicolor', '90s_filmic', 'neo_noir']):
    """ Fully couples classifier prediction directly into targeted color matching """
    # 1. Classification
    classifier = CinematicAestheticClassifier(num_classes=len(classes)).to(device)
    try:
        classifier.load_state_dict(torch.load('aesthetic_classifier.pth', map_location=device))
    except FileNotFoundError:
        print("Warning: Classifier weights missing. Using uninitialized state for demo runs.")
    
    classifier.eval()
    img_tensor = load_image(input_image_path, max_size=224)
    
    with torch.no_grad():
        prediction = classifier(img_tensor)
        predicted_class_idx = torch.argmax(prediction, dim=1).item()
        predicted_era = classes[predicted_class_idx]
        
    print(f"Pipeline Result: Input matches the closest profile of -> {predicted_era}")
    
    # 2. Automatically map to style reference asset
    style_reference_path = f"dataset/styles/{predicted_era}_style.jpg"
    
    # 3. Generate new stylized output array
    stylized_output = run_style_transfer(input_image_path, style_reference_path, steps=150)
    return predicted_era, stylized_output