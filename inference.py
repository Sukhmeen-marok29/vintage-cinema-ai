import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from models import CinematicAestheticClassifier, VGGStyleExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    image = tensor.to("cpu").clone().detach().numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return (image * 255).astype(np.uint8)

def get_gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def fast_color_transfer(content_path, style_path, intensity=1.0):
    """ Instantly maps the color distribution of the vintage asset to the input image """
    src = cv2.imread(content_path)
    ref = cv2.imread(style_path)
    
    src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    
    s_mean, s_std = cv2.meanStdDev(src)
    r_mean, r_std = cv2.meanStdDev(ref)
    
    s_mean, s_std = s_mean.flatten(), s_std.flatten()
    r_mean, r_std = r_mean.flatten(), r_std.flatten()
    
    l, a, b = cv2.split(src)
    l = ((l - s_mean[0]) * (r_std[0] / (s_std[0] + 1e-5))) + r_mean[0]
    a = ((a - s_mean[1]) * (r_std[1] / (s_std[1] + 1e-5))) + r_mean[1]
    b = ((b - s_mean[2]) * (r_std[2] / (s_std[2] + 1e-5))) + r_mean[2]
    
    l = np.clip(l, 0, 255).astype(np.uint8)
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    transfer = cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB)
    
    # Blend with original based on slider intensity
    orig = cv2.cvtColor(cv2.imread(content_path), cv2.COLOR_BGR2RGB)
    transfer = cv2.resize(transfer, (orig.shape[1], orig.shape[0]))
    blended = cv2.addWeighted(transfer, intensity, orig, 1 - intensity, 0)
    return blended

def run_style_transfer(content_img_path, style_img_path, steps=30, intensity=0.5):
    content = load_image(content_img_path)
    style = load_image(style_img_path)
    
    extractor = VGGStyleExtractor().to(device).eval()
    for param in extractor.parameters():
        param.requires_grad = False
        
    content_features = extractor(content)
    style_features = extractor(style)
    style_grams = [get_gram_matrix(sf) for sf in style_features]
    
    target = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)
    
    style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}
    
    # Dynamic weights scaled by user intensity slider
    content_weight = 1e1 
    style_weight = 1e4 * (intensity * 10) 
    
    for i in range(1, steps + 1):
        target_features = extractor(target)
        content_loss = torch.mean((target_features[3] - content_features[3])**2)
        
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

def process_pipeline(input_image_path, selected_era=None, mode="Instant Color Mode", intensity=1.0, classes=['70s_technicolor', '90s_filmic', 'neo_noir']):
    # 1. Classification (Always runs to show AI capabilities)
    classifier = CinematicAestheticClassifier(num_classes=len(classes)).to(device)
    try:
        classifier.load_state_dict(torch.load('aesthetic_classifier.pth', map_location=device))
    except FileNotFoundError:
        pass
    
    classifier.eval()
    img_tensor = load_image(input_image_path, max_size=224)
    with torch.no_grad():
        prediction = classifier(img_tensor)
        predicted_class_idx = torch.argmax(prediction, dim=1).item()
        predicted_era = classes[predicted_class_idx]
    
    # Use override if selected by user
    final_era = selected_era if selected_era and selected_era != "Auto-Detect" else predicted_era
    style_reference_path = f"dataset/styles/{final_era}_style.jpg"
    
    # 2. Render based on execution mode
    if mode == "Instant Color Mode":
        stylized_output = fast_color_transfer(input_image_path, style_reference_path, intensity)
    else:
        stylized_output = run_style_transfer(input_image_path, style_reference_path, steps=30, intensity=intensity)
        
    return predicted_era, final_era, stylized_output