import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage import exposure
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

class ImageEnhancementNetwork(nn.Module):
    def __init__(self):
        super(ImageEnhancementNetwork, self).__init__()
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)
        return torch.tanh(x3)

class AdvancedImageEnhancer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImageEnhancementNetwork().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_image(self, image_path):
        """Load and preprocess image."""
        return cv2.imread(image_path)

    def adjust_gamma(self, image, gamma=1.2):
        """Adjust image gamma."""
        return exposure.adjust_gamma(image, gamma)

    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def sharpen_image(self, image):
        """Apply sharpening filter."""
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    def denoise_image(self, image):
        """Apply denoising."""
        return cv2.fastNlMeansDenoisingColored(image)

    def deep_enhance(self, image):
        """Apply deep learning enhancement."""
        # Convert to PIL Image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Transform for model
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert back to numpy array
        output = output.squeeze().cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    def enhance_image(self, image_path, output_path):
        """Main enhancement pipeline."""
        # Load image
        image = self.load_image(image_path)
        if image is None:
            raise ValueError("Could not load image")

        # Apply enhancement pipeline
        enhanced = self.denoise_image(image)
        enhanced = self.adjust_gamma(enhanced)
        enhanced = self.enhance_contrast(enhanced)
        enhanced = self.sharpen_image(enhanced)
        enhanced = self.deep_enhance(enhanced)

        # Save result
        cv2.imwrite(output_path, enhanced)
        return enhanced

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists("enhanced_images"):
        os.makedirs("enhanced_images")

    enhancer = AdvancedImageEnhancer()
    
    # Example usage
    input_path = "input_image.jpg"  # Replace with your image path
    output_path = "enhanced_images/enhanced_image.jpg"
    
    try:
        enhanced_image = enhancer.enhance_image(input_path, output_path)
        print(f"Image enhanced successfully! Saved to {output_path}")
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")

if __name__ == "__main__":
    main() 