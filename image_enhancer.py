import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class SuperResolutionNet(nn.Module):
    def __init__(self, num_channels=3, num_residuals=8):
        super(SuperResolutionNet, self).__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_residuals):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Upscaling (2x)
        self.conv_up1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        
        # Final convolution
        self.conv_output = nn.Conv2d(64, num_channels, kernel_size=9, padding=4)

    def forward(self, x):
        x = F.relu(self.conv_input(x))
        residual = x
        x = self.res_blocks(x)
        x += residual
        x = F.relu(self.pixel_shuffle1(self.conv_up1(x)))
        x = self.conv_output(x)
        return torch.tanh(x)

class AdvancedImageEnhancer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SuperResolutionNet().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def enhance_image_quality(self, image):
        """Enhance image quality using advanced techniques"""
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if image is too small
        min_size = 256
        h, w = image.shape[:2]
        if h < min_size or w < min_size:
            scale = min_size / min(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        # Convert to PIL Image for processing
        image_pil = Image.fromarray(image)
        
        # Transform for model
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Apply super-resolution
            output = self.model(input_tensor)
            
            # Post-process the output
            output = output.squeeze().cpu().numpy()
            output = np.transpose(output, (1, 2, 0))
            output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Apply additional enhancements
            output = cv2.detailEnhance(output, sigma_s=10, sigma_r=0.15)
            
            # Adjust contrast and brightness
            lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            output = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

        return output

    def process_image(self, input_image):
        """Main processing pipeline"""
        if input_image is None:
            return None

        try:
            # Convert to numpy array if needed
            if isinstance(input_image, str):
                input_image = cv2.imread(input_image)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            elif isinstance(input_image, np.ndarray):
                if input_image.dtype != np.uint8:
                    input_image = (input_image * 255).astype(np.uint8)
                if len(input_image.shape) == 2:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
                elif input_image.shape[2] == 4:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)

            # Apply enhancement
            enhanced = self.enhance_image_quality(input_image)
            return enhanced

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists("enhanced_images"):
        os.makedirs("enhanced_images")

    enhancer = AdvancedImageEnhancer()
    
    # Example usage
    input_path = "input_image.jpg"  # Replace with your image path
    output_path = "enhanced_images/enhanced_image.jpg"
    
    try:
        enhanced_image = enhancer.process_image(input_path)
        if enhanced_image is not None:
            cv2.imwrite(output_path, enhanced_image)
            print(f"Image enhanced successfully! Saved to {output_path}")
        else:
            print("Enhanced image is None")
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 