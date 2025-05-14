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
    def __init__(self, num_channels=3, num_residuals=16):
        super(SuperResolutionNet, self).__init__()
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_residuals):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Post residual convolution
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)
        
        # Upscaling layers
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        # Final convolutions
        self.conv_output = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, num_channels, kernel_size=9, padding=4)
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.prelu1(self.conv_input(x))
        
        # Store residual
        residual = x
        
        # Residual blocks
        x = self.res_blocks(x)
        x = self.bn_mid(self.conv_mid(x))
        
        # Skip connection
        x = x + residual
        
        # Upscaling
        x = self.upscale(x)
        
        # Final convolutions
        x = self.conv_output(x)
        
        return torch.tanh(x)

class AdvancedImageEnhancer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SuperResolutionNet().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def enhance_image_quality(self, image):
        """Enhance image quality using advanced techniques"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3 and image.dtype == np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Store original size for later
            original_h, original_w = image.shape[:2]

            # Ensure dimensions are multiples of 4
            h, w = ((original_h + 3) // 4) * 4, ((original_w + 3) // 4) * 4
            if h != original_h or w != original_w:
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)

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
                
                # Denormalize using ImageNet stats
                output = output * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                output = (output * 255).clip(0, 255).astype(np.uint8)
                
                # Color correction to match original
                output_lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
                original_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                
                # Match color statistics
                for i in range(3):  # For each LAB channel
                    output_mean = np.mean(output_lab[:,:,i])
                    output_std = np.std(output_lab[:,:,i])
                    target_mean = np.mean(original_lab[:,:,i])
                    target_std = np.std(original_lab[:,:,i])
                    
                    output_lab[:,:,i] = ((output_lab[:,:,i] - output_mean) * (target_std / output_std)) + target_mean
                
                output = cv2.cvtColor(output_lab, cv2.COLOR_LAB2RGB)
                
                # Enhance details without introducing artifacts
                detail_kernel = np.array([[-1,-1,-1],
                                        [-1, 9,-1],
                                        [-1,-1,-1]]) / 9
                output = cv2.filter2D(output, -1, detail_kernel)
                
                # Apply subtle contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = clahe.apply(l)
                output = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
                
                # Final touch: subtle bilateral filtering to preserve edges
                output = cv2.bilateralFilter(output, 5, 50, 50)

            return output

        except Exception as e:
            print(f"Error in enhance_image_quality: {str(e)}")
            return None

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
            if enhanced is None:
                return None
                
            # Ensure we maintain original colors
            enhanced_yuv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YUV)
            original_yuv = cv2.cvtColor(cv2.resize(input_image, (enhanced.shape[1], enhanced.shape[0]), 
                                                 interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_RGB2YUV)
            
            # Keep enhanced luminance but original colors
            enhanced_yuv[:,:,1:] = original_yuv[:,:,1:]
            enhanced = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2RGB)
            
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