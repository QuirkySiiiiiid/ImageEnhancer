import gradio as gr
import os
from image_enhancer import AdvancedImageEnhancer
import cv2
from PIL import Image
import numpy as np
import torch

# Initialize the image enhancer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
enhancer = AdvancedImageEnhancer()

def check_file_size(file):
    """Check if file size is under 10MB"""
    MAX_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    return os.path.getsize(file.name) <= MAX_SIZE

def process_image(input_image, output_format):
    """Process the image and return enhanced version"""
    if input_image is None:
        return None, "Please upload an image."
    
    try:
        # Convert input to RGB if needed
        if len(input_image.shape) == 2:  # Grayscale
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        elif input_image.shape[2] == 4:  # RGBA
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
        elif input_image.shape[2] == 3 and input_image.dtype == np.uint8:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
        # Apply enhancement
        enhanced = enhancer.process_image(input_image)
        if enhanced is None:
            return None, "Error processing image."
        
        # Ensure the enhanced image is in RGB format
        if len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Create temporary file for download with high quality settings
        temp_path = f"temp_enhanced.{output_format.lower()}"
        if output_format.upper() == 'PNG':
            # Use maximum quality for PNG
            cv2.imwrite(temp_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = no compression
        else:  # JPEG
            # Use maximum quality for JPEG
            cv2.imwrite(temp_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        return enhanced, temp_path
    except Exception as e:
        import traceback
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return None, f"Error processing image: {str(e)}"

# Create the Gradio interface
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🖼️ Advanced Image Enhancer Pro
        
        Upload your image and get an enhanced version with superior quality!
        
        **Features:**
        - AI-powered 4x upscaling
        - Advanced denoising & sharpening
        - Smart contrast enhancement
        - High-quality detail preservation
        - Maximum quality output
        
        **Note:** Maximum file size: 10MB
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    interactive=True,
                    image_mode="RGB"
                )
                output_format = gr.Radio(
                    choices=["PNG", "JPEG"],
                    value="PNG",
                    label="Output Format",
                    info="PNG: Lossless quality (recommended) | JPEG: Smaller file size"
                )
                enhance_btn = gr.Button("✨ Enhance Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(
                    label="Enhanced Image Preview",
                    type="numpy",
                    image_mode="RGB"
                )
                download_btn = gr.File(
                    label="Download Enhanced Image"
                )
        
        # Handle enhancement button click
        enhance_btn.click(
            fn=process_image,
            inputs=[input_image, output_format],
            outputs=[output_image, download_btn],
            api_name="enhance"
        )
        
        # Add file size warning
        input_image.upload(
            lambda x: gr.Warning("File size exceeds 10MB limit!") if not check_file_size(x) else None,
            inputs=[input_image],
            api_name="check_size"
        )

    return app

# For local testing
if __name__ == "__main__":
    app = create_ui()
    try:
        # Try to launch on default port first
        app.launch(
            server_name="0.0.0.0",  # Allow external connections
            share=True,             # Create public URL
            enable_queue=True       # Enable queuing for better handling of multiple users
        )
    except Exception as e:
        print(f"Error launching app: {e}")
        # Try alternate ports if the first attempt fails
        for port in range(7000, 8000):
            try:
                app.launch(
                    server_name="0.0.0.0",
                    server_port=port,
                    share=True,
                    enable_queue=True
                )
                break
            except:
                continue 