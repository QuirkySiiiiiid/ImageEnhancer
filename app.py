import gradio as gr
import os
from image_enhancer import AdvancedImageEnhancer
import cv2
from PIL import Image
import numpy as np

# Initialize the image enhancer
enhancer = AdvancedImageEnhancer()

def check_file_size(file):
    """Check if file size is under 10MB"""
    MAX_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    return os.path.getsize(file.name) <= MAX_SIZE

def process_image(input_image, output_format):
    """Process the image and return enhanced version"""
    if input_image is None:
        return None, "Please upload an image."
    
    # Convert to numpy array if needed
    if isinstance(input_image, str):
        input_image = cv2.imread(input_image)
    elif isinstance(input_image, np.ndarray):
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    try:
        # Apply enhancement
        enhanced = enhancer.denoise_image(input_image)
        enhanced = enhancer.adjust_gamma(enhanced)
        enhanced = enhancer.enhance_contrast(enhanced)
        enhanced = enhancer.sharpen_image(enhanced)
        enhanced = enhancer.deep_enhance(enhanced)
        
        # Convert back to RGB for display
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        
        # Create temporary file for download
        temp_path = f"temp_enhanced.{output_format.lower()}"
        if output_format.upper() == 'PNG':
            cv2.imwrite(temp_path, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:  # JPEG
            cv2.imwrite(temp_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        return enhanced_rgb, temp_path
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Create the Gradio interface
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🖼️ Advanced Image Enhancer
        
        Upload your image and get an enhanced version with superior quality!
        
        **Features:**
        - Advanced denoising
        - Adaptive gamma correction
        - Smart contrast enhancement
        - AI-powered enhancement
        - High-quality output
        
        **Note:** Maximum file size: 10MB
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    interactive=True
                )
                output_format = gr.Radio(
                    choices=["PNG", "JPEG"],
                    value="PNG",
                    label="Output Format",
                    info="Choose the format for the enhanced image"
                )
                enhance_btn = gr.Button("✨ Enhance Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(
                    label="Enhanced Image Preview",
                    type="numpy"
                )
                download_btn = gr.File(
                    label="Download Enhanced Image"
                )
        
        # Handle enhancement button click
        enhance_btn.click(
            fn=process_image,
            inputs=[input_image, output_format],
            outputs=[output_image, download_btn]
        )
        
        # Add file size warning
        input_image.upload(
            lambda x: gr.Warning("File size exceeds 10MB limit!") if not check_file_size(x) else None,
            inputs=[input_image]
        )

    return app

# For local testing
if __name__ == "__main__":
    app = create_ui()
    app.launch(
        inbrowser=True,  # Automatically open in browser
        server_port=None  # Let Gradio find an available port
    ) 