# Advanced Image Enhancer 🖼️

This is a high-efficiency image enhancement tool that combines multiple advanced techniques to improve image quality. The enhancer uses a combination of traditional image processing methods and deep learning to achieve superior results, now with a user-friendly web interface!

## ✨ Features

- Advanced denoising using Non-Local Means algorithm
- Adaptive gamma correction
- Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Intelligent sharpening
- Deep learning-based enhancement using a custom neural network
- Support for both CPU and GPU processing
- User-friendly web interface
- High-quality PNG/JPEG output options
- Maximum file size limit: 10MB

## 🛠️ Requirements

- Python 3.7+
- CUDA-capable GPU (optional, for faster processing)
- Required packages (will be installed automatically)

## 🚀 Usage Options

### Method 1: Using Google Colab (Recommended for Beginners)

1. Open our Colab notebook by clicking this link: [Image Enhancer Colab](https://colab.research.google.com/github/YOUR_USERNAME/image-enhancer/blob/main/image_enhancer.ipynb)
2. Click "Runtime" → "Run all" or run each cell in sequence
3. Wait for the setup to complete
4. Click the generated public URL to access the web interface
5. Upload your image (max 10MB)
6. Choose output format (PNG/JPEG)
7. Click "Enhance Image"
8. Download the enhanced image

### Method 2: Local Installation (For Advanced Users)

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/image-enhancer.git
cd image-enhancer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the web interface:
```bash
python app.py
```

4. Open your browser and go to: `http://localhost:7860`

### Method 3: Direct Code Usage (Without Cloning)

1. Create a new directory for your project:
```bash
mkdir image_enhancer
cd image_enhancer
```

2. Create the following files with their contents:

- Create `requirements.txt`:
```bash
echo "numpy>=1.21.0
opencv-python>=4.5.3
pillow>=8.3.1
scikit-image>=0.18.3
tensorflow>=2.8.0
torch>=1.9.0
torchvision>=0.10.0
gradio>=4.0.0" > requirements.txt
```

3. Download the necessary Python files:
```bash
# Download the files
curl -O https://raw.githubusercontent.com/YOUR_USERNAME/image-enhancer/main/image_enhancer.py
curl -O https://raw.githubusercontent.com/YOUR_USERNAME/image-enhancer/main/app.py
```

4. Install requirements:
```bash
pip install -r requirements.txt
```

5. Run the web interface:
```bash
python app.py
```

6. Open your browser and go to: `http://localhost:7860`

## 💻 Using the Web Interface

1. **Upload Image**
   - Click the upload button or drag & drop your image
   - Maximum file size: 10MB
   - Supported formats: JPG, PNG, JPEG

2. **Choose Output Format**
   - PNG: Best for high quality, lossless output
   - JPEG: Smaller file size, slight quality loss

3. **Enhance Image**
   - Click the "✨ Enhance Image" button
   - Wait for processing to complete
   - Preview will appear on the right

4. **Download**
   - Click the download button below the preview
   - Enhanced image will be saved in your chosen format

## 🔧 Customization

You can modify the enhancement parameters in `image_enhancer.py`:
- Adjust gamma value in `adjust_gamma()`
- Modify CLAHE parameters in `enhance_contrast()`
- Change sharpening kernel in `sharpen_image()`
- Customize the neural network architecture in `ImageEnhancementNetwork`

## 📝 Notes

- For best results, ensure input images are in common formats (JPG, PNG)
- Very large images may require more processing time
- GPU acceleration will be used automatically if available
- The web interface can be accessed from any device on your network
- Processing time depends on image size and your hardware

## 🐛 Troubleshooting

1. **Image Upload Issues**
   - Ensure file is under 10MB
   - Check if file format is supported
   - Try converting to JPG/PNG if using other formats

2. **Processing Errors**
   - Check if all requirements are installed correctly
   - Ensure you have enough RAM available
   - Try reducing image size if too large

3. **Performance Issues**
   - Enable GPU acceleration if available
   - Close other resource-intensive applications
   - Try processing smaller images first

## 📫 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Ensure you provide:
   - Error messages
   - Python version
   - Operating system
   - Hardware specifications (CPU/GPU) 