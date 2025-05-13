# Deploying Image Enhancer on Replit 🚀

## Step 1: Create New Replit Project

1. Go to [Replit](https://replit.com)
2. Click "Create Repl"
3. Choose "Python" as your template
4. Name your project (e.g., "image-enhancer")
5. Click "Create Repl"

## Step 2: Set Up Project Files

1. In the Files panel on the left, you'll need to create these files:
   - `main.py` (Replit's entry point)
   - `image_enhancer.py`
   - `app.py`
   - `requirements.txt`

2. Copy the contents of our files into each corresponding file in Replit:
   - Copy `image_enhancer.py` contents to `image_enhancer.py`
   - Copy `app.py` contents to `app.py`
   - Copy `requirements.txt` contents to `requirements.txt`

3. Create `main.py` with this content:
```python
import app

if __name__ == "__main__":
    demo = app.create_ui()
    demo.launch(server_name="0.0.0.0", server_port=8080)
```

## Step 3: Configure Replit

1. In the Replit shell, run:
```bash
pip install -r requirements.txt
```

2. Wait for all packages to install (this might take a few minutes)

## Step 4: Run the Application

1. Click the "Run" button at the top
2. Wait for the application to start
3. Replit will show you a web view of your application
4. Click "Open in new tab" for a better view

## Step 5: Using the Interface

1. Upload an image (remember 10MB limit)
2. Choose output format (PNG/JPEG)
3. Click "Enhance Image"
4. Download the enhanced result

## Troubleshooting Replit-Specific Issues

1. **If packages fail to install:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **If you get memory errors:**
   - Try with smaller images
   - Restart your Repl
   - Use the "Boost" feature in Replit (if available)

3. **If the interface doesn't load:**
   - Check the console for errors
   - Make sure port 8080 is specified in main.py
   - Try refreshing the page

## Important Notes for Replit Usage

1. **Resource Limitations:**
   - Free tier has limited RAM
   - Processing might be slower than local
   - GPU acceleration is not available

2. **Persistence:**
   - Temporary files are cleared on Repl restart
   - Save important results immediately

3. **URL Access:**
   - Your Replit URL will be public
   - Share the URL with others to let them use your enhancer

## Best Practices

1. **For Better Performance:**
   - Use smaller images when possible
   - Close unused Repls
   - Clear temporary files regularly

2. **For Reliability:**
   - Keep your Repl running
   - Use "Always On" feature if available
   - Monitor console for errors 