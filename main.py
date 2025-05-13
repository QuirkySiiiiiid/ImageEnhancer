import app
import os

# Configure environment for Replit
os.environ['GRADIO_SERVER_NAME'] = "0.0.0.0"
os.environ['GRADIO_SERVER_PORT'] = "8080"

if __name__ == "__main__":
    # Create and launch the interface
    demo = app.create_ui()
    # Launch with Replit-specific settings
    demo.launch(
        server_name="0.0.0.0",  # Required for Replit
        server_port=8080,       # Standard Replit port
        share=False,            # No need for additional sharing link
        debug=True             # Show detailed errors
    ) 