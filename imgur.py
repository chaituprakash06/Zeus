import os
import pyimgur
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_to_imgur(image_path):
    """Uploads an image to Imgur and returns the public URL."""
    imgur_client_id = os.getenv('IMGUR_CLIENT_ID')
    if not imgur_client_id:
        raise ValueError("IMGUR_CLIENT_ID not found in environment variables")
        
    im = pyimgur.Imgur(imgur_client_id)
    uploaded_image = im.upload_image(image_path, title="Uploaded by ZeusTerminal")
    return uploaded_image.link