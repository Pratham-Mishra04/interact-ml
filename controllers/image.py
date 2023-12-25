from PIL import Image
import numpy as np
from io import BytesIO
import base64
from blurhash import encode

def generate_blurhash_data_url(image_file):
    try:
        # Read the content of the UploadFile
        image_content = image_file.file.read()

        # Create a BytesIO object from the content
        image_content_io = BytesIO(image_content)

        img = Image.open(image_content_io)

        # Determine the x and y component counts based on the original image size
        x_components = min(max(img.width // 4, 1), 9)
        y_components = min(max(img.height // 4, 1), 9)

        # Resize the image to a smaller size
        img = img.resize((x_components * 4, y_components * 4))

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Ensure the image is a 3-dimensional array
        if img_array.ndim != 3:
            raise ValueError("Invalid image format")

        # Encode BlurHash
        blurhash = encode(img_array, x_components, y_components)

        # Convert the image to WebP format
        webp_data = BytesIO()
        img.save(webp_data, format='WebP')
        webp_data.seek(0)

        # Encode the WebP data to base64
        base64_data = base64.b64encode(webp_data.read()).decode('utf-8')

        # Build the data URL
        data_url = f'data:image/webp;base64,{base64_data}'

        return {
            'status': 'success',
            'message': '',
            'data_url': data_url,
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error generating BlurHash data URL: {str(e)}",
        }
