from PIL import Image
from io import BytesIO
import base64

def generate_blurhash_data_url(image_file):
    try:
        image_content = image_file.file.read()

        image_content_io = BytesIO(image_content)

        img = Image.open(image_content_io)

        # Determine the x and y component counts based on the original image size
        x_components = min(max(img.width // 4, 1), 9)
        y_components = min(max(img.height // 4, 1), 9)

        img = img.resize((x_components * 4, y_components * 4))

        # Convert the image to WebP format
        webp_data = BytesIO()
        img.save(webp_data, format='WebP')
        webp_data.seek(0)

        # Encode the WebP data to base64
        base64_data = base64.b64encode(webp_data.read()).decode('utf-8')

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
