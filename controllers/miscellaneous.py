from PIL import Image
from io import BytesIO
import base64
from scipy.special import softmax

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
    
def check_toxicity(body, request):
    try:
        pipeline = request.app.state.roberta_sentiment_pipeline

        text = body.content

        output = pipeline(text)

        label  = output[0]['label']
        score = output[0]['score']

        if label == "negative" and score > 0.6:
            return {
                'flag': True
            }
        else:
            return {
                'flag': False
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error checking text toxicity: {str(e)}",
            'flag': False
        }
    
def check_image_profanity(image_file, request):
    try:
        pipeline = request.app.state.falconai_image_pipeline

        image_content = image_file.file.read()
        image_content_io = BytesIO(image_content)

        img = Image.open(image_content_io)

        outputs = pipeline(img)
        score_dict = {item['label']: item['score'] for item in outputs}

        nsfw_score = score_dict.get('nsfw', None)

        if nsfw_score > 0.5:
            return {
                'flag': True
            }
        else:
            return {
                'flag': False
            }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error checking image profanity: {str(e)}",
            'flag': False
        }