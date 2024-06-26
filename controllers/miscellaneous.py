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
    tokenizer = request.app.state.roberta_sentiment_tokenizer
    model = request.app.state.roberta_sentiment_model

    text = body.content

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    [negative_score, neutral_score, positive_score] = scores

    if negative_score>0.6:
        return {
            'flag': True
        }
    else:
        return {
            'flag': False
        }