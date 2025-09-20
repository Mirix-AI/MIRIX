import base64
import io

# Convert images to base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def encode_image_from_pil(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # Change format if needed (JPEG, etc.)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")