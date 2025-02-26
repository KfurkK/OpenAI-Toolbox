import os
os.environ["OPENAI_API_KEY"] = "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"
from openai import OpenAI
import base64
import requests


class ImageDescriber:
    """ImageDescriber class to describe images using OpenAI's GPT-4o, GPT-4o-mini and o1 models.
        Supports: PNG (.png), JPEG (.jpeg and .jpg), WEBP (.webp), and non-animated GIF (.gif).
    
    limitations:
        https://platform.openai.com/docs/guides/vision#limitations
    """
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.available_extensions = ["png", "jpeg", "jpg", "webp", "gif"]
    @staticmethod
    def is_url(data):
        return data.startswith(("http", "https"))

    @staticmethod
    def is_path(data):
        return os.path.exists(data)

    @staticmethod
    def encode_image(image_path):
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise ValueError(f"File not found: {image_path}")
        except Exception as e:
            raise ValueError(f"Error encoding image: {e}")

    def describe_image(self, data, max_tokens=300, model="gpt-4o" ,detail="auto", lang="tr"):
        """
        Describe an image using OpenAI's GPT-4o, GPT-4o-mini, or o1 model.
        :param data: Local path to the image or URL.
        :param max_tokens: Maximum number of tokens to generate. Set this high for o1 completions.
            -Even 1024 tokens may not be enough for some images.
        :param model: Model to use for the completion. Options: gpt-4o, gpt-4o-mini, o1.
        :param detail: Level of detail to provide. Options: low, medium, high, auto.
        :param lang: Language of the prompt. Options: en, tr.
        :return: Description of the image.
        """
        if self.is_path(data):
            extension = data.split(".")[-1] # Get the extension of the image to construct data
            if extension not in self.available_extensions:
                raise ValueError("Image must be in PNG, JPEG, WEBP, or non-animated GIF format")
            data = self.encode_image(data)
            data = f"data:image/{extension};base64,{data}"
        elif self.is_url(data):
            pass
        else:
            raise ValueError("Data must be a valid local path that contains '.png' or a URL: [https://example.com/image.png]")

        # Define parameters dynamically
        params = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Sana girdi olarak ulaştırılan bu resmi analiz edip bana ne anlattığını söyler misin?"} if lang == "tr" else "Can you describe this image?",
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data,
                                "detail": detail
                            },
                        },
                    ],
                }
            ]
        }

        # Add the correct token limit parameter based on the model
        token_param = "max_completion_tokens" if model == "o1" or "o1-mini" else "max_tokens"
        params[token_param] = max_tokens

        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except requests.RequestException as e:
            raise ValueError(f"Network error: {e}")
        except Exception as e:
            raise ValueError(f"Error during API call: {e}")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    img_data    = "https://im.haberturk.com/l/2025/02/08/ver1739018781/3763710/jpg/640x360"
    describer   = ImageDescriber(api_key)
    description = describer.describe_image(img_data, max_tokens=2048,model="o1", detail="auto", lang="tr")
    print(description)
