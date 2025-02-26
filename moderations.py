import os
import base64
from openai import OpenAI
from urllib.parse import urlparse

class Moderation:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

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

    @staticmethod
    def is_url(string):
        """
        Check if the string is a valid URL.
        """
        try:
            
            result = urlparse(string)
            return all([result.scheme, result.netloc])  # Checks if it's a complete URL (e.g., https://www.example.com)
        except ValueError:
            return False

    @staticmethod
    def is_valid_path(string):
        """
        Check if the string is a valid file path, and also perform validation for file existence.
        Returns True if path is valid, or raises ValueError if the path is invalid.
        """
        if not isinstance(string, str) or not string.strip():
            raise ValueError(f"Invalid path format: {string}")
        
        # Check if path is absolute or relative
        if os.path.isabs(string):
            if not os.path.exists(string):
                raise ValueError(f"Couldn't find the file at absolute location: {string}")
        else:
            if not os.path.exists(string):
                raise ValueError(f"Couldn't find the file at relative location: {string}")
        
        return True  # Valid path or file exists

    def is_safe(self, prompt: str, image: str):
        """
        Accepts both text and image at the same time.
        text is string, image is either a local path or a URL.
        """

        # Check if the image is a URL
        if self.is_url(image):
            pass
        
        elif self.is_valid_path(image):
            image = self.encode_image(image_path=image)
            image = f"data:image/jpeg;base64,{image}"

        else:
            raise ValueError(f"Unknown type of image data: {type(image)}")

        response = self.client.moderations.create(
            model="omni-moderation-latest",
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                    }
                },
            ],
        )

        flagged = response.results[0].flagged
        if flagged:
            scores = response.results[0].category_scores
            # Convert the custom object to a dictionary:
            scores_dict = vars(scores)  
            highest_category = max(scores_dict, key=lambda k: scores_dict[k])
            highest_score = scores_dict[highest_category]
            print("Flagged! With highest score of {:.3f} at the category {}".format(highest_score, highest_category))
            return False

        else:
            print("Both image and prompt were clear!")
            return True

# Usage
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    image_link = ""
    prompt = ""
    guard = Moderation(api_key)
    print(guard.is_safe(image=image_link, prompt=prompt))
