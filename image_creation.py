import os
from openai import OpenAI

class Image:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key

        
    def create_from_scratch(self, prompt="A serene landscape with mountains during sunset",
                            model="dall-e-3",num_generations=1,
                            resolution="1024x1024", quality="standard"):
        """
            Generates an image based on the given prompt using the specified model.
        Args:
            prompt (str): A textual description of the image to be generated. Default is "A serene landscape with mountains during sunset".
            model (str): The name of the model to use for image generation. Default is "dall-e-3".
            num_generations (int): The number of image generations to produce. Default is 1.
            resolution (str): The resolution of the generated image. Default is "1024x1024".
            quality (str): The quality setting for the generated image. Default is "standard".
                -available only for dall-e-3 models
        Returns:
            picture: The generated image(s) links containing information.
        """
        if model=="dall-e-2":
            # Doesnt support quality.
            picture = self.client.images.generate(
                model=model,prompt=prompt,
                n=num_generations, size=resolution,
                )
            
        elif model == "dall-e-3":
            picture = self.client.images.generate(
                model=model,prompt=prompt,
                n=num_generations, size=resolution,
                quality=quality)
        return picture


    def create_variations_image(self, image_path, num_variations=1, resolution="1024x1024"):
        """
        Creates variations of the given image.
        
        Args:
            image_path (str): The path to the image file for which variations are to be created.
            num_variations (int): The number of variations to generate. Default is 1.
            resolution (str): The resolution of the generated image variations. Default is "1024x1024".
        
        Returns:
            picture: The generated image(s) links containing information.
        """
        with open(image_path, "rb") as image_file:
            picture = self.client.images.create_variation(
                model="dall-e-2", # model is hard-coded, no other models are supported.
                image=image_file,
                n=num_variations,
                size=resolution
            )
        return picture
        #picture = create_variations_image("D:/misc/opai/new-york-city-skyline.png")

    def manipulate(self,prompt="Add a plane to the sky that is visible by the angle",
                src_path="./", num_generations=1, resolution="1024x1024", mask=None):
        
        """
        Manipulates an image based on a given prompt using the DALL-E 2 model.
        Args:
            prompt (str): The description of the image to be generated. Defaults to "A serene landscape with mountains during sunset".
            src_path (str): The path to the source image file must be PNG. Defaults to "./".
            num_generations (int): The number of image variations to generate. Defaults to 1.
            resolution (str): The resolution of the generated image. Defaults to "1024x1024".
            mask (str, optional): The path to the mask image file. If None, no mask is used. Defaults to None.
        Returns:
            picture: The generated image(s) based on the provided prompt and parameters.

        https://platform.openai.com/docs/api-reference/images/createEdit#images-createedit-image
            -If no mask, image has to have some transparency.
        """    

        if mask:
            picture = self.client.images.edit(
                model="dall-e-2", # model is hard-coded, no other models are supported. 
                image=open(src_path, "rb"),
                mask=open(mask, "rb"),
                prompt=prompt,
                n=num_generations,
                size=resolution,
            )
            return picture
        
        picture = self.client.images.edit(
            model="dall-e-2", # model is hard-coded, no other models are supported. 
            image=open(src_path, "rb"),
            prompt=prompt,
            n=num_generations,
            size=resolution,
        )
        return picture

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "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"
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set") 
    
    imager = Image(api_key=api_key)
    # Example of creating an image from scratch
    response = imager.create_from_scratch(prompt="A futuristic cityscape at night", model="dall-e-3", num_generations=2)
    print("Create from scratch response:\n", response)

    # Example of creating variations of an existing image
    #response = imager.create_variations_image(image_path="D:/misc/opai/new-york-city-skyline.png", num_variations=3)
    #print("Create variations response:", response)

    # Example of manipulating an image
    #response = imager.manipulate(prompt="Add fireworks in the sky", src_path="D:/misc/opai/new-york-city-skyline.png", num_generations=1)
    #print("Manipulate image response:", response)
    #print(response)
