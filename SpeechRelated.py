import os
os.environ["OPENAI_API_KEY"] = "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"
import base64
from openai import OpenAI


class Audio:
    """
    Audio class for Speech-to-Text (STT) and Text-to-Speech (TTS) purposes.
    This class uses OpenAI's tts-1 and Whisper-v2-1 models for TTS and STT respectively.
    It also includes advanced methods for interaction with a chatbot using GPT-4o models.
    """

    def __init__(self, api_key):
        """
        Initialize the Audio class with the provided API key.

        :param api_key: The API key for OpenAI.
        """
        self.client = OpenAI(api_key=api_key)

    def tts(self, text, output_file, model="tts-1", voice="alloy"):
        """
        Convert text to speech and save the output to a file.

        :param text: The text to be converted to speech.
        :param output_file: The file where the speech output will be saved.
        :param model: The model to be used for TTS. Default is 'tts-1'.
        :param voice: The voice to be used for TTS. Default is 'alloy'.
        """
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        with open(output_file, "wb") as audio_file:
            audio_file.write(response.content)
        print(f"Find the file at: {output_file}")

    def stt(self, file, model="whisper-1"):
        """
        Convert speech from a file to text.

        :param file: The file containing the speech to be converted to text.
        :param model: The model to be used for STT. Default is 'whisper-1'.
        :return: The transcribed text.
        """
        with open(file, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file
            )
        return transcript.text

    def advanced_sft(self, text, model="gpt-4o-audio-preview", voice="alloy", output="output.wav"):
        """
        Perform Speech from Text (SFT) using a multimodal LLM response.
        -LLM response is in speech, input is in text.

        :param text: The input text for the chatbot.
        :param model: The model to be used for the chatbot. Default is 'gpt-4o-audio-preview'.
        :param voice: The voice to be used for the audio response. Default is 'alloy'.
        :param output: The file where the audio response will be saved. Default is 'output.wav'.
        """
        completion = self.client.chat.completions.create(
            model=model,
            modalities=["text", "audio"],
            audio={"voice": voice, "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
        print(f"Model's response completed. Find the file at: {output}")

        with open(output, "wb") as f:
            f.write(wav_bytes)

    def advanced_tfs(self, input_file="input_file.wav", model="gpt-4o-audio-preview", voice="alloy"):
        """
        Perform Text from Speech (TFS) using a multimodal LLM response.
        -LLM response is in text, input is in speech.
        
        :param input_file: The file containing the input speech.
        :param model: The model to be used for the chatbot. Default is 'gpt-4o-audio-preview'.
        :param voice: The voice to be used for the audio response. Default is 'alloy'.
        :return: The transcribed text from the audio response.
        """
        format = input_file.split(".")[-1]
        with open(input_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')

        completion = self.client.chat.completions.create(
            model=model,
            modalities=["text", "audio"],
            audio={"voice": voice, "format": format},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Only respond according to the contents of the recording."
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_string,
                                "format": format,
                            }
                        }
                    ]
                },
            ]
        )

        return completion.choices[0].message.audio.transcript

# Usage
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    os.makedirs("voiceovers", exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    output_path = os.path.join(script_dir, "voiceovers/output.mp3")  # Create full path

    audio = Audio(api_key=api_key)
    prompt = "Today is 11th February. And Elon Musk tried to buy OpenAI with a bid of 97 billion dollars."
    audio.tts(prompt, output_path)
    transcription = audio.stt("voiceovers/output.mp3")
    print(transcription)

    audio.advanced_sft("Hello there! Who are you?", output="responseFromAI.wav")
    response_from_ai = audio.advanced_tfs("voiceovers/output.mp3")

    print("\n\nResponse from LLM:\n" + response_from_ai)
