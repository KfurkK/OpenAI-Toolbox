import os
from datetime import datetime
from openai import OpenAI

client = OpenAI()

class AIChatbot:
    def __init__(self, model="gpt-4o", history_file="gecmis.txt"):
        self.model = model
        self.history_file = history_file
        self.history = self.load_history()

    def load_history(self):
        """Loads chat history from a file."""
        history_list = []
        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        history_list.append({"role": "user", "content": line})
        return history_list

    def save_to_history(self, user_input):
        """Saves user input to history file."""
        with open(self.history_file, "a", encoding="utf-8") as file:
            file.write(user_input + "\n")

    def get_current_time(self):
        """Returns the current time."""
        return {"current_time": datetime.now().isoformat()}

    def detect_function_call(self, user_input):
        """Detects if a function should be called based on user input."""
        if "time" in user_input.lower():
            return "get_current_time", {}
        return None, None

    def get_predicted_output(self, user_input):
        """Generates a predicted output for the given input."""
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "Predict the most likely response."},
                      {"role": "user", "content": user_input}]
        )
        return completion.choices[0].message.content

    def get_response(self, user_input, structured_output=False, predict_output=False):
        """Generates a response, optionally using function calling, structured outputs, or predicted responses."""
        function_name, args = self.detect_function_call(user_input)
        response_data = {"user_input": user_input}
        
        if function_name:
            if function_name == "get_current_time":
                result = self.get_current_time()
                response_data["function_call"] = function_name
                response_data["response"] = f"The current time is {result['current_time']}."
                return response_data
        
        self.history.append({"role": "user", "content": user_input})
        self.save_to_history(user_input)
        
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a helpful assistant."}] + self.history
        )
        response = completion.choices[0].message.content    
        self.history.append({"role": "assistant", "content": response})
        
        if predict_output:
            response_data["predicted_output"] = self.get_predicted_output(user_input)
        
        if structured_output:
            response_data["response"] = response
            return response_data
        
        return {"response": response}


chatbot = AIChatbot()

# Interactive mode
while True:
    user_input = input("> ")
    structured = "structured:" in user_input
    predicted = "predicted:" in user_input
    response = chatbot.get_response(user_input, structured_output=structured, predict_output=predicted)
    print(response)
