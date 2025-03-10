import google.generativeai as genai

# Replace with your actual API key
api_key = "AIzaSyCeWYwsdLuoue4tqdG-y3TyIKLloug0Q-s"

# Configure the API key
genai.configure(api_key=api_key)

# Initialize the text-bison model
model = genai.GenerativeModel('text-bison-001')  # Use a different model

# Generate a response
try:
    response = model.generate_content("Hello, world!")
    print("Response from text-bison:", response.text)
except Exception as e:
    print("Error:", e)