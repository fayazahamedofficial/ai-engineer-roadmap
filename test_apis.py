"""Quick sanity check that both API keys work. Run once, then move on."""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables

# --- Gemini ---
from google import genai

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
gemini_response = gemini_client.models.generate_content(
    model="gemini-flash-latest",
    contents="Say 'Gemini is working' and nothing else.",
)
print("GEMINI:", gemini_response.text.strip())

# --- Groq ---
from groq import Groq

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
groq_response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say 'Groq is working' and nothing else."}],
)
print("GROQ:", groq_response.choices[0].message.content.strip())