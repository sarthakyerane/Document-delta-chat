import os
import sys

# Add the project root to the python path to import from src if needed, though not strictly required here
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai
from dotenv import load_dotenv

def main():
    # Load variables from .env file
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY is not set in your .env file.")
        print("Please add it like this: GOOGLE_API_KEY=\"AIzaSy...\"")
        sys.exit(1)

    print(f"Testing GOOGLE_API_KEY starting with: {api_key[:10]}...")

    try:
        genai.configure(api_key=api_key)
        # We use a very basic model just to verify the credentials work
        model = genai.GenerativeModel('gemini-3.5-flash')
        print("Sending a test ping to Google servers...")
        
        response = model.generate_content("Hello! Please reply with exactly: 'Your key is working!'")
        
        print("\n✅ SUCCESS! Your API key is fully valid and working.")
        print(f"Gemini says: {response.text.strip()}")
    except Exception as e:
        print(f"\n❌ FAILURE! Could not connect to Gemini API. Error:\n{e}")

if __name__ == "__main__":
    main()
