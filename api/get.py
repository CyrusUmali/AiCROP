import google.generativeai as genai
genai.configure(api_key="AIzaSyCWiZmhjdh1GmYKnvJMLvgsY-bh20wYOZs")

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)