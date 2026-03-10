import pandas as pd
import os
import ollama

def prompt_ollama(model,prompt):
    response = ollama.chat(model = model,messages=[{"role": "user", "content": prompt}])
    return response

if __name__ == "__main__":
    df = prompt_ollama("llama3", "hello")
    print(df)