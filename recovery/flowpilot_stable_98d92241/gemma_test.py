
from openai import OpenAI

# 1. Initialize the client
# Replace <YOUR_WINDOWS_IP> with the actual IP address of the Windows machine
client = OpenAI(
    base_url="http://10.13.24.45:11434/v1",
    api_key="ollama" # Required by the library, but ignored by Ollama
)

# 2. Make the API call
response = client.chat.completions.create(
    model="gemma4:31b", # Must match the exact name of the model pulled in Ollama
    messages=[
        {"role": "system", "content": "You are a concise scientific assistant."},
        {"role": "user", "content": "What are the primary benefits of using Self-Driving Laboratories (SDLs) for catalyst synthesis? please explain to me in complete and full form. i want your responset to be complete, coherent and everything makes sense totally. use your full  brain and generate a really coherent results for me"}
    ],
    temperature=0.7,
   
)

# 3. Print the output
print(response.choices[0].message.content)



