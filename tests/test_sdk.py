import os
import sys
import dotenv

# Add the project root to Python path so we can import mirix
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from mirix import Mirix

from dotenv import load_dotenv
load_dotenv()

# Initialize memory agent (defaults to Google Gemini 2.0 Flash)
memory_agent = Mirix(api_key=os.getenv("GEMINI_API_KEY"))

# Add memories
memory_agent.add("The moon now has a president")
memory_agent.add("John loves Italian food and is allergic to peanuts")

# Chat with memory context
response = memory_agent.chat("Does the moon have a president?")
print(response)  # "Yes, according to my memory, the moon has a president."

response = memory_agent.chat("What does John like to eat?") 
print(response)  # "John loves Italian food. However, he's allergic to peanuts."

### List all users
users = memory_agent.list_users()
assert users[0].name == "default_user"
default_user = users[0]

### Now we need to test the multi-user functionality
user = memory_agent.create_user(user_name="Alice")
memory_agent.add("John likes to eat British food.", user_id=user.id)
response = memory_agent.chat("What does John like to eat?", user_id=user.id)
print("Response for Alice:", response)

### Then send the message to the default user
response = memory_agent.chat("What does John like to eat?", user_id=default_user.id)
print("Response for Default User:",response)

### Test duplicate user creation
original_length = len(memory_agent.list_users())
user = memory_agent.create_user(user_name="Alice")
new_length = len(memory_agent.list_users())
assert new_length == original_length



