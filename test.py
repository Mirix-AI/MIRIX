from mirix.agent import AgentWrapper

agent = AgentWrapper("mirix/configs/mirix_custom_model.yaml")

response = agent.send_message(message="Hello, how are you?", memorizing=False)