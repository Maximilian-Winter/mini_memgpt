from llama_cpp import Llama

from llama_cpp_agent.messages_formatter import MessagesFormatterType, MessagesFormatter
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

from mini_memgpt_agent import MiniMemGptAgent
main_model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")

SYS_PROMPT_START_NEURAL_CHAT = """### Instructions:\n"""
SYS_PROMPT_END_NEURAL_CHAT = """\n"""
USER_PROMPT_START_NEURAL_CHAT = """### User:\n"""
USER_PROMPT_END_NEURAL_CHAT = """ \n"""
ASSISTANT_PROMPT_START_NEURAL_CHAT = """### Agent:\n"""
ASSISTANT_PROMPT_END_NEURAL_CHAT = """\n"""
FUNCTION_PROMPT_START_NEURAL_CHAT = """### Function Response:\n"""
FUNCTION_PROMPT_END_NEURAL_CHAT = """\n"""
DEFAULT_NEURAL_CHAT_STOP_SEQUENCES = ["### User:"]


neural_chat_formatter = MessagesFormatter("", SYS_PROMPT_START_NEURAL_CHAT, SYS_PROMPT_END_NEURAL_CHAT,
                                          USER_PROMPT_START_NEURAL_CHAT,
                                          USER_PROMPT_END_NEURAL_CHAT, ASSISTANT_PROMPT_START_NEURAL_CHAT,
                                          ASSISTANT_PROMPT_END_NEURAL_CHAT, False, DEFAULT_NEURAL_CHAT_STOP_SEQUENCES, USE_USER_ROLE_FUNCTION_CALL_RESULT=False, FUNCTION_PROMPT_START=FUNCTION_PROMPT_START_NEURAL_CHAT, FUNCTION_PROMPT_END=FUNCTION_PROMPT_END_NEURAL_CHAT, STRIP_PROMPT=True)

llama_cpp_agent = MiniMemGptAgent(main_model, debug_output=True,
                                  # custom_messages_formatter=neural_chat_formatter,
                                  messages_formatter_type=MessagesFormatterType.CHATML
                                  )

while True:
    user_input = input(">")
    llama_cpp_agent.get_response(user_input)
