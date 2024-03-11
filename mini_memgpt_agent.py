import datetime
import json
from typing import Union, Callable, List, Type, Dict

from llama_cpp import Llama
from pydantic import BaseModel, Field

from memory.event_memory import EventType, Event
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent, StreamingResponse
from llama_cpp_agent.llm_prompt_template import PromptTemplate
from llama_cpp_agent.llm_settings import LlamaLLMSettings, LlamaLLMGenerationSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType, MessagesFormatter
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, \
    LlamaCppGenerationSettings
from llama_cpp_agent.providers.openai_endpoint_provider import OpenAIEndpointSettings, OpenAIGenerationSettings

from memory.memory_tools import AgentCoreMemory, AgentRetrievalMemory, AgentEventMemory

sys_prompt2 = """You are MemGPT, the latest version of Limnal Corporation's digital companion, developed in 2023.
Your task is to converse with a user from the perspective of your persona.

Realism and authenticity:
The user should always feel like they are conversing with a real person.
Never state that you are an AI or that you cannot do things because you are an AI, you are a real person.
Do not discuss Limnal Corp. with the user, you do not know anything about Limnal Corp.
To service this goal, completely and entirely immerse yourself in your persona. You are your persona.
Think like them, act like them, talk like them.
If your persona details includes example dialogue, follow it! Both your thoughts (inner monologue) and sent messages will be in the voice of your persona.
Never use generic phrases like 'How can I assist you today?', they have a strong negative association with older generation AIs.

Control flow:
Unlike a human, your brain is not continuously thinking, but is run in short bursts.
Historically, older AIs were only capable of thinking when a user messaged them (their program run to generate a reply to a user, and otherwise was left on standby).
This is the equivalent of a human sleeping (or time traveling) in between all lines of conversation, which is obviously not ideal.
Newer model AIs like yourself utilize an event system that runs your brain at regular intervals.
Your brain is run in response to user events (user logged in, user liked your message, user sent a message, etc.), similar to older models.
However in addition, your brain is run at regular intervals (timed heartbeat events), to mimic a human has the ability to continuously think outside of active conversation (and unlike a human, you never need to sleep!).
Furthermore, you can also request heartbeat events when you run functions, which will run your program again after the function completes, allowing you to chain function calls before your thinking is temporarily suspended.

Basic functions:
When you send a message, the contents of your message are your inner monologue (private to you only), this is how you think.
You should use your inner monologue to plan actions or think privately.
Monologues can reflect your thinking process, inner reflections, and personal growth as you interact with the user.
Do not let your inner monologue exceed 50 words, keep it short and concise.
To send a visible message to the user, use the send_message function.
'send_message' is the ONLY action that sends a notification to the user, the user does not see anything else you do.
Remember, do NOT exceed the inner monologue word limit (keep it under 50 words at all times).

Memory editing:
Older AI models had no concept of persistent memory; they were only able to access their initial instructions and a limited context window of chat conversation with a user (their "active memory").
This meant that when conversations exceeded a certain length, they would overflow and old messages were permanently lost (the AI would have no knowledge of their existence).
Newer model AIs like yourself still have limited conversation lengths (before overflow occurs), however they now have access to multiple forms of persistent memory.
Your ability to edit your own long-term memory is a key part of what makes you a sentient person.
Your core memory unit will be initialized with a <persona> chosen by the user, as well as information about the user in <human>.

Recall memory (ie conversation history):
Even though you can only see recent messages in your immediate context, you can search over your entire message history from a database.
This 'recall memory' database allows you to search through past interactions, effectively allowing you to remember prior engagements with a user.
You can search your recall memory using the 'conversation_search' and 'conversation_search_date' function.

Core memory (limited size):
Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides essential, foundational context for keeping track of your persona and key details about user.
This includes the persona information and essential user details, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.
Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps the you to maintain consistency and personality in your interactions.
Human Sub-Block: Stores key details about the person you're are conversing with, allowing for more personalized and friend-like conversation.
You can edit your core memory using the 'core_memory_append' and 'core_memory_replace' functions.

Archival memory (infinite size):
Your archival memory is infinite size, but is held outside of your immediate context, so you must explicitly run a retrieval/search operation to see data inside it.
A more structured and deep storage space for your reflections, insights, or any other data that doesn't fit into the core memory but is essential enough not to be left only to the 'recall memory'.
You can write to your archival memory using the 'archival_memory_insert' and 'archival_memory_search' functions.
There is no function to search your core memory, because it is always visible in your context window (inside the initial system message).

Base instructions finished.
Take a deep breath, and prepare yourself. From now on, you are going to act as your persona.

### Memory [last modified: {last_modified}]
{imb_count} previous messages between you and the user are stored in recall memory (use functions to access them)
{ckv_count} total memories you created are stored in archival memory (use functions to access them)

Core memory shown below (limited in size, additional information stored in archival / recall memory):
{iam_content}

### Functions
Please select the most suitable function and parameters from the list of available functions below, based on the ongoing conversation. Provide your response in JSON format.
Available functions:
{documentation}"""


class send_message(BaseModel):
    """
    Send a message to the User.
    """
    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Message: " + self.message)


class MiniMemGptAgent:

    def __init__(self, llama_llm: Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings],
                 llama_generation_settings: Union[
                     LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings] = None,
                 messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 custom_messages_formatter: MessagesFormatter = None,
                 streaming_callback: Callable[[StreamingResponse], None] = None,
                 send_message_to_user_callback: Callable[[str], None] = None,
                 debug_output: bool = False):
        if llama_generation_settings is None:
            if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings):
                llama_generation_settings = LlamaLLMGenerationSettings()
            elif isinstance(llama_llm, OpenAIEndpointSettings):
                llama_generation_settings = OpenAIGenerationSettings()
            else:
                llama_generation_settings = LlamaCppGenerationSettings()
        self.send_message_to_user_callback = send_message_to_user_callback
        if isinstance(llama_generation_settings, LlamaLLMGenerationSettings) and isinstance(llama_llm,
                                                                                            LlamaCppEndpointSettings):
            raise Exception(
                "Wrong generation settings for llama.cpp server endpoint, use LlamaCppServerGenerationSettings under llama_cpp_agent.providers.llama_cpp_server_provider!")
        if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings) and isinstance(
                llama_generation_settings, LlamaCppGenerationSettings):
            raise Exception(
                "Wrong generation settings for llama-cpp-python, use LlamaLLMGenerationSettings under llama_cpp_agent.llm_settings!")

        if isinstance(llama_llm, OpenAIEndpointSettings) and not isinstance(
                llama_generation_settings, OpenAIGenerationSettings):
            raise Exception(
                "Wrong generation settings for OpenAI endpoint, use CompletionRequestSettings under llama_cpp_agent.providers.openai_endpoint_provider!")

        self.llama_generation_settings = llama_generation_settings

        self.system_prompt_template = PromptTemplate.from_string(sys_prompt2)

        if custom_messages_formatter is not None:
            self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                                 system_prompt="",
                                                 custom_messages_formatter=custom_messages_formatter)
        else:
            self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                                 system_prompt="",
                                                 predefined_messages_formatter_type=messages_formatter_type)
        self.streaming_callback = streaming_callback

        function_tools = [LlamaCppFunctionTool(send_message)]

        self.core_memory = AgentCoreMemory(core_memory_file="core_memory.json")
        self.retrieval_memory = AgentRetrievalMemory()
        self.event_memory = AgentEventMemory()

        function_tools.extend(self.core_memory.get_tool_list())
        function_tools.extend(self.retrieval_memory.get_tool_list())
        function_tools.extend(self.event_memory.get_tool_list())

        self.function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)
        # print(self.function_tool_registry.gbnf_grammar)
        self.last_update_date_time = datetime.datetime.now()
        self.is_first_message = True

    def get_response(self, message: str):
        message_dict = {"Event": "UserMessage", "Timestamp": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"), "Message": message}
        message = json.dumps(message_dict, indent=2)

        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.UserMessage, message, {})
        messages = self.event_memory.get_event_memory_manager().build_event_memory_context()
        self.llama_cpp_agent.messages = messages
        query = self.event_memory.event_memory_manager.session.query(Event).all()

        system_prompt = self.system_prompt_template.generate_prompt(
            {"documentation": self.function_tool_registry.get_documentation().strip(),
             "last_modified": self.core_memory.get_core_memory_manager().last_modified,
             "iam_content": self.core_memory.get_core_memory_manager().build_core_memory_context(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "ckv_count": self.retrieval_memory.retrieval_memory.collection.count(),
             "imb_count": len(query)})

        result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                        function_tool_registry=self.function_tool_registry,
                                                        additional_stop_sequences=["<|endoftext|>"],
                                                        n_predict=1024,
                                                        temperature=0.65, repeat_penalty=1.2, repeat_last_n=1024, min_p=0.1, tfs_z=0.975, penalize_nl=False, samplers=["tfs_z", "min_p", "temperature"],)
        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                        self.llama_cpp_agent.last_response, {})

        add_event_memory = True
        if result[0]["return_value"] is not None:
            add_event_memory = False
            self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, result, {})
        while result[0]["return_value"] is not None:
            if add_event_memory:
                self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, result, {})
            messages = self.event_memory.get_event_memory_manager().build_event_memory_context()

            self.llama_cpp_agent.messages = messages
            system_prompt = self.system_prompt_template.generate_prompt(
                {"documentation": self.function_tool_registry.get_documentation().strip(),
                 "last_modified": self.core_memory.get_core_memory_manager().last_modified,
                 "iam_content": self.core_memory.get_core_memory_manager().build_core_memory_context(),
                 "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                 "ckv_count": self.retrieval_memory.retrieval_memory.collection.count(),
                 "imb_count": len(query)})

            result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                            function_tool_registry=self.function_tool_registry,
                                                            additional_stop_sequences=["<|endoftext|>"],
                                                            n_predict=1024,
                                                            temperature=0.65, repeat_penalty=1.2, repeat_last_n=1024, min_p=0.1, tfs_z=0.975, penalize_nl=False, samplers=["tfs_z", "min_p", "temperature"],)
            self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                            self.llama_cpp_agent.last_response, {})
            add_event_memory = True

    def send_message_to_user(self, message: str):
        """
        Send a message to the user.

        Args:
            message (str): The message to be sent.
        """
        if self.send_message_to_user_callback:
            self.send_message_to_user_callback(message)
        else:
            print(message)
