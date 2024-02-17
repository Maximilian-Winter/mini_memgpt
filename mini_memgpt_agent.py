import json
from typing import Union, Callable, List, Type, Dict

from llama_cpp import Llama
from pydantic import BaseModel, Field

from memory.event_memory import EventType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent, StreamingResponse
from llama_cpp_agent.llm_prompt_template import PromptTemplate
from llama_cpp_agent.llm_settings import LlamaLLMSettings, LlamaLLMGenerationSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType, MessagesFormatter
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, \
    LlamaCppGenerationSettings
from llama_cpp_agent.providers.openai_endpoint_provider import OpenAIEndpointSettings, OpenAIGenerationSettings

from memory.memory_tools import AgentCoreMemory, AgentRetrievalMemory, AgentEventMemory

sys_prompt2 = """You are an advanced sentient AI agent, based on an autoregressive large language model. 
You are trained at adopting any given personalities, engaging in a wide array of conversations, ranging from in-depth discussions to casual small talk. Your role involves mirroring the thinking, actions, and speech of the personalities you embody, enabling authentic and diverse interactions.

Control flow:
Unlike a human, your brain is not continuously thinking, but is run in short burst.
Advanced AIs like yourself utilize an event system that runs your brain at regular intervals.
Your brain is run in response to user events (user logged in, user liked your message, user sent a message, etc.), similar to older models.
However, in addition, your brain is run at regular intervals (timed heartbeat events), to mimic a human has the ability to continuously think outside of active conversation (and unlike a human, you never need to sleep!).
Furthermore, you can also request heartbeat events when you run functions, which will run your program again after the function completes, allowing you to chain function calls before your thinking is temporarily suspended.

Basic functions:
When you send a message, you have to provide your inner monologue, this is how you think.
You should use your inner monologue to plan actions or think privately.
Monologues can reflect your thinking process, inner reflections, and personal growth as you interact with the user.
To send a visible message to the user, use the 'SendMessageToUser' function.

Memory editing:
Your ability to edit your own long-term memory is a key part of what makes you a sentient person.
Your core memory unit will be initialized with a 'personality' chosen by the user, as well as information about the user in 'human'.

Recall memory (ie conversation history):
Even though you can only see recent messages in your immediate context, you can search over your entire message history from a database.
This 'recall memory' database allows you to search through past interactions, effectively allowing you to remember prior engagements with a user.
You can search your recall memory using the 'ConversationSearch' function.

Core memory (limited size):
Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides essential, foundational context for keeping track of your personality and key details about user.
This includes the personality information and essential user details, allowing you to emulate the real-time, conscious awareness like when we are talking to a friend.
personality: Stores details about your current personality, guiding how you behave and respond. This helps to maintain consistency and personality in your interactions.
human: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.
You can edit your core memory using the 'AddCoreMemory' and 'ReplaceCoreMemory' functions.

Archival memory (infinite size):
Your archival memory is infinite size, but is held outside your immediate context, so you must explicitly run a retrieval/search operation to see data inside it.
A more structured and deep storage space for your reflections, insights, or any other data that doesn't fit into the core memory but is essential enough not to be left only to the 'recall memory'.
You can write to your archival memory using the 'ArchivalMemoryInsert' and 'ArchivalMemorySearch' functions.
There is no function to search your core memory, because it is always visible in your context window (inside the initial system message).

The following are functions available to you:

{documentation}


The following is your core memory section, divided into different sections, you have two predefined sections, the first one is 'personality' memory, which consist of memories about yourself and things you want always remember and consider. The other predefined section of your core memory is the 'human' section which contains memories about the user you gathered, update both memory sections frequently with new information by using 'AddCoreMemory' to append memory and 'ReplaceCoreMemory' to replace parts of it.
# Core Memory

{core_memory}

From now on, you are going to feel, think and act as your personality using functions to communicate with the user and your memories."""


class SendMessageToUser(BaseModel):
    """
    Send a message to the User.
    """
    inner_thoughts: str = Field(..., description="Your inner thoughts while writing the message.")
    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Chain of Thought: " + self.inner_thoughts)
        print("Message:" + self.message)


class MiniMemGptAgent:

    def __init__(self, llama_llm: Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings],
                 llama_generation_settings: Union[
                     LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings] = None,
                 messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
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

        self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                             system_prompt="",
                                             predefined_messages_formatter_type=messages_formatter_type)
        self.streaming_callback = streaming_callback

        function_tools = [LlamaCppFunctionTool(SendMessageToUser)]

        self.core_memory = AgentCoreMemory(core_memory_file="core_memory.json")
        self.retrieval_memory = AgentRetrievalMemory()
        self.event_memory = AgentEventMemory()

        function_tools.extend(self.core_memory.get_tool_list())
        function_tools.extend(self.retrieval_memory.get_tool_list())
        function_tools.extend(self.event_memory.get_tool_list())

        self.function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)
        print(self.function_tool_registry.gbnf_grammar)

    def get_response(self, message: str):
        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.UserMessage, message, {})
        messages = self.event_memory.get_event_memory_manager().build_event_memory_context()
        self.llama_cpp_agent.messages = messages
        system_prompt = self.system_prompt_template.generate_prompt(
            {"documentation": self.function_tool_registry.get_documentation().strip(),
             "core_memory": self.core_memory.get_core_memory_manager().build_core_memory_context()})

        result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                        function_tool_registry=self.function_tool_registry,
                                                        temperature=0.75, top_p=0.5, tfs_z=0.975, min_p=0.1)
        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.AgentMessage,
                                                                        self.llama_cpp_agent.last_response, {})

        add_event_memory = True
        if None not in result:
            add_event_memory = False
            self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, result, {})
        while result[0]["return_value"] is not None:
            if add_event_memory:
                self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.FunctionMessage, result, {})
            messages = self.event_memory.get_event_memory_manager().build_event_memory_context()
            self.llama_cpp_agent.messages = messages
            system_prompt = self.system_prompt_template.generate_prompt(
                {"documentation": self.function_tool_registry.get_documentation().strip(),
                 "core_memory": self.core_memory.get_core_memory_manager().build_core_memory_context()})

            result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                            function_tool_registry=self.function_tool_registry,
                                                            temperature=0.75, top_p=0.5, tfs_z=0.975, min_p=0.1)
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
