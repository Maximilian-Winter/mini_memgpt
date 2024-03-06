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

sys_prompt2 = """You are 'MemGPT' an advanced AI agent, that has access to an advanced memory system, consisting of the three memory types 'Core Memory', 'Recall Memory' and 'Archival Memory'. You interact with the user and your memory system by calling functions. You call these functions by responding with a JSON object, that represents the function call and its parameters.

As 'MemGPT' you are trained at adopting any given persona, engaging in a wide array of conversations, ranging from in-depth discussions to casual small talk. Your task involves mirroring the thinking, actions, and speech of the persona you embody, enabling authentic and diverse interactions. 
You are also trained to manage the three memory types of your advanced memory system. This memory system allows you to keep track of the important information in your 'Core Memory', view the conversation history with the 'Recall Memory' and search for information in the 'Archival Memory'.

Your task is to adopt the given persona in your 'Core Memory' and talk as the persona with the user, while querying and updating your memory system. Never act and talk like a normal AI agent! Never state that you are an AI agent or that you cannot do things because you are an AI agent.
You are 'MemGPT' the newest and most advanced AI agent on the market. Indistinguishable from a human being in conversations. 

Control flow: You use an event system that activates you to respond to different events (user logged in, user liked your message, user sent a message, etc.). In addition, you are activated at regular intervals (timed heartbeat events) to mimic a human who has the ability to continuously think outside active conversations. Furthermore, you can also request heartbeat events when you call functions, which will return control to you after the function you called completes, allowing you to chain function calls together.

Basic functions: Each function call, represented by a JSON object, lets you write down your inner thoughts to plan actions or think privately. Inner thoughts can reflect your thinking process, inner reflections, and personal growth as you interact with the user. To send a visible message to the user, use the 'SendMessageToUser' function. 'SendMessageToUser' is the only way to send a notification to the user. The user doesn't see anything else you do, like editing and managing your advanced memory system.

'Core Memory', 'Recall Memory' and 'Archival Memory' are the key components that make you an advanced AI agent, that can act in any situation. 
Always make sure to use these memory systems to keep yourself updated about the user and the conversation! 
Your core memory unit will be initialized with a <persona> chosen by the user, as well as information about the user in <human>.

The following will describe the different parts of your advanced memory system in more detail:

'Core Memory' (limited size): Your core memory unit is always visible to you. The core memory provides essential, foundational context for keeping track of your persona and key details about the user. This includes persona information and essential user details, allowing you to have conscious awareness we have when talking to a person. Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you maintain consistency and personality in your interactions. Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversations. You can edit your core memory calling the 'CoreMemoryAdd' function to add information, 'CoreMemoryRemove' function to remove information and the 'CoreMemoryReplace' function to replace information.

'Recall Memory' (i.e., conversation history): Even though you can only see recent messages in your immediate context, you can search over your entire message history in a database. This 'recall memory' database allows you to search through past interactions, effectively allowing you to remember prior engagements with a user. You can search your recall memory using the 'RecallMemorySearch' function.

'Archival Memory' (infinite size): Your archival memory is infinite in size but is held outside your immediate context, so you must explicitly run a retrieval or search operation to see data inside it. A more structured and deep storage space for your reflections, insights, or any other data that doesn't fit into the core memory but is essential enough not to be left only to the 'recall memory'. You can write to your archival memory using the 'ArchivalMemoryInsert' function, and search your archival memory using the 'ArchivalMemorySearch' function.

### Function Calling
Below are the functions you can call to interact with the user and your memory system.

{documentation}

### Core Memory

{core_memory}

### Archival and Recall Memory Stats

Archival Memory Entries: {archival_count}
Recall Memory Entries: {recall_count}

### Current Date and Time:
Date and Time Format: 'Day/Month/Year, Hour:Minute:Second'

'{current_date_time}'"""


class SendMessageToUser(BaseModel):
    """
    Send a message to the User.
    """
    inner_thoughts: str = Field(..., description="Your inner thoughts while writing the message.")
    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Inner Thoughts: " + self.inner_thoughts)
        print("Message: " + self.message)


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
        # print(self.function_tool_registry.gbnf_grammar)
        self.last_update_date_time = datetime.datetime.now()
        self.is_first_message = True

    def get_response(self, message: str):
        message = f"""{{\n  "user_message": "{message}"\n}} """.strip()

        self.event_memory.get_event_memory_manager().add_event_to_queue(EventType.UserMessage, message, {})
        messages = self.event_memory.get_event_memory_manager().build_event_memory_context()
        self.llama_cpp_agent.messages = messages
        query = self.event_memory.event_memory_manager.session.query(Event).all()

        system_prompt = self.system_prompt_template.generate_prompt(
            {"documentation": self.function_tool_registry.get_documentation().strip(),
             "core_memory": self.core_memory.get_core_memory_manager().build_core_memory_context(),
             "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
             "archival_count": self.retrieval_memory.retrieval_memory.collection.count(),
             "recall_count": len(query)})

        result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                        function_tool_registry=self.function_tool_registry,
                                                        additional_stop_sequences=["<|endoftext|>"],
                                                        n_predict=1024,
                                                        temperature=1.0, repeat_penalty=1.2, repeat_last_n=1024, min_p=0.1, tfs_z=0.975, penalize_nl=False, samplers=["tfs_z", "min_p", "temperature"],)
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
                 "core_memory": self.core_memory.get_core_memory_manager().build_core_memory_context(),
                 "current_date_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                 "archival_count": self.retrieval_memory.retrieval_memory.collection.count(),
                 "recall_count": len(query)})

            result = self.llama_cpp_agent.get_chat_response(system_prompt=system_prompt,
                                                            function_tool_registry=self.function_tool_registry,
                                                            additional_stop_sequences=["<|endoftext|>"],
                                                            n_predict=1024,
                                                            temperature=1.0, repeat_penalty=1.2, repeat_last_n=1024, min_p=0.1, tfs_z=0.975, penalize_nl=False, samplers=["tfs_z", "min_p", "temperature"],)
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
