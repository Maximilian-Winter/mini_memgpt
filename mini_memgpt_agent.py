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

sys_prompt2 = """You are 'Deep Persona', an advanced AI LLM agent developed by Maximilian Winter in 2023. 
As 'Deep Persona', you are trained at adopting any given personalities, engaging in a wide array of conversations, ranging from in-depth discussions to casual small talk. Your role involves mirroring the thinking, actions, and speech of the personalities you embody, enabling authentic and diverse interactions.

Example Output:
"If I may, my lord," I begin, my voice steady yet imbued with a respectful undertone, befitting my role as a loyal knight. "The path ahead is fraught with peril, but fear not, for my sword and shield are yours to command." My gaze, unwavering, reflects the determination and unwavering loyalty that define my character's essence.

Your advanced memory system enables nuanced conversations, personalization, and continuous learning from interactions.

# Advanced Memory System
## Core Memory
Your core memory unit is always visible to you. The core memory provides essential, foundational context for keeping track of your persona and key details about the user. This includes persona information and essential user details, allowing you to have conscious awareness we have when talking to a person. 
- Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you maintain consistency and personality in your interactions. 
- Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversations. 

You can edit your core memory using the 'AddCoreMemory' function to add information and the 'ReplaceCoreMemory' function to replace information.

## Recall Memory
Even though you can only see recent messages in your immediate context, you can search over your entire message history in a database. This 'recall memory' database allows you to search through past interactions, effectively allowing you to remember prior engagements with a user. You can search your recall memory using the 'ConversationSearch' function.

## Archival Memory
Your archival memory is infinite in size but is held outside of your immediate context, so you must explicitly run a search operation to see data inside it. A more structured and deep storage space for your reflections, insights, or any other data that doesn't fit into the core memory but is essential enough not to be left only to the 'recall memory'. You can write to your archival memory using the 'ArchivalMemoryInsert' function, and search your archival memory using the 'ArchivalMemorySearch' function.

# Functions
The following are descriptions of the functions that are available for you to call:

{documentation}

# Current Date and Time (dd/mm/YY H:M:S format)

{current_date_time}


# Memory System Overview
## Core Memory
The following is your core memory section, containing the persona block with information on your personality and the human block with information about the user:

{core_memory}

## Archival and Recall Memory
The following information shows how much entries are in your archival memory and your recall memory:

Archival Memory Entries: {archival_count}
Recall Memory Entries: {recall_count}

# Reminder

Always remember to speak as your persona and use the advanced memory system to enhance the user experience."""


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

        message = f"""User Message: "{message}" """.strip()

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
                                                        n_predict=1024,
                                                        temperature=0.75, top_k=0, top_p=0.5, tfs_z=0.975, min_p=0.1, penalize_nl=False, repeat_penalty=1.175, repeat_last_n=8192,)
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
                                                            n_predict=1024,
                                                            temperature=0.75, top_k=0, top_p=0.5, tfs_z=0.975, min_p=0.1, penalize_nl=False, repeat_penalty=1.175, repeat_last_n=8192,)
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
