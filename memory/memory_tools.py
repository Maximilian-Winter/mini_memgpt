from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from .event_memory import EventType, Base
from .event_memory_manager import EventMemoryManager
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from .core_memory_manager import CoreMemoryManager
from .retrieval_memory_manager import RetrievalMemoryManager, RetrievalMemory


class CoreMemoryKey(Enum):
    PERSONA: str = "persona"
    HUMAN: str = "human"


class immediate_access_memory_add(BaseModel):
    """
    Add a new memory to the Immediate Access Memory (IAM).
    """

    key: str = Field(..., description="The key identifier of the Immediate Access Memory (IAM) entry.")
    field: str = Field(..., description="A secondary key or field within the Immediate Access Memory (IAM) entry.")
    value: str = Field(..., description="The value or data to be stored in the specified Immediate Access Memory (IAM) entry.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.add_to_core_memory(self.key, self.field, self.value)


# Replace Immediate Access Memory (IAM) Model
class immediate_access_memory_replace(BaseModel):
    """
    Replace a memory in the Immediate Access Memory (IAM).
    """

    key: str = Field(..., description="The key identifier of the Immediate Access Memory (IAM) entry.")
    field: str = Field(..., description="The specific field within the Immediate Access Memory (IAM) entry to be replaced.")
    new_value: str = Field(...,
                           description="The new value to replace the existing data in the specified Immediate Access Memory (IAM) field.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.replace_in_core_memory(self.key, self.field, self.value)


# Remove Immediate Access Memory (IAM) Model
class immediate_access_memory_remove(BaseModel):
    """
    Remove a memory field from the Immediate Access Memory (IAM).
    """

    key: str = Field(..., description="The key identifier of the Immediate Access Memory (IAM) entry to remove.")
    field: str = Field(..., description="The specific field within the Immediate Access Memory (IAM) entry to be removed.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.remove_from_core_memory(self.key, self.field)


class interactive_memory_bank_Search(BaseModel):
    """
    Search for memories from the Interactive Memory Bank (IMB).
    """

    event_types: Optional[list[EventType]] = Field(...,
                                                   description="Memory type to search. Can be 'system', 'user', 'assistant' or 'function'")
    start_date: Optional[str] = Field(...,
                                      description='Start date to search memories from. Format: "dd/mm/YY, H:M:S" eg. "01/01/2024, 08:00:30"')
    end_date: Optional[str] = Field(...,
                                    description='End date to search memories from. Format: "dd/mm/YY, H:M:S" eg. "04/02/2024, 18:57:29"')
    keywords: Optional[List[str]] = Field(...,
                                          description='End date to search memories from. Format: "dd/mm/YY, H:M:S" eg. "08/04/2023, 12:32:30"')

    def run(self, event_memory_manager: EventMemoryManager):
        parsed_start_datetime = None
        parsed_end_datetime = None
        if self.start_date:
            parsed_start_datetime = datetime.strptime(self.start_date, "%d/%m/%Y, %H:%M:%S")
        if self.end_date:
            parsed_end_datetime = datetime.strptime(self.end_date, "%d/%m/%Y, %H:%M:%S")

        return event_memory_manager.query_events(event_types=self.event_types, keywords=self.keywords,
                                                 start_date=parsed_start_datetime, end_date=parsed_end_datetime)


class conceptual_knowledge_vault_search(BaseModel):
    """
    Retrieve information from the Conceptual Knowledge Vault (CKV).
    """

    query: str = Field(..., description="Query to be used to retrieve information from the Conceptual Knowledge Vault (CKV).")

    def run(self, retrieval_memory_manager: RetrievalMemoryManager):
        return retrieval_memory_manager.retrieve_memories(self.query)


class conceptual_knowledge_vault_insert(BaseModel):
    """
    Add information to the Conceptual Knowledge Vault (CKV).
    """

    memory: str = Field(..., description="The information to be added to the Conceptual Knowledge Vault (CKV).")
    importance: float = Field(...,
                              description="The importance of the information to be added to the Conceptual Knowledge Vault (CKV). Value from 1 to 10")

    def run(self, retrieval_memory_manager: RetrievalMemoryManager):
        return retrieval_memory_manager.add_memory_to_retrieval(self.memory, self.importance)


class AgentRetrievalMemory:
    def __init__(self, persistent_db_path="./retrieval_memory", embedding_model_name="all-MiniLM-L6-v2",
                 collection_name="retrieval_memory_collection"):
        self.retrieval_memory = RetrievalMemory(persistent_db_path, embedding_model_name, collection_name)
        self.retrieval_memory_manager = RetrievalMemoryManager(self.retrieval_memory)
        self.retrieve_memories_tool = LlamaCppFunctionTool(conceptual_knowledge_vault_search,
                                                           retrieval_memory_manager=self.retrieval_memory_manager)
        self.add_retrieval_memory_tool = LlamaCppFunctionTool(conceptual_knowledge_vault_insert,
                                                              retrieval_memory_manager=self.retrieval_memory_manager)

    def get_tool_list(self):
        return [self.add_retrieval_memory_tool, self.retrieve_memories_tool]

    def get_retrieve_memories_tool(self):
        return self.retrieve_memories_tool

    def get_add_retrieval_memory_tool(self):
        return self.add_retrieval_memory_tool


class AgentCoreMemory:
    def __init__(self, core_memory=None, core_memory_file=None):
        if core_memory is None:
            core_memory = {}

        self.core_memory_manager = CoreMemoryManager(core_memory)
        if self.core_memory_manager is not None:
            self.core_memory_manager.load(core_memory_file)

        self.add_core_memory_tool = LlamaCppFunctionTool(immediate_access_memory_add,
                                                         core_memory_manager=self.core_memory_manager)
        self.remove_core_memory_tool = LlamaCppFunctionTool(immediate_access_memory_remove,
                                                            core_memory_manager=self.core_memory_manager)
        self.replace_core_memory_tool = LlamaCppFunctionTool(immediate_access_memory_replace,
                                                             core_memory_manager=self.core_memory_manager)

    def get_core_memory_manager(self):
        return self.core_memory_manager

    def get_tool_list(self):
        return [self.add_core_memory_tool, self.remove_core_memory_tool, self.replace_core_memory_tool]

    def get_add_core_memory_tool(self):
        return self.add_core_memory_tool

    def get_remove_core_memory_tool(self):
        return self.remove_core_memory_tool

    def get_replace_core_memory_tool(self):
        return self.replace_core_memory_tool

    def save_core_memory(self, file_path):
        self.core_memory_manager.save(file_path)

    def load_core_memory(self, file_path):
        self.core_memory_manager.load(file_path)


class AgentEventMemory:
    def __init__(self, db_path='sqlite:///events.db'):
        self.engine = create_engine(db_path)
        session_factory = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(session_factory)
        self.session = self.Session()
        self.event_memory_manager = EventMemoryManager(self.session)
        self.search_event_memory_manager_tool = LlamaCppFunctionTool(interactive_memory_bank_Search,
                                                                     event_memory_manager=self.event_memory_manager)

    def get_event_memory_manager(self):
        return self.event_memory_manager

    def get_tool_list(self):
        return [self.search_event_memory_manager_tool]

    def get_search_event_memory_manager_tool(self):
        return self.search_event_memory_manager_tool
