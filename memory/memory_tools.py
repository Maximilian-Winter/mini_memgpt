from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from .event_memory import EventType, Base
from .event_memory_manager import EventMemoryManager
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from .core_memory_manager import CoreMemoryManager
from .retrieval_memory_manager import RetrievalMemoryManager, RetrievalMemory


class AddCoreMemory(BaseModel):
    """
    Add a new entry to the core memory.
    """
    inner_thoughts: str = Field(..., description="Your inner thoughts.")
    key: str = Field(..., description="The key identifier for the core memory entry.")
    field: str = Field(..., description="A secondary key or field within the core memory entry.")
    value: str = Field(..., description="The value or data to be stored in the specified core memory entry.")
    require_heartbeat: bool = Field(...,
                                    description="Set this to true to get control back after execution, to chain functions together.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.add_to_core_memory(self.key, self.field, self.value)


# Replace Core Memory Model
class ReplaceCoreMemory(BaseModel):
    """
    Replace an entry in the core memory.
    """
    inner_thoughts: str = Field(..., description="Your inner thoughts.")
    key: str = Field(..., description="The key identifier for the core memory entry.")
    field: str = Field(..., description="The specific field within the core memory entry to be replaced.")
    new_value: str = Field(...,
                           description="The new value to replace the existing data in the specified core memory field.")
    require_heartbeat: bool = Field(...,
                                    description="Set this to true to get control back after execution, to chain functions together.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.replace_in_core_memory(self.key, self.field, self.value)


class ConversationSearch(BaseModel):
    """
    Search for memories from the recall memory.
    """
    inner_thoughts: str = Field(..., description="Your inner thoughts while writing the search query.")
    event_types: Optional[list[EventType]] = Field(...,
                                                   description="Event types to search. Can be system, 'user', 'assistant' or 'function'")
    start_date: Optional[str] = Field(..., description='Start date to search events from. Format: "%Y-%m-%d %H:%M"')
    end_date: Optional[str] = Field(..., description='End date to search events from. Format: "%Y-%m-%d %H:%M"')
    keywords: Optional[List[str]] = Field(..., description='End date to search events from. Format: "%Y-%m-%d %H:%M"')
    require_heartbeat: bool = Field(...,
                                    description="Set this to true to get control back after execution, to chain functions together.")

    def run(self, event_memory_manager: EventMemoryManager):
        parsed_start_datetime = None
        parsed_end_datetime = None
        if self.start_date:
            parsed_start_datetime = datetime.strptime(self.start_date, "%Y-%m-%d %H:%M")
        if self.end_date:
            parsed_end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d %H:%M")

        return event_memory_manager.query_events(event_types=self.event_types, keywords=self.keywords,
                                                 start_date=parsed_start_datetime, end_date=parsed_end_datetime)


class ArchivalMemorySearch(BaseModel):
    """
    Retrieve memories from the archival memory based on a query.
    """
    inner_thoughts: str = Field(..., description="Your inner thoughts while writing the archival query.")
    query: str = Field(..., description="The query to be used to retrieve memories from the archival memory.")
    require_heartbeat: bool = Field(...,
                                    description="Set this to true to get control back after execution, to chain functions together.")

    def run(self, retrieval_memory_manager: RetrievalMemoryManager):
        return retrieval_memory_manager.retrieve_memories(self.query)


class ArchivalMemoryInsert(BaseModel):
    """
    Add memory to the archival memory.
    """
    inner_thoughts: str = Field(..., description="Your inner thoughts while adding archival memory.")
    memory: str = Field(..., description="The memory to be added to the archival memory.")
    importance: float = Field(..., description="The importance of the memory to be added to the archival memory.")
    require_heartbeat: bool = Field(...,
                                    description="Set this to true to get control back after execution, to chain functions together.")

    def run(self, retrieval_memory_manager: RetrievalMemoryManager):
        return retrieval_memory_manager.add_memory_to_retrieval(self.memory, self.importance)


class AgentRetrievalMemory:
    def __init__(self, persistent_db_path="./retrieval_memory", embedding_model_name="all-MiniLM-L6-v2",
                 collection_name="retrieval_memory_collection"):
        self.retrieval_memory = RetrievalMemory(persistent_db_path, embedding_model_name, collection_name)
        self.retrieval_memory_manager = RetrievalMemoryManager(self.retrieval_memory)
        self.retrieve_memories_tool = LlamaCppFunctionTool(ArchivalMemorySearch,
                                                           retrieval_memory_manager=self.retrieval_memory_manager)
        self.add_retrieval_memory_tool = LlamaCppFunctionTool(ArchivalMemoryInsert,
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

        self.add_core_memory_tool = LlamaCppFunctionTool(AddCoreMemory,
                                                         core_memory_manager=self.core_memory_manager)
        self.replace_core_memory_tool = LlamaCppFunctionTool(ReplaceCoreMemory,
                                                             core_memory_manager=self.core_memory_manager)

    def get_core_memory_manager(self):
        return self.core_memory_manager

    def get_tool_list(self):
        return [self.add_core_memory_tool, self.replace_core_memory_tool]

    def get_add_core_memory_tool(self):
        return self.add_core_memory_tool

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
        self.search_event_memory_manager_tool = LlamaCppFunctionTool(ConversationSearch,
                                                                     event_memory_manager=self.event_memory_manager)

    def get_event_memory_manager(self):
        return self.event_memory_manager

    def get_tool_list(self):
        return [self.search_event_memory_manager_tool]

    def get_search_event_memory_manager_tool(self):
        return self.search_event_memory_manager_tool
