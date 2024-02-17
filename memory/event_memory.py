from sqlalchemy import Column, Integer, Text, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base

from enum import Enum as PyEnum
import json


class EventType(PyEnum):
    SystemMessage = "system"
    AgentMessage = "assistant"
    UserMessage = "user"
    FunctionMessage = "function"


Base = declarative_base()


class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    event_type = Column(Enum(EventType))
    timestamp = Column(DateTime, index=True)
    content = Column(Text)
    event_keywords = Column(Text)  # Storing keywords as JSON string

    def __str__(self):
        content = f'Timestamp: {self.timestamp.strftime("%Y-%m-%d %H:%M")}\n{self.content}'
        return content

    def add_keyword(self, keyword):
        """Add a keyword to the event."""
        if self.event_keywords:
            keywords = json.loads(self.event_keywords)
        else:
            keywords = []
        keywords.append(keyword)
        self.event_keywords = json.dumps(keywords)


