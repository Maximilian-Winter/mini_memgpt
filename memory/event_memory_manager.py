from sqlalchemy.orm import Session
from .event_memory import Event, EventType
import datetime
import json


class EventMemoryManager:
    def __init__(self, session: Session):
        self.session = session
        self.event_queue: list[Event] = []

    def build_event_memory_context(self):
        messages = []
        for event in self.event_queue:
            messages.append({"role": event.event_type.value, "content": event.content})
        return messages

    def add_event_to_queue(self, event_type, content, metadata):
        new_event = Event(
            event_type=event_type,
            timestamp=datetime.datetime.now(),
            content=content,
            metadata=json.dumps(metadata)
        )
        self.event_queue.append(new_event)

        # Optional: Limit the size of the event queue
        if len(self.event_queue) > 20:
            self.commit_oldest_event()

    def commit_oldest_event(self):
        if self.event_queue:
            oldest_event = self.event_queue.pop(0)
            try:
                self.session.add(oldest_event)
                self.session.commit()
                return "Oldest event committed successfully."
            except Exception as e:
                self.session.rollback()
                return f"Error committing oldest event: {e}"
        else:
            return "Skipped committing event to database."

    def modify_event_in_queue(self, modification, event_index=-1):
        if not self.event_queue:
            return "Event queue is empty."

        if event_index < -len(self.event_queue) or event_index >= len(self.event_queue):
            return "Invalid event index."

        event_to_modify = self.event_queue[event_index]
        for key, value in modification.items():
            if hasattr(event_to_modify, key):
                setattr(event_to_modify, key, value)

        return "Event modified successfully."

    def query_events(self, event_types: list = None, start_date: datetime.datetime = None,
                     end_date: datetime.datetime = None,
                     content_keywords: list = None, keywords: list = None) -> str:
        query = self.session.query(Event)

        # Filtering based on provided criteria
        if event_types:
            query = query.filter(Event.event_type.in_(event_types))
        if start_date and end_date:
            query = query.filter(Event.timestamp.between(start_date, end_date))

        for keyword in content_keywords:
            query = query.filter(Event.content.contains(keyword))

        if keywords:
            for value in keywords:
                query = query.filter(Event.event_keywords.contains(value))

        events = query.all()
        return "\n".join([str(event) for event in events])
