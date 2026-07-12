"""Data models for TaskFlow.

These are plain dataclasses rather than an ORM model, since the in-memory
store used in local development doesn't need query capabilities beyond
simple filtering. A production deployment backed by Postgres would swap
these for SQLAlchemy models with the same field names.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    """The set of statuses a task may be in.

    Transitions between these are validated in task_service.transition_status
    rather than here, since the valid-transition graph is business logic,
    not a property of the enum itself.
    """

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"


class Priority(str, Enum):
    """Task priority levels, low to high."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class AssignmentHistoryEntry:
    """A single record of a task being assigned to someone.

    Kept so audits can answer "who has worked on this task" rather than only
    "who is currently assigned to it".
    """

    assignee: str
    assigned_at: datetime


@dataclass
class Task:
    """A single unit of work tracked by TaskFlow.

    id is assigned by the store on creation and is never set by callers
    directly, to guarantee uniqueness without needing a database round trip
    before the object exists.
    """

    id: Optional[str]
    title: str
    board_id: str
    description: str = ""
    status: TaskStatus = TaskStatus.TODO
    priority: Priority = Priority.MEDIUM
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    history: list = field(default_factory=list)

    def is_overdue(self, as_of: Optional[datetime] = None) -> bool:
        """Return True if the task has a due date in the past and isn't done.

        A done task is never considered overdue even if it was completed
        after its due date, since overdue is meant to flag work that still
        needs attention, not to record historical lateness.
        """
        if self.due_date is None or self.status == TaskStatus.DONE:
            return False
        reference_time = as_of or datetime.utcnow()
        return self.due_date < reference_time


@dataclass
class Board:
    """A named collection of tasks, roughly equivalent to a project."""

    id: Optional[str]
    name: str
    description: str = ""
