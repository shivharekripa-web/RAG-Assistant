"""Business logic for creating, assigning, and transitioning tasks.

This module is intentionally storage-agnostic: it accepts a `store` object
that implements get/save/delete, so the same logic works whether the store
is the in-memory dict used locally or a Postgres-backed store in production.
"""

import uuid
from datetime import datetime
from typing import Optional

from models import AssignmentHistoryEntry, Priority, Task, TaskStatus

# Valid status transitions. Any status can move to BLOCKED (work can stall
# at any point), but only specific forward/backward moves are otherwise
# allowed.
_VALID_TRANSITIONS = {
    TaskStatus.TODO: {TaskStatus.IN_PROGRESS, TaskStatus.DONE, TaskStatus.BLOCKED},
    TaskStatus.IN_PROGRESS: {TaskStatus.DONE, TaskStatus.BLOCKED, TaskStatus.TODO},
    TaskStatus.BLOCKED: {TaskStatus.TODO, TaskStatus.IN_PROGRESS},
    TaskStatus.DONE: set(),  # done is terminal; reopen by creating a new task
}


class TaskServiceError(Exception):
    """Base error for task_service failures, carrying a stable error code.

    The error code (e.g. ERR_TASK_NOT_FOUND) is what callers should match
    on, not the message text, since the message may be reworded over time.
    """

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def create_task(
    store,
    title: str,
    board_id: str,
    description: str = "",
    priority: Priority = Priority.MEDIUM,
    due_date: Optional[datetime] = None,
) -> Task:
    """Create a new task on the given board and persist it to the store.

    Raises ValueError if title is empty, since an untitled task is almost
    always a client bug rather than something intentional.
    """
    if not title or not title.strip():
        raise ValueError("Task title must not be empty")

    task = Task(
        id=str(uuid.uuid4()),
        title=title.strip(),
        board_id=board_id,
        description=description,
        status=TaskStatus.TODO,
        priority=priority,
        due_date=due_date,
    )
    store.save(task)
    return task


def assign_task(store, task_id: str, assignee: str) -> Task:
    """Assign (or reassign) a task to a user, recording history.

    Reassignment is expected and allowed — work gets handed off between
    engineers constantly — but the previous assignee is preserved in
    task.history rather than overwritten, so audits stay possible.
    """
    task = store.get(task_id)
    if task is None:
        raise TaskServiceError("ERR_TASK_NOT_FOUND", f"No task with id {task_id}")

    task.history.append(
        AssignmentHistoryEntry(assignee=assignee, assigned_at=datetime.utcnow())
    )
    task.assignee = assignee
    store.save(task)
    return task


def transition_status(store, task_id: str, new_status: TaskStatus) -> Task:
    """Move a task to a new status if the transition is legal.

    This is the only supported way to change task.status. Mutating the
    field directly elsewhere bypasses this validation and can put a task
    into an inconsistent state.
    """
    task = store.get(task_id)
    if task is None:
        raise TaskServiceError("ERR_TASK_NOT_FOUND", f"No task with id {task_id}")

    allowed = _VALID_TRANSITIONS.get(task.status, set())
    if new_status not in allowed:
        raise TaskServiceError(
            "ERR_INVALID_TRANSITION",
            f"Cannot move task from {task.status} to {new_status}",
        )

    task.status = new_status
    store.save(task)
    return task


def list_overdue_tasks(store, board_id: Optional[str] = None) -> list:
    """Return all overdue tasks, optionally filtered to a single board.

    Delegates the actual overdue check to Task.is_overdue so the definition
    of "overdue" lives in one place.
    """
    tasks = store.list_all(board_id=board_id) if board_id else store.list_all()
    return [t for t in tasks if t.is_overdue()]
