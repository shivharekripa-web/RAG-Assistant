# TaskFlow

TaskFlow is an internal task management service used by the Platform team to
track work items across squads. It exposes a small REST API backed by an
in-memory store (swappable for Postgres in production) and handles task
creation, assignment, status transitions, and simple analytics.

## Why TaskFlow exists

Before TaskFlow, task tracking was split across spreadsheets and a shared doc.
Neither supported programmatic access, so no other internal tool could query
"what's overdue" or "what's assigned to me" without a human copying rows by
hand. TaskFlow gives every other internal service a single source of truth.

## Core concepts

- **Task** — the basic unit of work. Has a title, description, status,
  assignee, priority, and due date.
- **Board** — a named collection of tasks, roughly equivalent to a project.
- **Assignment** — the act of linking a task to a user. Reassignment keeps a
  history entry so we can audit who worked on what.
- **Status transition** — tasks move through `todo -> in_progress -> done`,
  or can be moved to `blocked` from any non-done state.

## Who maintains this

The Platform team owns TaskFlow. For bugs or feature requests, file an issue
in the `platform/taskflow` repository.
