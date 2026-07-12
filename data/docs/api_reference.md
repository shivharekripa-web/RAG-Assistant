# API Reference

This document describes the internal service functions that back the (not
included) HTTP routes. It's useful context for anyone integrating with
TaskFlow at the code level rather than over HTTP.

## Creating a task

Use `task_service.create_task`. It requires a title and board id; all other
fields are optional and take sensible defaults (`status=todo`,
`priority=medium`). It raises `ValueError` if the title is empty, since an
untitled task is almost always a client bug rather than an intentional
action.

## Assigning a task

Use `task_service.assign_task`. Reassigning a task that already has an
assignee is allowed and expected — it happens constantly when work gets
handed off between engineers — but it appends to the task's history rather
than silently overwriting the previous assignee, so audits stay possible.

## Transitioning status

Use `task_service.transition_status`. This is the only supported way to
change a task's status; do not mutate `task.status` directly anywhere else
in the codebase, since that bypasses the validation described in
`architecture.md`.

## Error codes

- `ERR_TASK_NOT_FOUND` — the task id doesn't exist in the store.
- `ERR_INVALID_TRANSITION` — the requested status change isn't a legal move.
- `ERR_AUTH_INVALID_TOKEN` — the bearer token failed validation in `auth.py`.
- `ERR_AUTH_EXPIRED_TOKEN` — the token was valid but has expired.
