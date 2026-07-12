# Architecture

## Overview

TaskFlow follows a simple layered architecture: an HTTP layer, a service
layer, and a storage layer. This keeps the business logic (in
`task_service.py`) independent of both the transport (FastAPI routes, not
shown in this sample) and the storage backend.

```
Client
  |
  v
API layer (routes, request/response models)
  |
  v
Service layer (task_service.py, auth.py)
  |
  v
Storage layer (models.py + in-memory / Postgres store)
```

## Authentication

Every request must include a bearer token. Tokens are validated in
`auth.py`. In production, tokens are short-lived JWTs issued by the internal
SSO provider; in local development, a static dev token is accepted so
engineers can hit the API without standing up SSO.

## Task lifecycle

A task is created in the `todo` status. It can move to `in_progress` once
someone is assigned, to `blocked` if work stalls, and to `done` when
complete. Direct transitions from `todo` to `done` are allowed for trivial
tasks, but transitions are always validated by
`task_service.transition_status` so invalid jumps (like `done` back to
`todo`) raise a clear error instead of silently corrupting state.

## Known limitations

- The in-memory store used in local development does not persist across
  restarts. Do not rely on it for anything you need to keep.
- There is currently no soft-delete; deleting a task removes it permanently.
- Rate limiting is not implemented yet — see the tracking issue
  `PLAT-1421` for the planned token-bucket approach.
