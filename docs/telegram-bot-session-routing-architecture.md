# Telegram Bot / Session Routing Architecture

## Mermaid Diagram

```mermaid
flowchart TD
    U1[Telegram User / Group Topic]
    U2[Telegram User / Group Topic]
    B1[Bot A]
    B2[Bot B]

    U1 --> B1
    U2 --> B2

    B1 --> G[Bot Gateway / Ingress]
    B2 --> G

    G --> N[Normalize Update]

    N --> R[Routing Layer]

    R --> K1["Route Key:\nbot_id + chat_id + topic_id + user_id"]
    K1 --> M[Binding Store]

    M --> S1[Session Worker A]
    M --> S2[Session Worker B]
    M --> S3[Session Worker C]

    S1 --> C1[Claude Code Session A]
    S2 --> C2[Claude Code Session B]
    S3 --> C3[Claude Code Session C]

    C1 --> D[Response Dispatcher]
    C2 --> D
    C3 --> D

    D --> B1
    D --> B2

    B1 --> U1
    B2 --> U2
```

## Notes

- Supports **single bot, multiple sessions** via `chat_id/topic_id -> session_key`
- Supports **multiple bots, multiple sessions** via `bot_id + chat_id + topic_id -> session_key`
- `Binding Store` is the critical mapping layer
- `Session Worker` can be backed by Claude Code CLI, tmux, or another runtime
