# agents-from-scratch
Building real AI agents in Python from scratch. The model chooses its tools and decides when to stop. Runs on a local Ollama model, tests first.

## Phase 1 plan: one agent, one tool

```mermaid
flowchart TD
    Start([Start: question in]) --> Build[Build messages]
    Build --> Call[call_model]
    Build -.->|counts each pass| Step[Step counter]
    Call --> Model{{Model: qwen3:8b decides}}
    Model --> Validate[Validate reply]
    Validate -->|tool call| Tool[Run the tool]
    Tool -->|result goes back| Build
    Validate -->|answer| Answered[Stop: answered]
    Step -->|limit reached| Limit[Stop: step_limit]
    Answered --> End([End: return answer and stop reason])
    Limit --> End
```
