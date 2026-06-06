# Architecture Diagrams

Place the following diagram images in this directory:

- `1_memory_pipeline.png` — Sources → Chunk+tag → Embed → Memory store → Retrieve+rank → Assemble → LLM
- `2_system_architecture.png` — Interface / Gateway / Agent core / Connectors / Data stores / External targets layers
- `3_sequence_diagram.png` — Request sequence: You → Orchestrator → Memory → Claude → Connector
- `4_data_schema_erd.png` — ERD: users, conversations, messages, tasks, task_steps, connectors, credentials, devices, memory_chunks
- `5_task_state_machine.png` — Task states: pending → planning → running → completed/failed/cancelled/waiting/blocked
