"""CollectiveOS agent clients — each wraps a peer agent service."""
from .base import AgentClient, AgentRegistry, AgentResult
from .task_agent import TaskAgentClient
from .visual_agent import VisualOSClient

__all__ = ["AgentClient", "AgentRegistry", "AgentResult", "TaskAgentClient", "VisualOSClient"]
