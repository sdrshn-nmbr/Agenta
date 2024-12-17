from mem0 import MemoryClient
from typing import Dict, Any, Optional, List
import os
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class MemoryManager(BaseModel):
    """Memory manager for handling persistent memory operations"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Public fields
    user_id: str = Field(default="default_user")
    session_id: str = Field(default_factory=lambda: os.urandom(16).hex())

    # Private fields that won't be validated by Pydantic
    _client: MemoryClient = PrivateAttr()

    def __init__(self, user_id: str = "default_user", **kwargs):
        super().__init__(user_id=user_id, **kwargs)
        # Initialize private attributes after Pydantic validation
        api_key = os.getenv("MEM0_API_KEY")
        self._client = MemoryClient(api_key=api_key)

    def add_memory(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> Dict:
        """Add a new memory entry."""
        try:
            messages = [{"role": "assistant", "content": data}]
            if metadata:
                messages[0]["metadata"] = {"session_id": self.session_id, **metadata}
            return self._client.add(messages, user_id=self.user_id)
        except Exception as e:
            print(f"Error adding memory: {str(e)}")
            return {}

    def search_memories(self, query: str) -> List[Dict]:
        """Search for relevant memories using semantic search."""
        try:
            # Use v2 search with proper filters
            response = self._client.search(
                query=query,
                version="v2",
                filters={
                    "AND": [
                        {"user_id": self.user_id},
                        {"metadata": {"session_id": self.session_id}},
                    ]
                },
            )
            return [
                {
                    "data": memory.get("memory", ""),
                    "metadata": memory.get("metadata", {}),
                }
                for memory in response
            ]
        except Exception as e:
            print(f"Error searching memories: {str(e)}")
            return []

    def get_session_memories(self) -> List[Dict]:
        """Get all memories from current session."""
        try:
            # Use get_all with proper filters
            response = self._client.get_all(
                user_id=self.user_id, metadata={"session_id": self.session_id}
            )
            return [
                {
                    "data": memory.get("memory", ""),
                    "metadata": memory.get("metadata", {}),
                }
                for memory in response
            ]
        except Exception as e:
            print(f"Error getting session memories: {str(e)}")
            return []

    def update_memory(self, memory_id: str, data: str) -> Dict:
        """Update an existing memory."""
        try:
            return self._client.update_memory(
                memory_id=memory_id,
                memory=data,
                metadata={"session_id": self.session_id},
            )
        except Exception as e:
            print(f"Error updating memory: {str(e)}")
            return {}
