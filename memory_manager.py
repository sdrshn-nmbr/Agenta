from mem0 import MemoryClient
from typing import Dict, Any, Optional, List, Literal
import os
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from enum import Enum

class MemoryCategory(str, Enum):
    ACTION = "action"
    REWARD = "reward"
    STATE = "state"
    STRATEGY = "strategy"

class MemoryMetadata(BaseModel):
    """Enhanced metadata structure for memory entries"""
    category: MemoryCategory
    success_score: Optional[float] = Field(None, ge=0, le=1)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    completion_time: Optional[float] = None  # in seconds
    resource_usage: Optional[Dict[str, float]] = None
    dependencies: Optional[List[str]] = None  # list of memory IDs
    session_id: str

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
        """Add a new memory entry with enhanced metadata."""
        try:
            base_metadata = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
            }
            
            if metadata:
                base_metadata.update(metadata)
                
            # Validate metadata if category is provided
            if "category" in base_metadata:
                MemoryMetadata(**base_metadata)
                
            messages = [{"role": "assistant", "content": data, "metadata": base_metadata}]
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
