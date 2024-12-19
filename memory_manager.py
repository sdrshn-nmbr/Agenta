from mem0 import MemoryClient
from typing import Dict, Any, Optional, List, Literal, Union, TypedDict, Tuple
import os
from datetime import datetime, timedelta
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import random

# Enhanced memory categories for RL
MemoryCategory = Literal[
    "action",  # Direct actions taken
    "reward",  # Rewards and feedback
    "state",  # Environment state
    "strategy",  # High-level strategies
    "experience",  # Completed action sequences
    "policy",  # Learned behavior patterns
    "error",  # Failures and exceptions
]


class MemoryMetadata(TypedDict):
    """Structured metadata for memory entries"""

    session_id: str
    category: MemoryCategory
    timestamp: str
    confidence: float  # Confidence level in the action/decision
    success_score: float  # Success metric (0-1)
    execution_time: float  # Time taken in seconds
    resource_usage: Dict[str, float]  # Resource metrics
    dependencies: List[str]  # Related memory IDs
    tags: List[str]  # Categorical tags
    version: str  # Version of the action/strategy
    agent_type: Optional[str]  # Type of agent if applicable
    parent_id: Optional[str]  # Parent memory ID if part of sequence


class Experience(TypedDict):
    """Represents a single experience entry for RL"""
    state: Dict[str, Any]  # State when action was taken
    action: str  # Action taken
    reward: float  # Reward received
    next_state: Dict[str, Any]  # Resulting state
    metadata: MemoryMetadata  # Associated metadata
    timestamp: str  # When this experience occurred


class ExperienceBuffer:
    """Manages experience replay for RL learning"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer: List[Experience] = []
        self.max_size = max_size
    
    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer"""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)  # Remove oldest experience
        self.buffer.append(experience)
    
    def sample(self, batch_size: int = 32) -> List[Experience]:
        """Sample a random batch of experiences"""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)
    
    def get_recent(self, n: int = 10) -> List[Experience]:
        """Get the n most recent experiences"""
        return self.buffer[-n:]
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer = []
    
    def __len__(self) -> int:
        return len(self.buffer)


class Policy(TypedDict):
    """Represents a learned policy for decision making"""
    name: str  # Unique identifier for the policy
    state_pattern: Dict[str, Any]  # State pattern this policy applies to
    action_weights: Dict[str, float]  # Action preferences with weights
    success_rate: float  # Historical success rate (0-1)
    confidence: float  # Confidence in this policy (0-1)
    use_count: int  # Number of times policy has been used
    last_updated: str  # Timestamp of last update
    metadata: MemoryMetadata  # Associated metadata


class PolicyManager:
    """Manages learning and updating policies from experiences"""
    
    def __init__(self, min_confidence: float = 0.1):
        self.policies: List[Policy] = []
        self.min_confidence = min_confidence
    
    def add_policy(self, policy: Policy) -> None:
        """Add a new policy"""
        self.policies.append(policy)
    
    def update_policy(
        self,
        policy_name: str,
        success: bool,
        confidence_delta: float = 0.1
    ) -> None:
        """Update policy success rate and confidence"""
        for policy in self.policies:
            if policy["name"] == policy_name:
                # Update success rate with exponential moving average
                alpha = 0.1  # Learning rate
                policy["success_rate"] = (
                    (1 - alpha) * policy["success_rate"] + 
                    alpha * (1.0 if success else 0.0)
                )
                
                # Update confidence
                if success:
                    policy["confidence"] = min(
                        1.0, 
                        policy["confidence"] + confidence_delta
                    )
                else:
                    policy["confidence"] = max(
                        self.min_confidence,
                        policy["confidence"] - confidence_delta
                    )
                
                policy["use_count"] += 1
                policy["last_updated"] = str(datetime.now())
                break
    
    def get_policy(
        self,
        state: Dict[str, Any],
        min_confidence: Optional[float] = None
    ) -> Optional[Policy]:
        """Get best matching policy for a state"""
        if not self.policies:
            return None
            
        min_conf = min_confidence if min_confidence is not None else self.min_confidence
        
        # Filter policies by minimum confidence
        valid_policies = [
            p for p in self.policies 
            if p["confidence"] >= min_conf
        ]
        
        if not valid_policies:
            return None
        
        # Score policies by state match and success rate
        def score_policy(policy: Policy) -> float:
            pattern = policy["state_pattern"]
            # Calculate state match score (simple exact match for now)
            match_score = sum(
                1 for k, v in pattern.items()
                if k in state and state[k] == v
            ) / max(len(pattern), 1)
            
            # Combine match score with success rate and confidence
            return (
                match_score * 0.4 +
                policy["success_rate"] * 0.4 +
                policy["confidence"] * 0.2
            )
        
        # Return policy with highest score
        return max(valid_policies, key=score_policy)
    
    def prune_policies(
        self,
        min_success_rate: float = 0.3,
        min_use_count: int = 5
    ) -> None:
        """Remove underperforming policies"""
        self.policies = [
            p for p in self.policies
            if (p["success_rate"] >= min_success_rate or 
                p["use_count"] < min_use_count)
        ]

    def remove_policy(self, policy_name: str) -> bool:
        """Remove a specific policy by name"""
        initial_count = len(self.policies)
        self.policies = [p for p in self.policies if p["name"] != policy_name]
        return len(self.policies) < initial_count


class PerformanceMetrics:
    """Tracks and analyzes agent performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "success_rate": [],  # Overall success rate history
            "reward_history": [],  # Reward history
            "execution_times": [],  # Action execution times
            "resource_usage": [],  # Resource utilization
            "strategy_effectiveness": {},  # Strategy-specific metrics
            "agent_performance": {},  # Agent-specific metrics
        }
        
    def add_metric(self, category: str, value: float, metadata: Optional[Dict] = None):
        """Add a new metric measurement"""
        if category not in self.metrics:
            self.metrics[category] = []
        
        entry = {
            "value": value,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.metrics[category].append(entry)
        
        # Keep only last 1000 entries for each category
        if len(self.metrics[category]) > 1000:
            self.metrics[category] = self.metrics[category][-1000:]
    
    def get_trend(self, category: str, window: int = 100) -> float:
        """Calculate trend for a metric category"""
        if category not in self.metrics or not self.metrics[category]:
            return 0.0
            
        recent = self.metrics[category][-window:]
        if not recent:
            return 0.0
            
        values = [entry["value"] for entry in recent]
        return sum(values) / len(values)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of current performance metrics"""
        return {
            "success_rate": self.get_trend("success_rate"),
            "avg_reward": self.get_trend("reward_history"),
            "avg_execution_time": self.get_trend("execution_times"),
            "resource_efficiency": self.get_trend("resource_usage")
        }


class StrategyPruning:
    """Manages strategy optimization and pruning"""
    
    def __init__(self, min_success_rate: float = 0.3):
        self.min_success_rate = min_success_rate
        self.strategy_stats = {}
        self.decay_factor = 0.95  # Exponential decay factor
        
    def update_strategy(
        self,
        strategy_id: str,
        success: bool,
        execution_time: float,
        resource_usage: Dict[str, float]
    ):
        """Update statistics for a strategy"""
        if strategy_id not in self.strategy_stats:
            self.strategy_stats[strategy_id] = {
                "successes": 0,
                "attempts": 0,
                "avg_execution_time": 0.0,
                "resource_usage": {},
                "last_used": datetime.now(),
                "effectiveness_score": 0.5  # Initial neutral score
            }
        
        stats = self.strategy_stats[strategy_id]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
        
        # Update moving averages
        alpha = 0.1  # Learning rate
        stats["avg_execution_time"] = (
            (1 - alpha) * stats["avg_execution_time"] + 
            alpha * execution_time
        )
        
        # Update resource usage tracking
        for resource, usage in resource_usage.items():
            if resource not in stats["resource_usage"]:
                stats["resource_usage"][resource] = usage
            else:
                stats["resource_usage"][resource] = (
                    (1 - alpha) * stats["resource_usage"][resource] + 
                    alpha * usage
                )
        
        # Update effectiveness score
        success_rate = stats["successes"] / stats["attempts"]
        time_score = 1.0 / (1.0 + stats["avg_execution_time"])  # Normalize time
        resource_score = 1.0 / (1.0 + sum(stats["resource_usage"].values()))
        
        stats["effectiveness_score"] = (
            0.5 * success_rate +
            0.3 * time_score +
            0.2 * resource_score
        )
        
        stats["last_used"] = datetime.now()
    
    def apply_decay(self, max_age_days: int = 30):
        """Apply time-based decay to strategy scores"""
        now = datetime.now()
        for strategy_id, stats in self.strategy_stats.items():
            age_days = (now - stats["last_used"]).days
            if age_days > 0:
                decay = self.decay_factor ** min(age_days, max_age_days)
                stats["effectiveness_score"] *= decay
    
    def get_strategies_to_prune(self) -> List[str]:
        """Identify strategies that should be pruned"""
        self.apply_decay()  # Apply time decay before pruning
        
        to_prune = []
        for strategy_id, stats in self.strategy_stats.items():
            if (stats["attempts"] >= 5 and  # Minimum attempts threshold
                stats["effectiveness_score"] < self.min_success_rate):
                to_prune.append(strategy_id)
        
        return to_prune
    
    def get_best_strategies(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top performing strategies"""
        self.apply_decay()  # Apply time decay before ranking
        
        return sorted(
            [
                (strategy_id, stats["effectiveness_score"])
                for strategy_id, stats in self.strategy_stats.items()
                if stats["attempts"] >= 5  # Minimum attempts threshold
            ],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]


class MemoryManager(BaseModel):
    """Memory manager for handling persistent memory operations with RL capabilities"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Public fields
    user_id: str = Field(default="default_user")
    session_id: str = Field(default_factory=lambda: os.urandom(16).hex())
    version: str = Field(default="1.0.0")

    # Private fields that won't be validated by Pydantic
    _client: MemoryClient = PrivateAttr()
    _start_time: datetime = PrivateAttr(default_factory=datetime.now)
    _experience_buffer: ExperienceBuffer = PrivateAttr()
    _policy_manager: PolicyManager = PrivateAttr()
    _performance_metrics: PerformanceMetrics = PrivateAttr()
    _strategy_pruning: StrategyPruning = PrivateAttr()

    def __init__(self, user_id: str = "default_user", **kwargs):
        super().__init__(user_id=user_id, **kwargs)
        api_key = os.getenv("MEM0_API_KEY")
        self._client = MemoryClient(api_key=api_key)
        self._start_time = datetime.now()
        self._experience_buffer = ExperienceBuffer()
        self._policy_manager = PolicyManager()
        self._performance_metrics = PerformanceMetrics()
        self._strategy_pruning = StrategyPruning()

    def _create_base_metadata(self, category: MemoryCategory) -> MemoryMetadata:
        """Create base metadata structure"""
        return MemoryMetadata(
            session_id=self.session_id,
            category=category,
            timestamp=str(datetime.now()),
            confidence=0.0,
            success_score=0.0,
            execution_time=0.0,
            resource_usage={},
            dependencies=[],
            tags=[],
            version=self.version,
            agent_type=None,
            parent_id=None,
        )

    def add_memory(
        self,
        data: str,
        category: MemoryCategory,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        dependencies: List[str] = None,
        tags: List[str] = None,
        parent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
    ) -> Dict:
        """Add a new memory entry with enhanced metadata."""
        try:
            base_metadata = self._create_base_metadata(category)

            # Calculate execution time if start_time provided
            if start_time:
                execution_time = (datetime.now() - start_time).total_seconds()
                base_metadata["execution_time"] = execution_time

            # Update with provided metadata
            if metadata:
                base_metadata.update(metadata)

            # Add additional fields
            base_metadata["confidence"] = confidence
            base_metadata["dependencies"] = dependencies or []
            base_metadata["tags"] = tags or []
            base_metadata["parent_id"] = parent_id
            base_metadata["agent_type"] = agent_type

            messages = [
                {"role": "assistant", "content": data, "metadata": base_metadata}
            ]

            return self._client.add(messages, user_id=self.user_id)
        except Exception as e:
            print(f"Error adding memory: {str(e)}")
            return {}

    def add_reward_memory(
        self,
        reward: float,
        action_id: str,
        success_score: float = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Add a reward memory with success metrics."""
        reward_metadata = {
            "reward_value": reward,
            "action_id": action_id,
            "success_score": (
                success_score
                if success_score is not None
                else max(0.0, min(1.0, (reward + 1) / 2))
            ),
            **(metadata or {}),
        }
        return self.add_memory(
            data=f"Reward {reward} received for action {action_id}",
            category="reward",
            metadata=reward_metadata,
            confidence=1.0,  # Rewards are certain
            dependencies=[action_id],
            tags=["reward", f"score_{reward_metadata['success_score']:.1f}"],
        )

    def add_experience_memory(
        self,
        action_sequence: List[str],
        outcome: str,
        success_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Add a completed experience (action sequence) to memory."""
        experience_metadata = {
            "action_sequence": action_sequence,
            "success_score": success_score,
            **(metadata or {}),
        }
        return self.add_memory(
            data=f"Experience: {outcome}",
            category="experience",
            metadata=experience_metadata,
            confidence=success_score,
            dependencies=action_sequence,
            tags=["experience", f"score_{success_score:.1f}"],
        )

    def add_policy_memory(
        self,
        policy_name: str,
        policy_data: Dict[str, Any],
        success_rate: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Add a learned policy pattern to memory."""
        policy_metadata = {
            "policy_name": policy_name,
            "policy_data": policy_data,
            "success_rate": success_rate,
            **(metadata or {}),
        }
        return self.add_memory(
            data=f"Policy: {policy_name}",
            category="policy",
            metadata=policy_metadata,
            confidence=success_rate,
            tags=["policy", f"success_rate_{success_rate:.1f}"],
        )

    def search_memories(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        min_confidence: float = 0.0,
        min_success_score: float = 0.0,
        tags: List[str] = None,
        agent_type: Optional[str] = None,
    ) -> List[Dict]:
        """Enhanced semantic search with filters."""
        try:
            filters = {
                "AND": [
                    {"user_id": self.user_id},
                    {"metadata": {"session_id": self.session_id}},
                ]
            }

            # Add category filter
            if category:
                filters["AND"].append({"metadata": {"category": category}})

            # Add confidence filter
            if min_confidence > 0:
                filters["AND"].append(
                    {"metadata": {"confidence": {"$gte": min_confidence}}}
                )

            # Add success score filter
            if min_success_score > 0:
                filters["AND"].append(
                    {"metadata": {"success_score": {"$gte": min_success_score}}}
                )

            # Add tags filter
            if tags:
                filters["AND"].append({"metadata": {"tags": {"$in": tags}}})

            # Add agent type filter
            if agent_type:
                filters["AND"].append({"metadata": {"agent_type": agent_type}})

            response = self._client.search(query=query, version="v2", filters=filters)

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

    def get_session_memories(
        self,
        category: Optional[MemoryCategory] = None,
        min_confidence: float = 0.0,
        min_success_score: float = 0.0,
        tags: List[str] = None,
    ) -> List[Dict]:
        """Get filtered session memories."""
        try:
            filters = {"session_id": self.session_id}

            if category:
                filters["category"] = category
            if min_confidence > 0:
                filters["confidence"] = {"$gte": min_confidence}
            if min_success_score > 0:
                filters["success_score"] = {"$gte": min_success_score}
            if tags:
                filters["tags"] = {"$in": tags}

            response = self._client.get_all(user_id=self.user_id, metadata=filters)

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

    def update_memory(
        self,
        memory_id: str,
        data: str,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Update memory with new data and optional metadata updates."""
        try:
            base_metadata = {"session_id": self.session_id}
            if metadata_updates:
                base_metadata.update(metadata_updates)

            return self._client.update_memory(
                memory_id=memory_id, memory=data, metadata=base_metadata
            )
        except Exception as e:
            print(f"Error updating memory: {str(e)}")
            return {}

    def add_experience(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        metadata: Optional[MemoryMetadata] = None
    ) -> None:
        """Add an experience to the replay buffer"""
        if metadata is None:
            metadata = self._create_base_metadata("experience")
        
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            metadata=metadata,
            timestamp=str(datetime.now())
        )
        
        self._experience_buffer.add(experience)
        
        # Also store in persistent memory
        self.add_memory(
            data=f"Experience: Action '{action}' with reward {reward}",
            category="experience",
            metadata={
                **metadata,
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state
            }
        )

    def get_experiences(self, batch_size: int = 32) -> List[Experience]:
        """Sample experiences from the replay buffer"""
        return self._experience_buffer.sample(batch_size)

    def get_recent_experiences(self, n: int = 10) -> List[Experience]:
        """Get the n most recent experiences"""
        return self._experience_buffer.get_recent(n)

    def clear_experiences(self) -> None:
        """Clear the experience buffer"""
        self._experience_buffer.clear()

    def add_policy(
        self,
        name: str,
        state_pattern: Dict[str, Any],
        action_weights: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new policy"""
        if metadata is None:
            metadata = self._create_base_metadata("policy")
        
        policy = Policy(
            name=name,
            state_pattern=state_pattern,
            action_weights=action_weights,
            success_rate=0.5,  # Initial neutral success rate
            confidence=0.1,  # Initial low confidence
            use_count=0,
            last_updated=str(datetime.now()),
            metadata=metadata
        )
        
        self._policy_manager.add_policy(policy)
        
        # Store in persistent memory
        self.add_memory(
            data=f"New policy created: {name}",
            category="policy",
            metadata={
                **metadata,
                "policy_name": name,
                "state_pattern": state_pattern,
                "action_weights": action_weights
            }
        )

    def get_policy(
        self,
        state: Dict[str, Any],
        min_confidence: Optional[float] = None
    ) -> Optional[Policy]:
        """Get best matching policy for current state"""
        return self._policy_manager.get_policy(state, min_confidence)

    def update_policy(
        self,
        policy_name: str,
        success: bool,
        confidence_delta: float = 0.1
    ) -> None:
        """Update policy with execution result"""
        self._policy_manager.update_policy(
            policy_name,
            success,
            confidence_delta
        )
        
        # Store update in memory
        self.add_memory(
            data=f"Policy update: {policy_name} - {'Success' if success else 'Failure'}",
            category="policy",
            metadata={
                "type": "policy_update",
                "policy_name": policy_name,
                "success": success,
                "confidence_delta": confidence_delta
            }
        )

    def optimize_policies(
        self,
        min_success_rate: float = 0.3,
        min_use_count: int = 5
    ) -> None:
        """Optimize policy set by removing underperforming ones"""
        before_count = len(self._policy_manager.policies)
        self._policy_manager.prune_policies(min_success_rate, min_use_count)
        after_count = len(self._policy_manager.policies)
        
        if before_count != after_count:
            self.add_memory(
                data=f"Policy optimization: Removed {before_count - after_count} policies",
                category="policy",
                metadata={
                    "type": "policy_optimization",
                    "removed_count": before_count - after_count,
                    "min_success_rate": min_success_rate,
                    "min_use_count": min_use_count
                }
            )

    def learn_from_experiences(self, batch_size: int = 32) -> None:
        """Learn from past experiences to create or update policies"""
        experiences = self.get_experiences(batch_size)
        
        for exp in experiences:
            # Check if there's an existing policy for this state pattern
            policy = self.get_policy(exp["state"])
            
            if policy:
                # Update existing policy
                success = exp["reward"] > 0
                self.update_policy(
                    policy["name"],
                    success,
                    confidence_delta=abs(exp["reward"]) * 0.1
                )
            else:
                # Create new policy from this experience
                state_pattern = exp["state"]
                action_weights = {exp["action"]: max(0.1, exp["reward"])}
                
                self.add_policy(
                    name=f"learned_policy_{len(self._policy_manager.policies)}",
                    state_pattern=state_pattern,
                    action_weights=action_weights,
                    metadata={
                        "source_experience": exp["timestamp"],
                        "initial_reward": exp["reward"]
                    }
                )
        
        # Optimize policies after learning
        self.optimize_policies()

    def track_performance(
        self,
        category: str,
        value: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Track a performance metric"""
        self._performance_metrics.add_metric(category, value, metadata)
        
        # Store in persistent memory
        self.add_memory(
            data=f"Performance metric: {category} = {value}",
            category="state",
            metadata={
                "type": "performance_metric",
                "category": category,
                "value": value,
                **(metadata or {})
            }
        )

    def get_performance_summary(self) -> Dict[str, float]:
        """Get current performance metrics summary"""
        return self._performance_metrics.get_performance_summary()

    def update_strategy_performance(
        self,
        strategy_id: str,
        success: bool,
        execution_time: float,
        resource_usage: Dict[str, float]
    ) -> None:
        """Update performance tracking for a strategy"""
        self._strategy_pruning.update_strategy(
            strategy_id,
            success,
            execution_time,
            resource_usage
        )
        
        # Store strategy update in memory
        self.add_memory(
            data=f"Strategy update: {strategy_id}",
            category="strategy",
            metadata={
                "type": "strategy_update",
                "strategy_id": strategy_id,
                "success": success,
                "execution_time": execution_time,
                "resource_usage": resource_usage
            }
        )

    def optimize_strategies(self) -> None:
        """Run strategy optimization and pruning"""
        # Get strategies to prune
        to_prune = self._strategy_pruning.get_strategies_to_prune()
        
        # Get best performing strategies
        best_strategies = self._strategy_pruning.get_best_strategies()
        
        # Store optimization results
        self.add_memory(
            data="Strategy optimization performed",
            category="strategy",
            metadata={
                "type": "strategy_optimization",
                "pruned_strategies": to_prune,
                "best_strategies": best_strategies
            }
        )
        
        # Actually prune the strategies from policy manager
        for strategy_id in to_prune:
            self._policy_manager.remove_policy(strategy_id)

    def get_strategy_performance(self, strategy_id: str) -> Optional[Dict]:
        """Get performance metrics for a specific strategy"""
        if strategy_id in self._strategy_pruning.strategy_stats:
            return self._strategy_pruning.strategy_stats[strategy_id]
        return None
