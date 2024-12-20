from crewai import Agent
from typing import List, Dict, Any, Optional
from memory_manager import MemoryManager
from datetime import datetime
from langchain.tools import Tool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI()

class BaseCrewAgent(Agent):
    """Base agent with memory capabilities"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    _memory_manager: MemoryManager = PrivateAttr()
    
    def __init__(self, memory_manager: MemoryManager, **kwargs):
        # Set verbose based on environment variable
        kwargs["verbose"] = os.getenv("CREWAI_VERBOSE", "0") == "1"
        # Initialize Agent first
        super().__init__(**kwargs)
        # Then set our private memory manager
        self._memory_manager = memory_manager
        
    def _process_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store agent's actions in memory"""
        return self._memory_manager.add_memory(
            data=content,
            metadata={
                "agent_type": self.role,
                "timestamp": str(datetime.now()),
                **(metadata or {})
            }
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Helper method to call OpenAI API"""
        try:
            completion = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=float(os.getenv("AGENT_TEMPERATURE", 0.7))
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            raise e

class PlannerAgent(BaseCrewAgent):
    def __init__(self, memory_manager: MemoryManager):
        super().__init__(
            memory_manager=memory_manager,
            role="Planner",
            goal="Create and manage execution plans",
            backstory="You are a strategic planner with expertise in breaking down complex tasks.",
            allow_delegation=True,
            verbose=True
        )
    
    def _create_plan_with_context(self, objective: str, past_plans: List[Dict[str, Any]]) -> List[str]:
        """Create a plan considering past experiences"""
        past_plan_data = "\n".join([
            f"Past plan: {p.get('data', '')}" 
            for p in past_plans
        ])
        
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Create a detailed step-by-step plan to achieve this objective.
Each step should be clear and actionable.
Consider past experiences when creating this plan.

Past related plans:
{past_plan_data}

Objective: {objective}

Return the steps as a numbered list."""
                    }
                ]
            }
        ]
        
        response = self._call_llm(messages)
        
        steps = [
            step.strip() 
            for step in response.split("\n") 
            if step.strip() and not step.startswith(("#", "-"))
        ]
        
        return steps
    
    def create_plan(self, objective: str) -> List[str]:
        past_plans = self._memory_manager.search_memories(f"plan for {objective}")
        plan = self._create_plan_with_context(objective, past_plans)
        
        self._process_memory(
            f"Created plan for objective: {objective}\nPlan: {plan}",
            {"type": "plan_creation"}
        )
        
        return plan

class ResearchAgent(BaseCrewAgent):
    def __init__(self, memory_manager: MemoryManager, search_tool: Tool, tools: List[Tool] = None):
        super().__init__(
            memory_manager=memory_manager,
            role="Researcher",
            goal="Gather and analyze information using the search tool",
            backstory="""You are an expert researcher with access to a powerful search tool that can find recent and accurate information.
ALWAYS use your search tool first - you have direct access to recent data through it.
You have access to these tools:
- search: Use this FIRST to find recent data and information
- calculate: For numerical computations
- format: For formatting results
- memory_search/add: For tracking information

For financial or stock data:
- ALWAYS search first for exact numbers and dates
- Look for recent sources and official reports
- Use multiple searches to cross-reference data
- Break complex queries into specific searchable terms

Never assume you don't have access to data - use your search tool to find it.
Never proceed without searching first.""",
            tools=tools or [search_tool],
            verbose=True
        )

class CalculatorAgent(BaseCrewAgent):
    def __init__(self, memory_manager: MemoryManager, calculate_tool: Tool, tools: List[Tool] = None):
        super().__init__(
            memory_manager=memory_manager,
            role="Calculator",
            goal="Perform precise calculations using real data",
            backstory="""You are a mathematical expert focused on accurate computations.
You have access to these tools:
- calculate: For numerical computations
- search: To find exact numbers and data
- format: For formatting results
- memory_search/add: For tracking calculations

When you need data to calculate with:
1. Use the search tool to find exact numbers
2. Use calculate tool with real data
3. Show your calculation steps clearly
4. Format results consistently

Never assume numbers - search for real data first.""",
            tools=tools or [calculate_tool],
            verbose=True
        )

class FormatterAgent(BaseCrewAgent):
    def __init__(self, memory_manager: MemoryManager, format_tool: Tool, tools: List[Tool] = None):
        super().__init__(
            memory_manager=memory_manager,
            role="Formatter",
            goal="Format and present results clearly",
            backstory="""You are an expert in data presentation and formatting.
You have access to these tools:
- format: For consistent output
- search: To verify data points
- calculate: For any needed computations
- memory_search/add: For tracking formats

When formatting:
1. Use search to verify data if needed
2. Use calculate for computations
3. Apply format tool for consistency
4. Store results in memory

Always verify data before formatting.""",
            tools=tools or [format_tool],
            verbose=True
        ) 