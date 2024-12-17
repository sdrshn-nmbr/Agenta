from typing import List, Dict, Any, Union, TypedDict, Annotated, Optional
import operator
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from exa_py import Exa
from rich.console import Console
import os
from formatting import format_plan, format_step_output, format_reflection
from memory_manager import MemoryManager
from datetime import datetime
from crew_agents import PlannerAgent, ResearchAgent, CalculatorAgent, FormatterAgent, BaseCrewAgent
from crewai import Crew, Process, Task

load_dotenv()

# Initialize Exa client and console
exa = Exa(os.getenv("EXA_API_KEY"))
console = Console()


class Plan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    steps: List[str] = Field(description="Steps to follow in sequence")
    reasoning: str = Field(description="Reasoning behind the plan")


class Action(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    tool_name: str
    tool_input: str
    thought: str = Field(description="Reasoning for using this tool")


class AgentState(TypedDict):
    plan: List[str]
    current_step: int
    observations: Annotated[List[Dict], operator.add]
    messages: List[Any]
    reflection: str
    context: Dict[str, Any]
    memory_context: Dict[str, Any]
    memory_id: Optional[str]


def extract_number(text: str) -> float:
    """Extract the first number from text, handling various formats."""
    # Remove commas from numbers
    text = text.replace(",", "")
    # Find numbers in scientific notation or decimal format
    matches = re.findall(r"[-+]?\d*\.\d+e[-+]?\d+|\d*\.\d+|\d+", text)
    if matches:
        return float(matches[0])
    return 0.0


def format_currency(amount: float) -> str:
    """Format number as USD currency."""
    if amount >= 1e9:
        return f"${amount/1e9:.2f}B"
    if amount >= 1e6:
        return f"${amount/1e6:.2f}M"
    return f"${amount:,.2f}"


class ReflectiveAgent:
    def __init__(self):
        # Initialize base components first
        self.llm = ChatOpenAI(
            temperature=float(os.getenv("AGENT_TEMPERATURE", 0.7)),
            model="gpt-4o",
        )
        self.memory_manager = MemoryManager()
        self.tools = self._get_tools()
        
        # Initialize CrewAI agents with tools
        self.planner = PlannerAgent(memory_manager=self.memory_manager)
        self.researcher = ResearchAgent(
            memory_manager=self.memory_manager, 
            search_tool=self.tools[0],  # search tool
            tools=self.tools  # give access to all tools
        )
        self.calculator = CalculatorAgent(
            memory_manager=self.memory_manager, 
            calculate_tool=self.tools[1],  # calculate tool
            tools=self.tools  # give access to all tools
        )
        self.formatter = FormatterAgent(
            memory_manager=self.memory_manager, 
            format_tool=self.tools[3],  # format tool
            tools=self.tools  # give access to all tools
        )
        
        # Create CrewAI crew
        self.crew = Crew(
            agents=[self.planner, self.researcher, self.calculator, self.formatter],
            tasks=[],  # Tasks will be added dynamically
            process=Process.sequential,
            verbose=True
        )
        
        # Initialize graph last since it depends on other components
        self.graph = self._build_graph()

    def _get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search",
                func=self._smart_search,
                description="Search for information with neural search and content retrieval",
            ),
            Tool(
                name="calculate",
                func=self._smart_calculate,
                description="Perform calculations with unit handling",
            ),
            Tool(
                name="reflect",
                func=self._reflect_on_actions,
                description="Reflect on past actions and update strategy",
            ),
            Tool(
                name="format",
                func=self._format_result,
                description="Format numbers and results properly",
            ),
            Tool(
                name="memory_search",
                func=self._search_memory,
                description="Search through past memories and experiences",
            ),
            Tool(
                name="memory_add",
                func=self._add_memory,
                description="Add new memory entry from current context",
            ),
        ]

    def _smart_search(self, query: str) -> str:
        """Enhanced search using Exa AI with neural search and content retrieval."""
        try:
            # Log search query and parameters
            console.print(f"[dim]🔍 Search query: {query}[/dim]")
            
            # Prepare search parameters according to docs
            search_params = {
                "query": query,
                "type": "neural",
                "use_autoprompt": True,
                "num_results": 10,
                "start_published_date": "2018-01-01"
            }
            console.print(f"[dim]🔧 Search parameters: {search_params}[/dim]")
            
            # Execute search
            search_response = exa.search(**search_params)
            
            # Extract results from the SearchResponse object
            if not search_response or not hasattr(search_response, 'results'):
                console.print("[red]❌ No results in search response[/red]")
                memory_result = self.memory_manager.add_memory(
                    data="No search results found",
                    category="action",
                    metadata={
                        "type": "search_failure",
                        "query": query,
                        "timestamp": str(datetime.now())
                    }
                )
                if memory_result and "id" in memory_result:
                    self.memory_manager.add_reward_memory(
                        reward=-0.5,  # Penalize failed search
                        action_id=memory_result["id"],
                        metadata={
                            "type": "search_reward",
                            "query": query,
                            "success": False
                        }
                    )
                return "No search results found"

            results = search_response.results
            console.print(f"[dim]📊 Found {len(results)} results[/dim]")

            # Get document IDs for content retrieval
            doc_ids = [result.id for result in results if hasattr(result, 'id')]
            
            if not doc_ids:
                console.print("[red]❌ No valid document IDs found[/red]")
                memory_result = self.memory_manager.add_memory(
                    data="No valid document IDs found",
                    category="action",
                    metadata={
                        "type": "search_failure",
                        "query": query,
                        "timestamp": str(datetime.now())
                    }
                )
                if memory_result and "id" in memory_result:
                    self.memory_manager.add_reward_memory(
                        reward=-0.3,  # Smaller penalty for partial failure
                        action_id=memory_result["id"],
                        metadata={
                            "type": "search_reward",
                            "query": query,
                            "success": False
                        }
                    )
                return "No valid document IDs found"

            # Get contents for the documents
            contents_params = {
                "ids": doc_ids,
                "text": {
                    "max_characters": 3000,
                    "include_html_tags": False
                },
                "highlights": {
                    "num_sentences": 5,
                    "highlights_per_url": 3,
                    "query": query
                }
            }
            
            contents_response = exa.get_contents(**contents_params)
            
            if not contents_response or not hasattr(contents_response, 'results'):
                console.print("[red]❌ No content results[/red]")
                memory_result = self.memory_manager.add_memory(
                    data="No content could be retrieved",
                    category="action",
                    metadata={
                        "type": "search_failure",
                        "query": query,
                        "timestamp": str(datetime.now())
                    }
                )
                if memory_result and "id" in memory_result:
                    self.memory_manager.add_reward_memory(
                        reward=-0.3,  # Smaller penalty for partial failure
                        action_id=memory_result["id"],
                        metadata={
                            "type": "search_reward",
                            "query": query,
                            "success": False
                        }
                    )
                return "No content could be retrieved"

            # Process results
            summary = []
            total_score = 0.0  # Initialize as float
            for result in contents_response.results:
                title = getattr(result, 'title', 'No title')
                url = getattr(result, 'url', '')
                text = getattr(result, 'text', '')
                highlights = getattr(result, 'highlights', [])
                published_date = getattr(result, 'published_date', 'No date')
                score = getattr(result, 'score', 0.0)  # Default to 0.0 if None
                if score is not None:  # Only add if score is not None
                    total_score += float(score)  # Convert to float to be safe

                console.print(f"[dim]📄 Processing result:[/dim]")
                console.print(f"[dim]   Title: {title[:100]}...[/dim]")
                console.print(f"[dim]   Score: {score}[/dim]")
                console.print(f"[dim]   Date: {published_date}[/dim]")

                summary.append(f"Source: {title}")
                summary.append(f"URL: {url}")
                summary.append(f"Published: {published_date}")

                if text:
                    summary.append("Content:")
                    summary.append(text[:500] + "..." if len(text) > 500 else text)

                if highlights:
                    summary.append("Key highlights:")
                    for highlight in highlights:
                        summary.append(f"- {highlight}")
                
                summary.append("")

            if not summary:
                console.print("[red]❌ No content could be extracted[/red]")
                memory_result = self.memory_manager.add_memory(
                    data="No relevant information could be extracted",
                    category="action",
                    metadata={
                        "type": "search_failure",
                        "query": query,
                        "timestamp": str(datetime.now())
                    }
                )
                if memory_result and "id" in memory_result:
                    self.memory_manager.add_reward_memory(
                        reward=-0.4,  # Penalty for extraction failure
                        action_id=memory_result["id"],
                        metadata={
                            "type": "search_reward",
                            "query": query,
                            "success": False
                        }
                    )
                return "No relevant information could be extracted"

            # Store search results in memory with reward based on quality
            search_summary = "\n".join(summary)
            memory_result = self.memory_manager.add_memory(
                data=search_summary,
                category="action",
                metadata={
                    "type": "search_result",
                    "query": query,
                    "num_results": len(results),
                    "avg_score": total_score / len(results) if results else 0,
                    "timestamp": str(datetime.now())
                }
            )
            
            # Add reward based on search quality
            if memory_result and "id" in memory_result:
                # Calculate reward based on number of results and average score
                result_count_factor = min(len(results) / 5, 1.0)  # Max out at 5 results
                avg_score = total_score / len(results) if results else 0
                score_factor = min(avg_score / 0.7, 1.0)  # Normalize score, max at 0.7
                reward = 0.5 + (0.25 * result_count_factor) + (0.25 * score_factor)
                
                self.memory_manager.add_reward_memory(
                    reward=reward,
                    action_id=memory_result["id"],
                    metadata={
                        "type": "search_reward",
                        "query": query,
                        "success": True,
                        "num_results": len(results),
                        "avg_score": avg_score
                    }
                )

            return search_summary

        except Exception as e:
            console.print(f"[red]❌ Search error: {str(e)}[/red]")
            import traceback
            console.print(f"[red]Stack trace: {traceback.format_exc()}[/red]")
            
            # Store error in memory
            memory_result = self.memory_manager.add_memory(
                data=f"Search error: {str(e)}",
                category="action",
                metadata={
                    "type": "search_error",
                    "query": query,
                    "error": str(e),
                    "timestamp": str(datetime.now())
                }
            )
            
            # Add negative reward for error
            if memory_result and "id" in memory_result:
                self.memory_manager.add_reward_memory(
                    reward=-0.5,  # Penalize errors
                    action_id=memory_result["id"],
                    metadata={
                        "type": "search_reward",
                        "query": query,
                        "success": False,
                        "error": str(e)
                    }
                )
            
            return f"Search error: {str(e)}"

    def _smart_calculate(self, expression: str) -> str:
        """Enhanced calculator with unit handling."""
        try:
            numbers = re.findall(r"[-+]?\d*\.?\d+", expression)
            result = None
            
            if len(numbers) == 2:
                num1, num2 = map(float, numbers)
                if "divide" in expression.lower() or "/" in expression:
                    result = num1 / num2
                elif "multiply" in expression.lower() or "*" in expression:
                    result = num1 * num2
                else:
                    result = eval(expression)
            else:
                result = eval(expression)
                
            # Store calculation in memory
            memory_result = self.memory_manager.add_memory(
                data=f"Calculation: {expression} = {result}",
                category="action",
                metadata={
                    "type": "calculation",
                    "expression": expression,
                    "result": result,
                    "timestamp": str(datetime.now())
                }
            )
            
            # Add reward for successful calculation
            if memory_result and "id" in memory_result:
                self.memory_manager.add_reward_memory(
                    reward=1.0,  # Reward successful calculation
                    action_id=memory_result["id"],
                    metadata={
                        "type": "calculation_reward",
                        "expression": expression,
                        "success": True
                    }
                )
            
            return str(result)
        except Exception as e:
            error_msg = f"Error in calculation: {str(e)}"
            # Store error in memory
            memory_result = self.memory_manager.add_memory(
                data=error_msg,
                category="action",
                metadata={
                    "type": "calculation_error",
                    "expression": expression,
                    "error": str(e),
                    "timestamp": str(datetime.now())
                }
            )
            
            # Add negative reward for failed calculation
            if memory_result and "id" in memory_result:
                self.memory_manager.add_reward_memory(
                    reward=-0.5,  # Penalize failed calculation
                    action_id=memory_result["id"],
                    metadata={
                        "type": "calculation_reward",
                        "expression": expression,
                        "success": False
                    }
                )
            
            return error_msg

    def _reflect_on_actions(self, observations: List[Dict]) -> str:
        """Reflect on past actions and update strategy."""
        try:
            # Get relevant memories for context
            memories = self.memory_manager.get_session_memories()
            memory_context = "\n".join([m.get("data", "") for m in memories])
            
            # Create reflection task
            reflection_task = Task(
                description=f"""
                Review the following observations and provide insights:
                {observations}
                
                Context from memory:
                {memory_context}
                
                Analyze the execution, success rate, and potential improvements.
                """,
                agent=self.planner
            )
            
            # Execute reflection
            result = self.crew.execute_task(reflection_task)
            
            # Store reflection in memory
            self.memory_manager.add_memory(
                data=str(result),
                category="strategy",
                metadata={
                    "type": "reflection",
                    "observations": observations,
                    "timestamp": str(datetime.now())
                }
            )
            
            return result
        except Exception as e:
            error_msg = f"Reflection error: {str(e)}"
            self.memory_manager.add_memory(
                data=error_msg,
                category="strategy",
                metadata={
                    "type": "reflection_error",
                    "error": str(e),
                    "timestamp": str(datetime.now())
                }
            )
            return error_msg

    def _format_result(self, text: str) -> str:
        """Format numbers and add proper units."""
        try:
            number = extract_number(text)
            result = format_currency(number)
            
            # Store formatting result
            self.memory_manager.add_memory(
                data=f"Formatted {text} to {result}",
                category="action",
                metadata={
                    "type": "formatting",
                    "input": text,
                    "output": result,
                    "timestamp": str(datetime.now())
                }
            )
            
            return result
        except Exception as e:
            error_msg = f"Formatting error: {str(e)}"
            self.memory_manager.add_memory(
                data=error_msg,
                category="action",
                metadata={
                    "type": "formatting_error",
                    "input": text,
                    "error": str(e),
                    "timestamp": str(datetime.now())
                }
            )
            return text

    def _search_memory(self, query: str) -> str:
        """Search through memory for relevant information."""
        try:
            memories = self.memory_manager.search_memories(query)
            if not memories:
                return "No relevant memories found"
            
            memory_text = []
            for mem in memories[:3]:
                memory_text.append(f"Memory: {mem.get('data', '')}")
                if metadata := mem.get('metadata', {}):
                    memory_text.append(f"Context: {metadata}")
                memory_text.append("")
            
            return "\n".join(memory_text)
        except Exception as e:
            return f"Memory search error: {str(e)}"

    def _add_memory(self, data: str) -> str:
        """Add new memory entry from current context."""
        try:
            result = self.memory_manager.add_memory(
                data,
                category="state",
                metadata={
                    "type": "agent_memory",
                    "timestamp": str(datetime.now())
                }
            )
            return f"Memory added successfully: {result.get('id', 'unknown')}"
        except Exception as e:
            return f"Failed to add memory: {str(e)}"

    def _create_planner(self):
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Create a detailed plan to accomplish the objective.
Consider:
1. What specific information needs to be searched
2. What calculations are needed
3. How to format the final result
4. Potential data inconsistencies to watch for

For simple calculations:
- Use the calculate tool directly
- No need for multiple steps
- Return result immediately

For complex tasks:
- Break down into logical steps
- Include verification steps
- Consider data dependencies""",
                ),
                ("human", "{objective}"),
            ]
        )

        return planner_prompt | self.llm.with_structured_output(Plan)

    def _create_executor(self):
        executor_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Execute the current step of the plan.
Available tools: {tool_names}
Current step: {current_step}
Plan context: {plan}
Previous observations: {observations}
Reflection insights: {reflection}
Memory context: {memory_context}

For searches:
- Use specific, targeted queries
- Include relevant context and timeframes
- Look for numerical data and facts
- Consider source credibility
- Leverage past experiences from memory

For calculations:
- Use precise numbers from search results
- Include units in calculations
- Show your reasoning
- Learn from past similar calculations

For formatting:
- Use appropriate units
- Round to reasonable precision
- Use standard formats (e.g., USD for currency)
- Maintain consistency with past formats""",
                ),
                ("human", "{input}"),
            ]
        )

        return executor_prompt | self.llm.with_structured_output(Action)

    def _create_task_for_agent(self, step: str, agent: BaseCrewAgent) -> Task:
        """Create a task with proper context for an agent"""
        # Get relevant memories for context
        memories = self.memory_manager.search_memories(step)
        context = "\n".join([m.get("data", "") for m in memories])
        
        return Task(
            description=f"""
            Step to execute: {step}
            
            Context from memory:
            {context}
            
            Execute this step considering the context and your expertise.
            """,
            agent=agent,
            expected_output="Detailed results of the step execution"
        )

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Planning node with Mem0 integration
        async def planning_step(state: AgentState):
            # Get relevant memories for context
            memories = self.memory_manager.get_session_memories()
            memory_context = "\n".join([m.get("data", "") for m in memories])
            
            # Create plan using standard LLM
            plan_response = self.llm.invoke(
                f"""Create a detailed plan to accomplish the objective.
Consider past experiences:
{memory_context}

Break down into clear, actionable steps.

Objective: {state["messages"][-1].content}"""
            ).content
            
            # Parse plan from response
            steps = [
                step.strip() 
                for step in plan_response.split("\n") 
                if step.strip() and not step.startswith(("#", "-"))
            ]
            
            # Store the plan in memory
            self.memory_manager.add_memory(
                f"Created plan: {steps}",
                category="strategy",
                metadata={"type": "plan", "step": "planning"}
            )
            
            return {
                "planning_step": {"plan": steps},
                "current_step": 0,
                "plan": steps,
            }

        # Execution node with memory integration
        async def execution_step(state: AgentState):
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)

            if not plan or current_step >= len(plan):
                return {"messages": [AIMessage(content="Plan completed")]}

            try:
                step = plan[current_step]
                
                # Get relevant memories for context
                memories = self.memory_manager.search_memories(step)
                context = "\n".join([m.get("data", "") for m in memories])
                
                # Use standard LLM for agent selection
                agent_selection = self.llm.invoke(
                    f"""Based on the step and context, select the best agent:
Step: {step}
Context: {context}

Available agents:
- Researcher (search and analysis)
- Calculator (numerical computations)
- Formatter (data presentation)

Return only the agent name."""
                ).content.strip().lower()
                
                primary_agent = {
                    "researcher": self.researcher,
                    "calculator": self.calculator,
                    "formatter": self.formatter
                }.get(agent_selection, self.researcher)

                # Create and execute task
                task = self._create_task_for_agent(step, primary_agent)
                
                # Update crew's tasks and execute
                self.crew = Crew(
                    agents=[self.planner, self.researcher, self.calculator, self.formatter],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )
                result = self.crew.kickoff()

                # Store result in memory
                self.memory_manager.add_memory(
                    str(result),
                    category="action",
                    metadata={
                        "step": current_step,
                        "agent": primary_agent.role,
                        "task": step,
                        "context_used": context
                    }
                )

                return {
                    "observations": [{"action": step, "result": result}],
                    "current_step": current_step + 1,
                    "context": {
                        **state.get("context", {}),
                        f"step_{current_step}": result
                    },
                    "result": str(result),
                    "memory_context": {
                        "step": current_step,
                        "agent": primary_agent.role
                    }
                }

            except Exception as e:
                error_msg = f"Error in execution: {str(e)}"
                print(f"Execution error details: {str(e)}")
                
                self.memory_manager.add_memory(
                    error_msg,
                    category="action",
                    metadata={
                        "type": "error",
                        "step": current_step
                    }
                )
                
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "current_step": current_step + 1,
                    "context": state.get("context", {}),
                    "result": error_msg,
                    "memory_context": state.get("memory_context", {})
                }

        # Reflection node with memory integration
        async def reflection_step(state: AgentState):
            if "observations" in state and state["observations"]:
                memories = self.memory_manager.get_session_memories()
                memory_context = "\n".join([m.get("data", "") for m in memories])
                
                reflection = self.llm.invoke(
                    f"""Analyze the execution and provide insights.
Observations: {state['observations']}
Memory Context: {memory_context}

Consider:
1. Effectiveness of actions
2. Learning points
3. Potential improvements"""
                ).content
                
                self.memory_manager.add_memory(
                    reflection,
                    category="strategy",
                    metadata={"type": "reflection"}
                )
                
                return {"reflection": reflection}
                
            return {}

        # Add nodes
        workflow.add_node("planning_step", planning_step)
        workflow.add_node("execution_step", execution_step)
        workflow.add_node("reflection_step", reflection_step)

        # Add edges
        workflow.add_edge(START, "planning_step")
        workflow.add_edge("planning_step", "execution_step")
        workflow.add_edge("execution_step", "reflection_step")

        # Conditional edges
        def should_continue(state: AgentState):
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)
            observations = state.get("observations", [])

            # If we have observations and have completed all steps, end
            if observations and current_step >= len(plan):
                return END

            # If we have a final result in the last observation, end
            if observations and observations[-1].get("result"):
                return END

            # Continue execution
            return "execution_step"

        workflow.add_conditional_edges("reflection_step", should_continue)

        return workflow.compile()

    async def run(self, objective: str):
        initial_state = {
            "messages": [HumanMessage(content=objective)],
            "plan": [],
            "current_step": 0,
            "observations": [],
            "reflection": "",
            "context": {},
            "memory_context": {},
            "memory_id": None,
        }

        final_result = None

        try:
            async for state in self.graph.astream(initial_state):
                # Print plan when available
                if "planning_step" in state and isinstance(state["planning_step"], dict):
                    plan = state["planning_step"].get("plan", [])
                    if plan:
                        console.print("\n")
                        console.print(format_plan(plan))
                        console.print("\n")

                # Handle execution results
                if "observations" in state and state["observations"]:
                    latest_observation = state["observations"][-1]
                    if "result" in latest_observation:
                        final_result = latest_observation["result"]
                        if isinstance(final_result, dict) and "Final Answer" in final_result:
                            final_result = final_result["Final Answer"]
                        console.print("\n")
                        console.print(format_step_output(
                            state.get("current_step", 0),
                            latest_observation.get("action", "Processing"),
                            final_result
                        ))
                        console.print("\n")

                # Handle reflection
                if "reflection" in state:
                    reflection = state.get("reflection", "")
                    if reflection:
                        if isinstance(reflection, (AIMessage, HumanMessage, SystemMessage)):
                            reflection = reflection.content
                        console.print("\n")
                        console.print(format_reflection(reflection))
                        console.print("\n")

            # Return the final result if we have one
            if final_result:
                return final_result
            
            # If no final result but we have observations, return the last result
            if "observations" in state and state["observations"]:
                last_observation = state["observations"][-1]
                if "result" in last_observation:
                    result = last_observation["result"]
                    if isinstance(result, dict) and "Final Answer" in result:
                        return result["Final Answer"]
                    return str(result)

            return "No result was produced"

        except Exception as e:
            print(f"Runtime error: {str(e)}")  # Debug print
            return f"Error during execution: {str(e)}"
