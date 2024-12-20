from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich.padding import Padding
from typing import Any


def format_plan(plan: Any) -> Panel:
    """Format a plan in a nice panel."""
    if hasattr(plan, 'steps') and hasattr(plan, 'reasoning'):
        # Handle structured plan
        steps = plan.steps
        reasoning = plan.reasoning
        estimated_steps = getattr(plan, 'estimated_steps', len(steps))
        
        content = Group(
            Text("📋 Plan Steps:", style="bold green"),
            *(Text(f"  {i+1}. {step}", style="green") for i, step in enumerate(steps)),
            Text("\n🤔 Reasoning:", style="bold blue"),
            Text(reasoning, style="blue"),
            Text(f"\n⏱️ Estimated Steps: {estimated_steps}", style="bold yellow")
        )
    else:
        # Handle legacy plan format (list of steps)
        if isinstance(plan, (list, tuple)):
            steps = plan
        else:
            steps = [str(plan)]
            
        content = Group(
            Text("📋 Plan:", style="bold green"),
            *(Text(f"  {i+1}. {step}", style="green") for i, step in enumerate(steps))
        )
    
    return Panel(
        Padding(content, (1, 2)),
        title="[bold green]🎯 Execution Plan[/bold green]",
        border_style="green",
        width=100
    )


def format_step_output(step_number: int, action: str, result: Any) -> Panel:
    """Format a step output with action and result."""
    if hasattr(result, 'content') and hasattr(result, 'success'):
        # Handle structured output
        content = result.content
        success = result.success
        is_final = getattr(result, 'is_final', False)
        
        if is_final:
            return Panel(
                Group(
                    Text(f"Final Answer:", style="bold green" if success else "bold red"),
                    Text(content)
                ),
                title="[bold]Final Result[/bold]",
                border_style="green" if success else "red",
                width=100
            )
    else:
        content = str(result)
    
    return Panel(
        Group(
            Text(f"Step {step_number}:", style="bold blue"),
            Text(f"Action: {action}", style="blue"),
            Text("Result:", style="bold"),
            Text(content)
        ),
        title="[bold]Step Output[/bold]",
        border_style="blue",
        width=100
    )


def format_reflection(reflection: str) -> Panel:
    """Format reflection in a nice panel with markdown."""
    # Enhanced markdown handling
    if "###" in reflection:
        # Split on headers but preserve the ### marker
        sections = [s if s.startswith("###") else f"### {s}" for s in reflection.split("###") if s.strip()]
        formatted_sections = []
        for section in sections:
            # Add some spacing between sections
            formatted_sections.append(Padding(Markdown(section), (1, 0)))
        content = Group(*formatted_sections)
    else:
        # Add bullet points if none exist
        if not any(line.strip().startswith(("- ", "* ", "1. ")) for line in reflection.split("\n")):
            reflection = "- " + "\n- ".join(line for line in reflection.split("\n") if line.strip())
        content = Markdown(reflection)

    return Panel(
        Padding(content, (1, 2)),
        title="[bold blue]💭 Reflection & Analysis[/bold blue]",
        border_style="blue",
        width=100,
        subtitle="[dim blue]Analyzing execution and insights[/dim blue]"
    )


def format_final_result(result: str, verbose: bool = False) -> Panel:
    """Format final result in a nice panel."""
    if not result:
        style = "yellow"
        content = "No result was produced"
    elif isinstance(result, dict) and "Final Answer" in result:
        style = "green"
        content = result["Final Answer"]
    elif "Final Answer:" in str(result):
        style = "green"
        content = str(result).split("Final Answer:", 1)[1].strip()
    elif "error" in str(result).lower():
        style = "red"
        content = str(result)
    else:
        style = "green"
        content = str(result)

    if verbose:
        table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
        table.add_row(
            Text("Result:", style=style),
            Text(content, style=style, overflow="fold")
        )
        return Panel(
            Padding(table, (1, 2)),
            title=f"[bold {style}]Final Result[/bold {style}]",
            border_style=style,
            width=100,
        )
    else:
        return Panel(
            Markdown(content) if "**" in content else Text(content, style=style, overflow="fold"),
            title=f"[bold {style}]Final Result[/bold {style}]",
            border_style=style,
            width=100
        )
