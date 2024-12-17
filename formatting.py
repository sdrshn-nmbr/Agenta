from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich.padding import Padding


def format_plan(plan: list) -> Panel:
    """Format plan steps in a nice panel."""
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2))
    for i, step in enumerate(plan, 1):
        # Convert potential markdown in steps to rich text
        step_md = Markdown(step)
        table.add_row(
            Text(f"Step {i}", style="bold cyan"), 
            Text("→", style="cyan"),
            step_md
        )
    return Panel(
        Padding(table, (1, 2)),
        title="[bold cyan]🎯 Execution Plan[/bold cyan]",
        border_style="cyan",
        width=100,
        subtitle="[dim cyan]Breaking down the objective into steps[/dim cyan]"
    )


def format_step_output(step: int, action: str, result: str) -> Panel:
    """Format step output in a nice panel."""
    # Truncate long results
    max_result_length = 1000  # Increased for better markdown visibility
    if len(result) > max_result_length:
        result = result[:max_result_length] + "..."
    
    content = Table(show_header=False, show_edge=False, box=None, padding=(0, 2))
    content.add_row(
        Text("🔍 Action", style="bold yellow"), 
        Text("→", style="yellow"),
        Markdown(action)  # Convert action to markdown
    )
    content.add_row(
        Text("📝 Result", style="bold yellow"), 
        Text("→", style="yellow"),
        Markdown(result)  # Convert result to markdown
    )
    return Panel(
        Padding(content, (1, 2)),
        title=f"[bold yellow]⚡ Step {step} Execution[/bold yellow]",
        border_style="yellow",
        width=100,
        subtitle="[dim yellow]Processing current step[/dim yellow]"
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
