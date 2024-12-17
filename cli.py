import typer
import asyncio
from rich import print
from rich.console import Console
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from agent import ReflectiveAgent
from formatting import format_final_result
from dotenv import load_dotenv
import os

app = typer.Typer()
console = Console()


def check_environment():
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            Panel.fit(
                "[red]Error: OPENAI_API_KEY not found in environment![/red]\n"
                "Please copy .env.template to .env and add your OpenAI API key."
            )
        )
        raise typer.Exit(1)
    if not os.getenv("EXA_API_KEY"):
        console.print(
            Panel.fit(
                "[red]Error: EXA_API_KEY not found in environment![/red]\n"
                "Please copy .env.template to .env and add your Exa API key."
            )
        )
        raise typer.Exit(1)
    if not os.getenv("MEM0_API_KEY"):
        console.print(
            Panel.fit(
                "[red]Error: MEM0_API_KEY not found in environment![/red]\n"
                "Please copy .env.template to .env and add your Mem0 API key."
            )
        )
        raise typer.Exit(1)


@app.command()
def run(
    objective: str = typer.Argument(
        ..., help="The objective you want the agent to accomplish"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """
    Run the reflective agent with a specific objective
    """
    check_environment()

    with console.status(
        "[bold green]Agent thinking...[/bold green]", spinner="dots"
    ) as status:
        # Set environment variable to reduce LLM output verbosity
        os.environ["LANGCHAIN_VERBOSE"] = "0" if not verbose else "1"
        os.environ["CREWAI_VERBOSE"] = "0" if not verbose else "1"
        
        agent = ReflectiveAgent()
        result = asyncio.run(agent.run(objective))

        # Clear the status line
        console.print()

        # Print final result
        console.print(format_final_result(result, verbose))


@app.command()
def info():
    """
    Show information about the agent's configuration
    """
    check_environment()

    # Create a table for features
    features = Table(show_header=False, show_edge=False, box=None)
    features.add_row("[cyan]•[/cyan]", "Neural search with Exa AI")
    features.add_row("[cyan]•[/cyan]", "Reflective planning with memory")
    features.add_row("[cyan]•[/cyan]", "Multi-agent execution")
    features.add_row("[cyan]•[/cyan]", "Dynamic agent selection")
    features.add_row("[cyan]•[/cyan]", "Persistent memory with Mem0")
    features.add_row("[cyan]•[/cyan]", "Agent handoffs and delegation")
    features.add_row("[cyan]•[/cyan]", "Context-aware decision making")

    # Create a table for configuration
    config = Table(show_header=False, show_edge=False, box=None)
    config.add_row(
        "[yellow]Temperature:[/yellow]", os.getenv("AGENT_TEMPERATURE", "0.7")
    )
    config.add_row("[yellow]Max Iterations:[/yellow]", os.getenv("MAX_ITERATIONS", "5"))
    config.add_row("[yellow]Model:[/yellow]", "GPT-4-turbo-preview")

    # Combine into panels
    layout = Layout()
    layout.split_column(
        Panel(
            config,
            title="[bold yellow]Configuration[/bold yellow]",
            border_style="yellow",
        ),
        Panel(features, title="[bold cyan]Features[/bold cyan]", border_style="cyan"),
    )

    console.print(layout)


if __name__ == "__main__":
    app()
