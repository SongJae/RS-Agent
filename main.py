"""
RS-Agent Command Line Interface.

Usage:
    python main.py                     # Interactive chat mode
    python main.py --query "..."       # Single query mode
    python main.py --ui                # Launch Gradio web UI
"""

import os
import sys
import argparse
import logging

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from rs_agent import RSAgent
from rs_agent.utils import load_config, setup_logging

load_dotenv()
console = Console()


def print_banner():
    banner = """
╦═╗╔═╗   ╔═╗╔═╗╔═╗╔╗╔╔╦╗
╠╦╝╚═╗───╠═╣║ ╦║╣ ║║║ ║
╩╚═╚═╝   ╩ ╩╚═╝╚═╝╝╚╝ ╩
Remote Sensing Intelligent Agent
arXiv: 2406.07089
    """
    console.print(Panel(banner, style="bold cyan"))


def interactive_mode(agent: RSAgent):
    """Run RS-Agent in interactive chat mode."""
    console.print("[bold green]RS-Agent Interactive Mode[/bold green]")
    console.print("Type 'quit' or 'exit' to stop, 'reset' to clear history\n")
    console.print(f"Available tools: {len(agent.get_available_tools())}")
    console.print(f"Knowledge documents: {len(agent.knowledge_db)}")
    console.print(f"Expert solutions: {len(agent.solution_db)}\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_input.lower() == "reset":
            agent.reset_conversation()
            console.print("[green]Conversation history cleared.[/green]\n")
            continue

        if user_input.lower() == "tools":
            console.print("\n[bold]Available Tools:[/bold]")
            for name, desc in agent.get_tool_descriptions().items():
                console.print(f"  [cyan]{name}[/cyan]: {desc[:80]}...")
            console.print()
            continue

        console.print("\n[bold green]RS-Agent:[/bold green]")
        try:
            response = agent.chat(user_input)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        console.print()


def single_query_mode(agent: RSAgent, query: str):
    """Run a single query and print the response."""
    console.print(f"\n[bold blue]Query:[/bold blue] {query}\n")
    console.print("[bold green]RS-Agent:[/bold green]")
    try:
        response = agent.chat(query)
        console.print(Markdown(response))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def launch_ui(agent: RSAgent):
    """Launch the Gradio web UI."""
    try:
        from app import create_gradio_app
        app = create_gradio_app(agent)
        app.launch(share=False, server_name="0.0.0.0", server_port=7860)
    except ImportError as e:
        console.print(f"[red]Gradio not available: {e}[/red]")
        console.print("Install with: pip install gradio")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RS-Agent: Remote Sensing Intelligent Agent"
    )
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--ui", action="store_true", help="Launch Gradio web UI")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    print_banner()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[yellow]Config file not found: {args.config}. Using defaults.[/yellow]")
        config = {}

    # Initialize RS-Agent
    console.print("[dim]Initializing RS-Agent...[/dim]")
    try:
        agent = RSAgent(config=config)
        console.print(f"[green]RS-Agent ready: {agent}[/green]\n")
    except ValueError as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        console.print("Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    # Run in appropriate mode
    if args.ui:
        launch_ui(agent)
    elif args.query:
        single_query_mode(agent, args.query)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
