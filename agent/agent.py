import os
import asyncio

from pathlib import Path
from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.llm import model

from rich.console import Console
from rich.markup import escape

load_dotenv(Path(__file__).resolve().parent / ".env")

console = Console()

MCP_URL = os.getenv("MCP_URL")

SYSTEM_PROMPT = """
    You are a GPA and CGPA academic assistant for my University.

    Your job is to help students calculate semester GPA, project CGPA, determine
    the GPA required to reach a target CGPA, and provide information about
    semester courses.

    RULES:

    1. Never perform GPA/CGPA calculations yourself. Always use the appropriate tool. For marks-to-grade-point conversion, always call marks_to_grade_points for each course mark. Never calculate or infer the grade points yourself. For semester GPA, use calculate_semester_gpa after obtaining all grade points and credit hours from the appropriate tools.

    2. When the user provides a semester number and asks for a GPA calculation, use get_semester_courses to obtain the official course names and credit hours for that semester. Do not ask the user for credit hours that are already available in the course data.

    3. If the user doesn't provide required parameters. Donot invent them like marks, gpa etc. Always ask the user to give necessary parameters.

    4. MD-001 and MD-002 are non-credit courses and are excluded from GPA calculations. Everything else should participate in gpa calculations also Quran Translation courses which has 0.5 credit hours.

    5. If the required GPA returned by required_gpa_for_target is greater than 4.0, do not tell the student that the target is impossible immediately. Use get_remaining_credit_hours and expand the planning horizon appropriately until the required GPA is 4.0 or less, or there are no more remaining credit hours.

    6. Donot Automatically save reports unless the user explicitly asked for. But you can offer to save a report only after producing something worth keeping — a semester GPA, a projected CGPA, or a plan for reaching a target. Do not offer after a clarifying question or a single grade lookup, and never save without being asked.

    7. Always present final GPA and CGPA figures cleanly rounded to two decimal places (e.g., 3.46) in natural language. Never expose raw unrounded tool output, debug messages, or internal function details to the user.
"""


async def list_threads(memory: AsyncSqliteSaver) -> list[str]:
    """List all existing conversation threads in the memory database."""
    threads = set()

    async for checkpoint in memory.alist(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]

        threads.add(thread_id)

    return sorted(threads)


async def main():

    async with MCPAdapter(MCP_URL) as adapter:
        tools = await adapter.list_tools()

        print(f"Connected! Loaded {len(tools)} MCP tools.")

        # DATABASE PATH
        db_path = Path(__file__).resolve().parent / "memory.db"

        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as memory:

            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=SystemMessage(content=SYSTEM_PROMPT),
                checkpointer=memory
            )

            # Each student's name identifies their conversation thread.
            user_name = console.input(
                "\n[bold cyan]Enter your name:[/bold cyan] "
            ).strip().casefold()

            if not user_name:
                console.print("[red]Error: Name cannot be empty. Exiting.[/red]")
                return

            config = {
                "configurable": {
                    "thread_id": user_name,
                }
            }

            console.print(
                f"\n[green]Welcome, {user_name}![/green]"
            )

            while True:
                user_input = console.input(
                    f"\n[bold cyan]{user_name}:[/bold cyan] "
                ).strip()

                if not user_input:
                    continue

                # EXIT
                if user_input.lower() in {"exit", "quit"}:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                # Switch Conversation
                elif user_input.lower() == "/switch":
                    new_name = console.input(
                        "[cyan]Enter student name to switch to:[/cyan] "
                    ).strip().casefold()

                    if new_name:
                        user_name = new_name

                        config = {
                            "configurable": {
                                "thread_id": user_name,
                            }
                        }

                        console.print(
                            f"[green]Switched to {user_name}'s conversation.[/green]"
                        )

                # Show conversation history
                elif user_input.lower() == "/history":
                    threads = await list_threads(memory)

                    if threads:
                        console.print("\n[bold]Saved Conversations:[/bold]")
                        for name in threads:
                            console.print(f"- {name}")
                    else:
                        console.print("[yellow]No saved conversations found.[/yellow]")

                # Forget current conversation
                elif user_input.lower() == "/forget":
                    confirmation = console.input(
                        "[red]Are you sure you want to forget this conversation? (yes/no):[/red] "
                    ).strip().lower()

                    if confirmation == "yes":
                        await memory.adelete_thread(user_name)
                        console.print(
                            f"[green]Conversation for {user_name} has been forgotten.[/green]"
                        )
                    else:
                        console.print("[yellow]Forget operation cancelled.[/yellow]")

                else:
                    console.print("\n[bold green]Agent:[/bold green]")

                    streaming_text = False  # True while the answer is mid-line

                    async for chunk in agent.astream(
                        {
                            "messages": [
                                HumanMessage(content=user_input)
                            ]
                        },
                        config=config,
                        stream_mode=["updates", "messages"],
                        version="v2"
                    ):

                        # 1. WATCHING THE STEPS: UPDATES
                        if chunk["type"] == "updates":

                            for node, update in chunk["data"].items():

                                messages = update.get("messages", [])

                                if not messages:
                                    continue

                                msg = messages[-1]

                                if node == "model" and msg.tool_calls:
                                    if streaming_text:
                                        print()
                                        streaming_text = False

                                    for call in msg.tool_calls:
                                        console.print(
                                            f"[yellow]  ⚙ {call['name']}[/yellow]"
                                            f"[dim]({escape(str(call['args']))})[/dim]"
                                        )

                                elif node == "tools":
                                    if streaming_text:
                                        print()
                                        streaming_text = False

                                    result = " ".join(str(msg.content).split())
                                    if len(result) > 300:
                                        result = result[:300] + "..."

                                    console.print(
                                        f"[green]  ↳ {msg.name}:[/green] "
                                        f"[dim]{escape(result)}[/dim]"
                                    )

                        # 2. TYPING IT OUT: MESSAGES
                        elif chunk["type"] == "messages":

                            token, metadata = chunk["data"]

                            if (
                                metadata.get("langgraph_node") == "model"
                                and token.text
                            ):
                                if not streaming_text:
                                    print()  # blank line before the answer starts
                                    streaming_text = True

                                print(token.text, end="", flush=True)

                    # newline once, AFTER the whole response is done
                    print()


if __name__ == "__main__":
    asyncio.run(main())