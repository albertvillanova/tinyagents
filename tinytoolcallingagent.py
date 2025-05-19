import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Optional

from huggingface_hub import AsyncInferenceClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SYSTEM_PROMPT = """You are a helpful assistant that can solve any task."""


class TinyToolCallingAgent:
    def __init__(self):
        # Initialize MCP client session and (LLM) model
        self.mcp_client_session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model = AsyncInferenceClient(model="Qwen/Qwen2.5-Coder-32B-Instruct", provider="hf-inference")
        self.system_prompt = SYSTEM_PROMPT
        self.tools = []

    async def _call_model(self, messages, tools=None):
        """Call the model with the given messages and tools.

        Args:
            messages: List of messages to send to the model.
            tools: List of tools available for the model to use.
        """
        return await self.model.chat_completion(messages, max_tokens=1000, tools=tools)

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if not server_script_path.endswith(".py") and not server_script_path.endswith(".js"):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if server_script_path.endswith(".py") else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.mcp_client_session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.mcp_client_session.initialize()

        # List available tools
        response = await self.mcp_client_session.list_tools()
        self.tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    async def process_query(self, query: str) -> str:
        """Process a query using model and available tools"""
        # Create messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        # List available tools for the model call
        tools = [
            {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description, "parameters": tool.inputSchema},
            }
            for tool in self.tools
        ]

        # Initial model call with tools
        response = await self._call_model(messages, tools=tools)
        # print("Initial response:", response)

        # Process response and handle tool calls
        final_text = []
        message = response.choices[0].message
        if message.tool_calls:
            if message.content:
                messages.append({"role": "assistant", "content": message.content})
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                # Execute tool call
                result = await self.mcp_client_session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                # Continue conversation with tool results
                messages.append({"role": "user", "content": result.content[0].text})
            # Get next response from model
            response = await self._call_model(messages)
            final_text.append(response.choices[0].message.content)
        elif message.content:
            final_text.append(message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!\nType your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {type(e).__name__}: {e}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_server_script>")
        sys.exit(1)
    agent = TinyToolCallingAgent()
    try:
        await agent.connect_to_server(sys.argv[1])
        await agent.chat_loop()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
