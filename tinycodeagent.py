import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SYSTEM_PROMPT = """You are a helpful assistant that can solve any task using Python code blobs.

Additionally, you can call the following Python functions:
{tools}
""".strip()

class TinyCodeAgent:
    def __init__(self):
        # Initialize MCP client session and (LLM) model
        self.mcp_client_session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model = self._init_model()
        self.system_prompt = SYSTEM_PROMPT

    @staticmethod
    def _init_model():
        """Initialize the model client"""
        from huggingface_hub import InferenceClient

        return InferenceClient(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            provider="hf-inference",
        )

    async def _call_model(self, messages):
        """Call the model with the given messages and tools.

        Args:
            messages: List of messages to send to the model.
        """
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.chat_completion(
                messages,
                max_tokens=1000,
            )
        )
        return response

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.mcp_client_session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.mcp_client_session.initialize()

        # List available tools
        response = await self.mcp_client_session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using model and available tools"""
        # List available tools
        response = await self.mcp_client_session.list_tools()
        tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        tools_str = "\n".join([f"{tool['function']['name']}: {tool['function']['description']}" for tool in tools])

        # Create messages
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(tools=tools_str),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        # Initial model call without tools
        response = await self._call_model(messages)
        # print("Initial response:", response)

        # Process response
        final_text = []
        message = response.choices[0].message
        final_text.append(message.content)
        # TODO: Execute Python code

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_server_script>")
        sys.exit(1)

    agent = TinyCodeAgent()
    try:
        await agent.connect_to_server(sys.argv[1])
        await agent.chat_loop()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
