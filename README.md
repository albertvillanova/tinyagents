# tinyagents
TinyAgents: LLM + MCP Tools

**TinyAgents** is a minimalist implementation of agents powered by LLMs and [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) tools.

This project is inspired by the [MCP Client Quickstart](https://modelcontextprotocol.io/quickstart/client#python) and provides a lightweight foundation for building LLM-based agent workflows.

## Agent Implementations

The repository includes two different agent implementations:

### TinyToolCallingAgent

`TinyToolCallingAgent` is a general-purpose agent that can solve tasks by calling external tools. It:

- Connects to Python or JavaScript MCP servers
- Processes user queries using the Qwen2.5-Coder-32B-Instruct model
- Dynamically discovers and calls tools provided by the MCP server
- Handles tool call results and continues the conversation
- Provides an interactive chat loop for user interaction

Usage:
```python
python tinytoolcallingagent.py <path_to_server_script>
```

### TinyCodeAgent

`TinyCodeAgent` is designed to solve tasks using Python code. It connects to an MCP server to access tools and can:

- Connect to Python or JavaScript MCP servers
- Process user queries using the Qwen2.5-Coder-32B-Instruct model
- Generate Python code solutions
- Execute Python code and display the results (TODO)
- Provide an interactive chat loop for user interaction

Usage:
```python
python tinycodeagent.py <path_to_server_script>
```

## Common Features

Both agents share these capabilities:
- Asynchronous operation using Python's asyncio
- Connection to MCP servers via stdio
- Interactive chat interface
- Dynamic tool discovery
- Integration with Hugging Face's InferenceClient

## Included Example: Weather Server

The repository includes an example MCP server implementation in the `servers/weather` directory. This server provides tools for accessing weather data from the National Weather Service API:

- `get_alerts`: Retrieves weather alerts for a specified US state
- `get_forecast`: Gets a detailed weather forecast for a location based on latitude and longitude

To use the weather server with one of the agents:

```bash
# With TinyToolCallingAgent
python tinytoolcallingagent.py servers/weather/weather.py

# With TinyCodeAgent
python tinycodeagent.py servers/weather/weather.py
```

## Future Enhancements

### Python Code Execution

The TinyCodeAgent should include a basic Python code executor that:
- Automatically extracts Python code blocks from the LLM's response
- Executes the code in a controlled environment
- Captures and displays standard output and error streams
- Reports execution status (success or failure)

This feature will enable users to immediately see the results of code solutions provided by the agent, making it more interactive and useful for programming tasks.

## Requirements

- Python 3.10+
- mcp >= 1.9.0
- huggingface-hub >= 0.31.2
- httpx (for the weather server example)

## Getting Started

1. Install the required dependencies
2. Set up an MCP server (use the included weather server or create your own)
3. Run one of the agent implementations pointing to your server script

Example queries for the weather server:
- "What are the current weather alerts in New York and California?"
- "What's the forecast for latitude 37.7749 and longitude -122.4194?"

> **Note**: The [MCP server](https://modelcontextprotocol.io/quickstart/server) is intended for testing and development purposes only.
