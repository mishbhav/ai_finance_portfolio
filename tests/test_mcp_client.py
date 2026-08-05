import asyncio
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT_DIR = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = ROOT_DIR / "mcp_servers" / "pandas_server" / "server.py"

server_params = StdioServerParameters(
    command=sys.executable,          # the exact interpreter running this script,
    args=[str(SERVER_SCRIPT)],       # not a bare "python" that may not be on PATH
    cwd=str(ROOT_DIR),
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()          # handshake

            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            result = await session.call_tool(
                "get_volatility",
                {"prices": [100, 102, 101, 105, 103]},
            )
            print("Result:", result)

asyncio.run(main())