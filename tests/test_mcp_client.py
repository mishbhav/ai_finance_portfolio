import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["mcp_servers/pandas_server/server.py"],
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