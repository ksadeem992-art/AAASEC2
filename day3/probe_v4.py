
import asyncio

from fastmcp import Client


async def probe(mode):
    async with Client(
        "http://localhost:8001/mcp",
        mode=mode,
    ) as client:
        tools = await client.list_tools()

        print(
            f"mode={mode!r} -> {len(tools)} tools:",
            [tool.name for tool in tools],
        )


async def main():
    await probe("auto")
    await probe("legacy")


if __name__ == "__main__":
    asyncio.run(main())