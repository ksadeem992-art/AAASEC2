import asyncio

from fastmcp import Client


async def main():
    async with Client("http://localhost:8001/mcp") as client:
        resources = await client.list_resources()

        print("Available skill resources:")
        for resource in resources:
            print(resource.uri)

        content = await client.read_resource(
            "skill://research-brief/SKILL.md"
        )

        print("\nResearch brief skill preview:")
        print(content[0].text[:200])


if __name__ == "__main__":
    asyncio.run(main())