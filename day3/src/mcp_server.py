"""
DAY 3 — MCP server.

READ FIRST:  ../06-fastmcp.md   then   ../07-skills-over-mcp.md

Do not continue until `uv run python src/mcp_server.py` serves on :8001
and a fastmcp Client can list your tools AND your skill resources.

Keep the two categories straight:
    TOOLS  = actions another agent can CALL   (@mcp.tool)
    SKILLS = knowledge another agent can READ (SkillsDirectoryProvider)

TODO:
  1. mcp = FastMCP("<your-name> Tools")
  2. Two @mcp.tool functions (calculate, word_stats — or your own).
  3. mcp.add_provider(SkillsDirectoryProvider(roots=<path to skills/>))
  4. __main__: mcp.run(transport="http", host="0.0.0.0", port=8001)
"""

# TODO
"""
FastMCP server for the Day 3 agent project.

This service exposes utility functions as MCP tools so that
MCP-compatible agents and clients can discover and call them.
"""

import ast
import operator

from fastmcp import FastMCP


# Create the MCP server
mcp = FastMCP("Sadeem Agent Utilities")


# Arithmetic operations allowed by the calculator
SUPPORTED_OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def evaluate_arithmetic(node):
    """Safely evaluate a basic arithmetic syntax tree."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

    if isinstance(node, ast.BinOp):
        operation_type = type(node.op)

        if operation_type in SUPPORTED_OPERATIONS:
            left_value = evaluate_arithmetic(node.left)
            right_value = evaluate_arithmetic(node.right)

            return SUPPORTED_OPERATIONS[operation_type](left_value, right_value)

    if isinstance(node, ast.UnaryOp):
     operation_type = type(node.op)

    if operation_type in SUPPORTED_OPERATIONS:
        operand_value = evaluate_arithmetic(node.operand)
        return SUPPORTED_OPERATIONSoperand_value
            

    raise ValueError(
        "Only basic numerical arithmetic operations are supported."
    )


@mcp.tool
def calculate(expression: str) -> float:
    """
    Calculate a basic arithmetic expression safely.

    Example:
        calculate("2 * (3 + 4) ** 2")
    """

    expression_tree = ast.parse(expression, mode="eval")
    return evaluate_arithmetic(expression_tree.body)


@mcp.tool
def word_stats(text: str) -> dict:
    """
    Return basic statistics for the provided text.

    The result includes the number of words, characters,
    non-empty lines, and unique words.
    """

    words = text.split()
    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    normalized_words = {
        word.lower().strip(".,!?;:")
        for word in words
    }

    return {
        "word_count": len(words),
        "character_count": len(text),
        "line_count": len(lines) if lines else 1,
        "unique_word_count": len(normalized_words),
    }


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8001,
    )