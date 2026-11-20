from fastmcp import FastMCP
from tools import TOOLS

import os

mcp = FastMCP("CGPA Guider")

for fn in TOOLS:
    mcp.tool(fn)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))

    mcp.run(
        transport="http", 
        host="0.0.0.0",
        port=port, 
        path="/mcp"
    )