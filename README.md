# GPA & CGPA Conversational Agent

An AI-powered conversational assistant that helps university students calculate Semester GPA, project CGPA, determine the GPA required to achieve a target CGPA, and retrieve academic course information.

Built using **LangChain, LangGraph, FastMCP, Groq, and SQLite**.

## Features

- **Semester GPA Calculation:** Calculate GPA using students' marks and official course credit hours.
- **CGPA Projection:** Estimate future CGPA based on completed and remaining credit hours.
- **Target CGPA Planning:** Determine the GPA required in upcoming semesters to achieve a target CGPA.
- **Course Information:** Retrieve semester-wise courses and their credit hours.
- **MCP Integration:** Expose academic calculation functions through a FastMCP server.
- **Persistent Conversation Memory:** Store conversations using LangGraph's SQLite checkpointer.
- **Conversation Management:** Switch between student conversations, view saved conversations, and delete conversation history.
- **Streaming Responses:** Display agent responses and tool execution updates as they happen.

## Tech Stack

- Python
- LangChain
- LangGraph
- FastMCP
- Groq API
- SQLite
- Rich
- Python-dotenv

## Project Structure

```text
BCSF24M022_Assignment3/
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── llm.py
│   └── .env
│
├── server/
│   ├── __init__.py
│   ├── server.py
│   └── tools.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/WaleedImran2007/gpa-cgpa-conversational-agent.git
cd gpa-cgpa-conversational-agent
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file inside the `agent/` directory:

```text
GROQ_API_KEY=your_groq_api_key
MCP_URL=http://127.0.0.1:8000/mcp
```

Replace `your_groq_api_key` with your actual Groq API key.

**Important:** Never commit your `.env` file or expose your API key publicly.

## Running the Application

The application consists of two components: the MCP server and the conversational agent.

### 1. Start the MCP Server

Open the first terminal in the project root and run:

```bash
python server/server.py
```

The MCP server will start locally at:

```text
http://127.0.0.1:8000/mcp
```

### 2. Start the Conversational Agent

Open a second terminal, activate the virtual environment, and run:

```bash
python -m agent.agent
```

The agent will connect to the MCP server and load the available academic tools.

Enter your name to start or resume your conversation.

## Conversation Commands

| Command | Description |
|---|---|
| `/switch` | Switch to another student's conversation |
| `/history` | Display saved conversations |
| `/forget` | Delete the current conversation |
| `exit` or `quit` | Close the application |

Conversations are stored persistently in a local SQLite database.

## Example Queries

- Calculate my Semester 5 GPA. My marks are 85, 90, 78, 92, 88, 75, 95.
- What courses do I have in Semester 6?
- Calculate my projected CGPA after Semester 6.
- What GPA do I need in my remaining semesters to achieve a 3.7 CGPA?

---

**Developed as part of the Artificial Intelligence course assignment.**