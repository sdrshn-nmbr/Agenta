# RL-Based Agent CLI

A command-line interface tool that uses reinforcement learning principles to accomplish tasks. The agent learns from experience using Q-learning and combines it with LLM capabilities.

## Features

- Q-learning based decision making
- Epsilon-greedy exploration strategy
- Persistent learning across sessions
- Configurable parameters via environment variables
- Rich CLI interface

## Setup

1. Clone the repository

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Copy `.env.template` to `.env` and add your OpenAI API key:

```bash
cp .env.template .env
```

## Usage

Run the agent with an objective:

```bash
python cli.py run "calculate 2 + 2"
```

Show verbose output:

```bash
python cli.py run "search for python tutorials" --verbose
```

View agent configuration:

```bash
python cli.py info
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `AGENT_TEMPERATURE`: Temperature for LLM responses (default: 0.7)
- `MAX_ITERATIONS`: Maximum steps per objective (default: 5)

## How it Works

The agent uses Q-learning to make decisions about which tools to use for a given objective. It maintains a Q-table that maps state-action pairs to expected rewards, learning from experience which actions are most effective in different situations.

### TODO

- Code Docs RAG Search (for documentation)
- Code Interpreter (for testing/running code)
- Github Search (for code examples/solutions)
- Directory RAG Search (for codebase navigation)
- File Read/Write (for code manipulation)
- Memory Manager (for storing and retrieving memories)
- Reward Manager (for storing and retrieving rewards)
- Q-Learning (for decision making)
- Epsilon-Greedy Exploration (for exploration strategy)
- LLM (for reasoning and decision making)
