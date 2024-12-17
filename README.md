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

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.template` to `.env` and add your OpenAI API key:

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

Key components:

- Epsilon-greedy exploration (20% random actions by default)
- Q-learning updates with configurable learning rate and discount factor
- Integration with LangChain for LLM capabilities
- Custom tools for various tasks
