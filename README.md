# Agenta: Advanced AI Agent System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

A sophisticated multi-agent AI system powered by reinforcement learning and GPT-4, designed for complex task execution and continuous learning.

## Key Features

- **Multi-Agent Architecture**

  - Strategic Planning Agent
  - Research & Information Agent
  - Calculation & Analysis Agent
  - Formatting & Presentation Agent

- **Advanced Capabilities**

  - Neural Search with Exa AI
  - Reinforcement Learning with Experience Replay
  - Persistent Memory Management
  - Dynamic Policy Optimization
  - Context-Aware Decision Making

- **Performance & Learning**
  - Continuous Performance Monitoring
  - Strategy Effectiveness Analysis
  - Automated Policy Refinement
  - Resource Usage Optimization

## Quick Start

1. **Setup Environment**

   ```bash
   # Clone repository
   git clone https://github.com/yourusername/agenta.git
   cd agenta

   # Create virtual environment
   python -m venv agenta
   source agenta/bin/activate  # Linux/Mac
   # or
   .\agenta\Scripts\activate  # Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure API Keys**

   ```bash
   cp .env.template .env
   ```

   Add your API keys to `.env`:

   ```
   OPENAI_API_KEY=your_openai_key
   EXA_API_KEY=your_exa_key
   MEM0_API_KEY=your_mem0_key
   ```

3. **Run the Agent**
   ```bash
   python cli.py run "your objective here"
   ```

## 💡 Usage Examples

```bash
# Basic task execution
python cli.py run "analyze the latest market trends for AI companies"

# Verbose output for detailed insights
python cli.py run "calculate ROI for Project X" --verbose

# View system configuration
python cli.py info
```

## 🛠 Architecture

```
agenta/
├── agent.py          # Core agent implementation
├── crew_agents.py    # Specialized agent definitions
├── memory_manager.py # Memory and learning systems
├── formatting.py     # Output formatting
└── cli.py           # Command-line interface
```

## 🔧 Configuration

| Parameter         | Default             | Description                            |
| ----------------- | ------------------- | -------------------------------------- |
| AGENT_TEMPERATURE | 0.7                 | Controls randomness in decision making |
| MAX_ITERATIONS    | 5                   | Maximum steps per objective            |
| MODEL             | gpt-4-turbo-preview | Language model used                    |

## Advanced Features

### Memory Management

- Experience replay buffer for reinforcement learning
- Policy management with confidence scoring
- Performance metrics tracking
- Strategy pruning for optimization

### Agent Capabilities

- Neural search with semantic understanding
- Context-aware decision making
- Dynamic agent selection and handoff
- Continuous learning from experiences

Built using OpenAI, Exa AI, and Mem0

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
