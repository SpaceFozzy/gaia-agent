# GAIA Agent

This project is a [LangGraph](https://www.langchain.com/langgraph) agent that attempts to answer questions from the [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) using Claude Sonnet 4 with a collection of tools.

## Getting Started

### Environment Secrets
You will need the following API keys set in your env:
* `ANTHROPIC_API_KEY`
* `TAVILY_API_KEY`
  
You will also need a Hugging Face login for access to the dataset.

### Running

1. `uv sync`
2. `python3 main.py`
3. This will answer [a pre-determined question](https://github.com/SpaceFozzy/gaia-agent/blob/30f4ce7e31a9b569891de130bfaa8dca7ed50c4c/utils/questions.py#L26), printing the LLM's messages to standard out. When the agent submits its answer, it will be recorded in `/answers/answers.json`.

## Details

### LLM

This project currently supports Claude Sonnet 4, using the following Claude-specific features:
* a thinking token budget
* token-efficient tool-use
* prompt caching

### Tools

* Math tools (calculator)
* Web search (currently via Tavily)
