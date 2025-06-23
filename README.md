# GAIA Agent

This project is a [LangGraph](https://www.langchain.com/langgraph) agent that answers questions from the [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) using Claude Sonnet 4 with a collection of tools. It currently uses web search and math functions for precise calculations.

>We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92\% vs. 15\% for GPT-4 equipped with plugins.

## Results

I began this as the final project of the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction). To pass the course, the agent needed to correctly answer 30% of a sample of 20 Level 1 questions. It scored 11/20, with two answers essentially timing out. On the class leaderboard of 4138, it scored the 1065th place (i.e. the top 25%) 🎉 

I bet we can do better!

Five of the questions the agent could not answer because it doesn't currently support reading files like images, pdf, mp3, etc. So this is next on the TODO list.

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
