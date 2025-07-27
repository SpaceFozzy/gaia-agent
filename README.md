# GAIA Agent

This project is a [LangGraph](https://www.langchain.com/langgraph) agent that answers questions from the [GAIA dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) using Claude Sonnet 4 with a collection of tools. It currently uses web search and math functions for precise calculations.

>We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92\% vs. 15\% for GPT-4 equipped with plugins.

## Results

I began this as the final project of the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction). To pass the course, the agent needed to correctly answer 30% of a sample of 20 Level 1 questions. It scored 11/20, with two answers essentially timing out. On the class leaderboard of 4138, it scored the 1065th place (i.e. the top 25%) 🎉 This was without external file support.

After adding some improvements including expanded file support, on a second run it scored 16/20, ranking #725 of 4257 (the top %18) 🎉

## Getting Started

### Environment Secrets
You will need the following API keys set in your env:
* `ANTHROPIC_API_KEY`
* `TAVILY_API_KEY`
  
You will also need a Hugging Face login for access to the dataset.

### Running

1. `uv sync`
3. `python3 main.py`
4. This will answer all questions in the [2023_level1 dataset](https://github.com/SpaceFozzy/gaia-agent/blob/9c9a06f96a2e0c8378af66b8624eaf1ffe9a431d/utils/questions.py#L13), printing the LLM's messages to standard out and recording traces / metrics with mlflow. When the agent submits its answers, they will be recorded in `/answers` in a json file named after the run. This answer file is also logged as an artifact with mlflow.

## Details

### MLflow

`mlflow ui` will run MLFlow to track your runs and provide tracing. 

<img width="1280" height="679" alt="Screenshot from 2025-07-26 21-52-07" src="https://github.com/user-attachments/assets/663b4849-aadd-4407-96e9-abb5809ee13d" />

### LLM

This project currently supports Claude Sonnet 4, using the following Claude-specific features:
* a thinking token budget
* token-efficient tool-use
* prompt caching

### Tools

* Math tools (calculator)
* Web search (currently via Tavily)

### Document Support

The agent currently supports the following [file extension types](https://github.com/SpaceFozzy/gaia-agent/blob/b5486151dbe98088eacd4f866ceeaf073069ca6c/utils/file_extractors.py#L14):

* docx
* xlsx
* py
* mp3

Images are't supported yet but coming soon.

