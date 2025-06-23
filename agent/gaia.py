import logging
import asyncio
import time

from pydantic import BaseModel
from typing import Annotated

from langchain_anthropic import ChatAnthropic, convert_to_anthropic_tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, ToolNode

from utils.file_extractors import FileExtractor


logger = logging.getLogger(__name__)

llm = ChatAnthropic(
    model_name="claude-sonnet-4-20250514",
    max_tokens=5000,
    timeout=None,
    thinking={"type": "enabled", "budget_tokens": 4000},
    model_kwargs={
        "extra_headers": {"anthropic-beta": "token-efficient-tools-2025-02-19"}
    },
)


class AgentState(BaseModel):
    question: dict
    final_agent_answer: dict | None
    messages: Annotated[list, add_messages]


@tool
def add(x: float, y: float):
    """This function adds two numbers."""
    logger.info(f"Added {x} and {y}")
    return x + y


@tool
def subtract(x: float, y: float):
    """This function subtracts two numbers."""
    logger.info(f"Subtracting {y} from {x}")
    return x - y


@tool
def multiply(x: float, y: float):
    """This function multiplies two numbers."""
    logger.info(f"Multiplying {x} and {y}")
    return x * y


@tool
def divide(x: float, y: float):
    """this function divides two numbers. handles division by zero."""
    logger.info(f"dividing {x} by {y}")
    if y == 0:
        return "error: cannot divide by zero."
    return x / y


@tool
def submit_final_answer(
    answer: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AgentState, InjectedState],
):
    """This function should be called to submit your final answer only once you have tetermined it. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

    logger.info(f"Submitting final answer: {answer}")

    answer_data = {
        "task_id": state.question["task_id"],
        "agent_answer": answer,
    }

    logger.info("Final answer written, updating state with final answer...")
    return Command(
        update={
            "final_agent_answer": answer_data,
            "messages": [
                ToolMessage(
                    "You have successfully submitted your final answer. There is nothing left to be done.",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


tavily = TavilySearch(max_results=2)
tools = [add, subtract, multiply, divide, tavily, submit_final_answer]
anthropic_tools = []
for raw_tool in tools:
    anthropic_tool = convert_to_anthropic_tool(raw_tool)
    anthropic_tools.append(anthropic_tool)

# To cache all tools we add the cache control block to the last tool
anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
llm_with_tools = llm.bind_tools(anthropic_tools)


class GaiaAgent:
    def __init__(self, *, handle_message_chunk=None):
        self.handle_message_chunk = handle_message_chunk
        self.llm = llm_with_tools
        self.agent_graph = self.compile_graph()

    def compile_graph(self):
        graph = StateGraph(AgentState)

        def should_continue(state):
            logger.info(
                "Checking for final answer in decide_next_node conditional edge"
            )
            logger.info(state.final_agent_answer)
            if state.final_agent_answer:
                logger.info("Final answer submitted. Ending agent flow.")
                return END
            else:
                logger.info("No final answer submitted yet, proceed to the tool nodes.")
                return "tools"

        graph.add_node(self.consider_question)
        graph.add_node("tools", ToolNode(tools))

        graph.add_edge(START, "consider_question")
        graph.add_edge("tools", "consider_question")
        graph.add_conditional_edges(
            "consider_question", should_continue, ["tools", END]
        )
        return graph.compile()

    async def consider_question(self, state: AgentState):
        """Home of the agent. Looks at all the messages so far, generates the next message."""
        logger.info("Considering question...")
        time.sleep(5)  # Avoid hitting rate limits on the Claude API
        if state.final_agent_answer is None:
            messages = state.messages
            response = await self.llm.ainvoke(messages)
            return {"messages": [response]}
        else:
            # If a final answer has been determined no more consideration is required
            logger.info(
                "Skipping question consideration because final answer is available"
            )
            return state

    async def answer_question(self, question):
        file_contents = None
        question_text = question["question"]
        if question["file_name"]:
            try:
                file_extractor = FileExtractor(question["file_name"])
                file_contents = file_extractor()
                question_text += "\n\nDocument contents:\n\n"
                question_text += file_contents
            except Exception as e:
                logger.error(f"Error: {e}")
                return "I don't know - I can't handle this file!"

        logger.debug("Initializing agent state to answer question...")
        system_prompt = """
        You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer by calling the submit_final_answer tool. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        To operate effectively, always remember:
            1. Before using any math tools for operations, make sure you have thought about the math problem sufficiently and stated the equation that you will solve. Plan the equation first, then use the math tools to solve it precisely.
        """
        initial_state = {
            "question": question,
            "final_agent_answer": None,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": question_text,
                },
            ],
        }

        async def get_final_answer(agent):
            final_output: dict | None = None
            async for mode, chunk in agent.astream(
                initial_state,
                stream_mode=["values", "messages"],
                config={"recursion_limit": 30},
            ):
                if mode == "values":
                    final_output = chunk
                if mode == "messages" and self.handle_message_chunk:
                    self.handle_message_chunk(chunk)

            if final_output is None:
                return "I don't know!"

            return final_output["final_agent_answer"]["agent_answer"]

        result = await get_final_answer(self.agent_graph)
        return result

    def __call__(self, question):
        return asyncio.run(self.answer_question(question))
