import logging
import os

from agent.gaia import GaiaAgent
from utils.questions import QuestionProvider, AnswerFileWriter
from utils.stream_handlers import MessageChunkPrinter


logging.basicConfig(level=os.getenv("LOGLEVEL", "WARNING"))
logger = logging.getLogger(__name__)


def main():
    question_provider = QuestionProvider()
    question, ground_truth = question_provider.get_question()
    logger.info(f"Running query for question {question["question"]}")

    agent = GaiaAgent(handle_message_chunk=MessageChunkPrinter())
    agent_answer = agent(question)
    logger.info(f"Agent answer: {agent_answer}")

    submit_answer = AnswerFileWriter()
    submit_answer(
        {
            "task_id": question["task_id"],
            "question": question["question"],
            "agent_answer": agent_answer,
            "ground_truth": ground_truth,
            "is_correct": agent_answer == ground_truth,
        }
    )


if __name__ == "__main__":
    main()
