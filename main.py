import logging
import os
import mlflow
import subprocess
from agent.gaia import GaiaAgent
from utils.questions import QuestionProvider, AnswerFileWriter
from utils.stream_handlers import MessageChunkPrinter
from typing import Dict, Union


logging.basicConfig(level=os.getenv("LOGLEVEL", "ERROR"))
logger = logging.getLogger(__name__)

mlflow.set_experiment("gaia-agent")
mlflow.langchain.autolog()
mlflow.anthropic.autolog()


def get_git_info() -> Dict[str, Union[str, bool]]:
    """Get git information"""
    try:
        return {
            "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip(),
            "short_hash": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"]
            )
            .decode("utf-8")
            .strip(),
            "has_uncommitted_changes": len(
                subprocess.check_output(["git", "status", "--porcelain"])
                .decode("utf-8")
                .strip()
            )
            > 0
        }
    except subprocess.CalledProcessError as e:
        return {"error": f"Git command failed: {e}"}


def main():
    git_info = get_git_info()
    with mlflow.start_run():
        current_run = mlflow.active_run()
        mlflow.log_param("commit_hash", git_info['commit_hash'])
        mlflow.log_param("commit_short", git_info['short_hash'])
        mlflow.log_param("uncommitted_changes", git_info['has_uncommitted_changes'])

        question_provider = QuestionProvider()
        question, ground_truth = question_provider.get_question()
        logger.info(f"Running query for question {question["question"]}")
        logger.info(question["file_name"])

        agent = GaiaAgent(handle_message_chunk=MessageChunkPrinter())
        agent_answer = agent(question)
        logger.info(f"Agent answer: {agent_answer}")

        answers_artifact_directory = os.path.join(os.path.dirname(__file__), "answers")
        os.makedirs(answers_artifact_directory, exist_ok=True)
        answers_save_file = os.path.join(answers_artifact_directory, f"{current_run.info.run_name}.json")

        submit_answer = AnswerFileWriter(answers_save_file)
        is_correct = agent_answer == ground_truth
        submit_answer(
            {
                "task_id": question["task_id"],
                "question": question["question"],
                "agent_answer": agent_answer,
                "ground_truth": ground_truth,
                "is_correct": agent_answer == ground_truth,
            }
        )
        total_correct = 0
        if is_correct:
            total_correct += total_correct

        mlflow.log_artifact(answers_save_file)
        mlflow.log_metric("total_correct", is_correct)


if __name__ == "__main__":
    main()
