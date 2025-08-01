import json
import os
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


class QuestionProvider:
    def __init__(self):
        self.gaia = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_level1",
            split="validation",
            trust_remote_code=True,
            streaming=False,
        )

    def summarize_dataset(self):
        for i in range(len(self.gaia)):
            question = self.gaia[i]
            print(f"Level {question["Level"]} files: {question["file_name"]}")
        print(len(self.gaia))

    def format_question(self, question):
        # This is the question format for the Hugging Face Agents Course:
        # https://agents-course-unit4-scoring.hf.space/docs#/default/get_questions_questions_get
        formatted_question = {
            "task_id": question["task_id"],
            "question": question["Question"],
            "file_name": question["file_name"],
            "Level": question["Level"],
        }
        return formatted_question, question["Final answer"]

    def get_question_count(self):
        return len(self.gaia)

    def get_questions(self):
        questions = self.gaia
        formatted_questions = []
        answers = []
        for question in questions:
            formatted_question, answer = self.format_question(question)
            formatted_questions.append(formatted_question)
            answers.append(answer)
        return formatted_questions, answers


class AnswerFileWriter:
    def __init__(self, answers_save_file):
        self.answers_save_file = answers_save_file
        logger.info(f"Initializing answer save file at {self.answers_save_file}")

    def __call__(self, answer_data):
        logger.info("Preparing to write to answers file...")

        if os.path.exists(self.answers_save_file):
            with open(self.answers_save_file, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
        if "answers" not in data or not isinstance(data["answers"], list):
            data["answers"] = []

        data["answers"].append(answer_data)
        with open(self.answers_save_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Answer file updated...")
