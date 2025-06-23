import os
import logging
from huggingface_hub import snapshot_download
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

logger = logging.getLogger(__name__)

supported_file_types = ["docx"]


class FileExtractor:
    def __init__(self, file_name):
        directory = os.path.join(os.path.dirname(__file__), "../", "downloaded_files")
        os.makedirs(directory, exist_ok=True)
        snapshot_download(
            repo_id="gaia-benchmark/GAIA", repo_type="dataset", local_dir=directory
        )

        self.file_name = file_name
        self.file_path = os.path.join(directory, "2023/validation", self.file_name)

        logger.info(f"Creating file extractor for {file_name}")
        is_supported = self.is_file_supported()
        if not is_supported:
            raise Exception(
                f"Unable to parse file {file_name}: file type not supported"
            )

    def get_extension(self):
        return os.path.splitext(self.file_name)[1].lstrip(".").lower()

    def is_file_supported(self):
        extension = self.get_extension()
        logger.info(f"Checking for file support for {extension}...")

        if extension in supported_file_types:
            logger.info("File type is supported.")
            return True

        logger.info("File type is not supported.")
        return False

    def iter_block_items(self, parent):
        for child in parent.element.body:
            if child.tag.endswith("p"):
                yield Paragraph(child, parent)
            elif child.tag.endswith("tbl"):
                yield Table(child, parent)

    def dump_doc(self, path):
        doc = Document(path)
        lines = []
        for block in self.iter_block_items(doc):
            if isinstance(block, Paragraph):
                style = block.style.name
                text = "".join(run.text for run in block.runs).strip()
                if not text:
                    continue
                if style.startswith("Heading"):
                    lines.append(f"# {text}")
                else:
                    lines.append(text)
            elif isinstance(block, Table):
                for row in block.rows:
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def __call__(self):
        logger.info(f"Extracting file contents from {self.file_name}...")
        if os.path.exists(self.file_path):
            logger.info("File exists on disk.")
            text = self.dump_doc(self.file_path)
            logger.info("Text extracted from file.")
            return text
        else:
            raise Exception("File does not exist on disk.")
