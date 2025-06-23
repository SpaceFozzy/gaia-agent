import logging

logger = logging.getLogger(__name__)


class MessageChunkPrinter:
    def handle_thinking_type_contents(self, contents):
        if "thinking" in contents:
            print(
                contents["thinking"],
                end="",
                flush=True,
            )
        elif "signature" in contents:
            pass
        else:
            print(f"unknown thinking node {contents}")

    def handle_typed_chunk_contents(self, contents):
        match contents["type"]:
            case "thinking":
                self.handle_thinking_type_contents(contents)
            case "text":
                print(contents["text"], end="", flush=True)
            case "tool_use":
                print(f"\n\nUsing {contents['name']}\n\n")
            case "input_json_delta":
                pass
            case _:
                print(f"unknown node {contents}")

    def handle_single_character_chunk_contents(self, contents, metadata):
        if "langgraph_node" in metadata:
            if (
                metadata["langgraph_node"] != "tools"
                and metadata["langgraph_node"] != "consider_question"
            ):
                print(contents)
                print(metadata)
            else:
                pass

    def handle_unknown_chunk_contents(self, contents):
        print(f"unknown node {contents} has no type")

    def __call__(self, chunk):
        message_chunk, metadata = chunk
        if message_chunk.content:
            for contents in message_chunk.content:
                if "type" in contents:
                    self.handle_typed_chunk_contents(contents)
                elif len(contents) == 1:
                    self.handle_single_character_chunk_contents(contents, metadata)
                else:
                    self.handle_unknown_chunk_contents(contents)
        elif message_chunk.response_metadata:
            if "stop_reason" in message_chunk.response_metadata:
                logger.info(
                    f"Stopping, reason: {message_chunk.response_metadata["stop_reason"]}"
                )
                # Don't print when the LLM stops for whatever reason
                pass
            else:
                print(
                    f"unknown message chunk with no content, but metadata: {message_chunk.response_metadata}"
                )
        else:
            print(
                f"unknown message chunk has no content or response_metadata: {message_chunk}"
            )
