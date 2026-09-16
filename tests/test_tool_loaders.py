import pytest

from ai.loader.base import ToolCapabilityError, ToolOutputError
from ai.loader.gguf_loader import GGUFLoader
from ai.loader.hf_loader import HFLoader
from ai.msgHandler import validate_tool_output


def test_hf_hermes_parser_handles_nested_json_and_truncation() -> None:
    loader = HFLoader()
    loader._tool_parser = "hermes_json"
    content, calls = loader._parse_tool_output(
        '稍等 <tool_call>{"name":"lookup","arguments":{"query":{"id":1}}}</tool_call>'
    )
    assert content == "稍等"
    assert calls[0]["function"]["arguments"] == '{"query":{"id":1}}'
    with pytest.raises(ToolOutputError):
        loader._parse_tool_output('<tool_call>{"name":"lookup"}')


def test_gguf_tool_failure_does_not_fall_back_to_text_generation() -> None:
    class BrokenModel:
        def create_chat_completion(self, **kwargs):
            raise TypeError("unsupported")

        def __call__(self, *args, **kwargs):
            raise AssertionError("tool request must not use text fallback")

    loader = GGUFLoader()
    loader._model = BrokenModel()
    loader._chat_format = "chatml-function-calling"
    with pytest.raises(ToolCapabilityError):
        loader.generate("", messages=[{"role": "user", "content": "hi"}],
                        tools=[], tool_choice="none")


def test_tool_output_constraints_are_enforced() -> None:
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    call = {"id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": {"id": 1}}}
    normalized = validate_tool_output([call], tools, "required", False)
    assert normalized[0]["function"]["arguments"] == '{"id":1}'
    with pytest.raises(ToolOutputError):
        validate_tool_output([], tools, "required", True)
    with pytest.raises(ToolOutputError):
        validate_tool_output([call, call], tools, "auto", True)
