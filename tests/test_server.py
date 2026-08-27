from ai.utils.server import ChatRequest


def test_request_defaults() -> None:
    request = ChatRequest(prompt="hello")
    assert request.special == 0 and request.think == 0 and request.stream is False
