import json

from fastapi.testclient import TestClient

from ai.loader.models_mgr import ModelsMgr
from ai.msgHandler import validate_tool_request
from ai.utils.serverApi import OpenAIChatRequest, serverApi


class CaptureHandler:
    def __init__(self, response: str = "ok", **result) -> None:
        self.response = response
        self.result = result
        self.calls = []

    async def handle(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"status": "ok", "response": self.response,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}, **self.result}


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "查询天气",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}}, "required": ["city"]},
}}]


def test_request_defaults() -> None:
    request = OpenAIChatRequest()
    assert request.message_id == 0 and request.args is None and request.stream is False


def test_server_api_port(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({
        "defaults": {"server": {"api_port": 8000}},
        "models": {"demo": {"path": str(tmp_path / "demo.gguf")}},
    }), encoding="utf-8")
    manager = ModelsMgr(str(config_path))

    api = serverApi(manager)
    client = TestClient(api.app)
    assert api._port() == 8000
    assert client.get("/").json()["openai_base_url"] == "/v1"
    assert client.get("/v1/models").json()["data"][0]["id"] == "demo"


def test_openai_bearer_auth(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({
        "defaults": {"server": {"api_key": "secret"}},
        "models": {"demo": {"path": str(tmp_path / "demo.gguf")}},
    }), encoding="utf-8")
    manager = ModelsMgr(str(config_path))
    client = TestClient(serverApi(manager).app)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_openwebui_generation_parameters_are_normalized(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    handler = CaptureHandler("answer")
    client = TestClient(serverApi(ModelsMgr(str(config_path)), handler).app)
    response = client.post("/v1/chat/completions", json={
        "model": "demo", "system_prompt": "be concise",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2, "top_k": 20, "min_p": 0.1,
        "stop": "END", "seed": 7, "max_completion_tokens": 64,
        "reasoning_effort": "low", "unknown_openwebui_field": True,
    })

    assert response.status_code == 200
    kwargs = handler.calls[0][1]
    assert kwargs["prompt"] == "system: be concise\nuser: hello"
    assert kwargs["think"] == 1
    assert kwargs["deploy"] == {
        "temperature": 0.2, "top_k": 20, "min_p": 0.1,
        "seed": 7, "max_tokens": 64, "stop_sequences": ["END"],
    }
    assert response.json()["usage"]["total_tokens"] == 3


def test_openwebui_stream_is_sse_and_chunked(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    client = TestClient(serverApi(ModelsMgr(str(config_path)), CaptureHandler("abcdef")).app)
    response = client.post("/v1/chat/completions", json={
        "model": "demo", "messages": [{"role": "user", "content": "hello"}],
        "stream": True, "stream_delta_chunk_size": 2,
    })

    assert response.status_code == 200
    assert '"content": "ab"' in response.text
    assert '"content": "cd"' in response.text
    assert '"finish_reason": "stop"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_openwebui_rejects_conflicting_token_limits(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    client = TestClient(serverApi(ModelsMgr(str(config_path)), CaptureHandler()).app)
    response = client.post("/v1/chat/completions", json={
        "model": "demo", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1,
        "max_completion_tokens": 2,
    })
    assert response.status_code == 400
    assert response.json()["detail"] == "max_tokens 与 max_completion_tokens 冲突"


def test_chat_requires_messages(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    client = TestClient(serverApi(ModelsMgr(str(config_path)), CaptureHandler()).app)

    response = client.post("/v1/chat/completions", json={"model": "demo"})
    assert response.status_code == 400
    assert response.json()["detail"] == "messages 不能为空"


def test_management_command_uses_chat_completions(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    handler = CaptureHandler('{"status":"ready"}')
    client = TestClient(serverApi(ModelsMgr(str(config_path)), handler).app)
    for message_id in range(1002, 1008):
        response = client.post("/v1/chat/completions", json={
            "message_id": message_id, "model": "demo",
            "args": {"message_id": message_id}, "stream": False,
        })
        assert response.status_code == 200
        call_args, call_kwargs = handler.calls[-1]
        assert call_args == (message_id, {"message_id": message_id})
        assert call_kwargs["model"] == "demo"
        assert response.json()["choices"][0]["message"]["content"] == '{"status":"ready"}'

    status = TestClient(serverApi(ModelsMgr(str(config_path))).app).post(
        "/v1/chat/completions", json={"message_id": 1001},
    )
    assert status.status_code == 200
    assert status.json()["choices"][0]["message"]["content"]


def test_management_command_supports_stream(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    client = TestClient(serverApi(ModelsMgr(str(config_path)), CaptureHandler("ready")).app)
    response = client.post("/v1/chat/completions", json={"message_id": 1002, "stream": True})

    assert response.status_code == 200
    assert '"content": "ready"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_unknown_message_id_and_removed_routes(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    client = TestClient(serverApi(ModelsMgr(str(config_path))).app)

    assert client.post("/v1/chat/completions", json={"message_id": 9999}).status_code == 400
    assert client.post("/v1/chat", json={}).status_code == 404
    assert client.post("/api/postMessage", json={}).status_code == 404


def test_tool_call_json_and_sse(tmp_path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    calls = [{"id": "call_weather", "type": "function", "function": {
        "name": "get_weather", "arguments": '{"city":"上海"}',
    }}]
    handler = CaptureHandler("", tool_calls=calls, finish_reason="tool_calls")
    client = TestClient(serverApi(ModelsMgr(str(config_path)), handler).app)
    request = {"model": "demo", "messages": [{"role": "user", "content": "天气？"}],
               "tools": TOOLS, "tool_choice": "required"}

    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": None, "tool_calls": calls}
    assert choice["finish_reason"] == "tool_calls"
    assert handler.calls[0][1]["messages"] == request["messages"]
    assert handler.calls[0][1]["tools"] == TOOLS

    streamed = client.post("/v1/chat/completions", json=request | {
        "stream": True, "stream_delta_chunk_size": 5,
    })
    assert streamed.status_code == 200
    assert '"tool_calls"' in streamed.text
    assert '"finish_reason": "tool_calls"' in streamed.text
    argument_parts = []
    for line in streamed.text.splitlines():
        if not line.startswith("data: {"):
            continue
        event = json.loads(line[6:])
        delta = event.get("choices", [{}])[0].get("delta", {}) if event.get("choices") else {}
        for item in delta.get("tool_calls", []):
            argument_parts.append(item.get("function", {}).get("arguments", ""))
    assert "".join(argument_parts) == calls[0]["function"]["arguments"]
    assert streamed.text.endswith("data: [DONE]\n\n")


def test_tool_history_and_invalid_requests(tmp_path) -> None:
    messages = [
        {"role": "user", "content": "天气？"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"上海"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature":25}'},
    ]
    _, choice = validate_tool_request(messages, TOOLS, None)
    assert choice == "auto"

    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({"models": {"demo": {"path": "demo.gguf"}}}), encoding="utf-8")
    client = TestClient(serverApi(ModelsMgr(str(config_path))).app)
    base = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    legacy = client.post("/v1/chat/completions", json=base | {"function_call": "auto"})
    assert legacy.status_code == 400
    strict = client.post("/v1/chat/completions", json=base | {"tools": [{
        "type": "function", "function": {"name": "x", "strict": True,
                                            "parameters": {"type": "object"}},
    }]})
    assert strict.status_code == 400

    bad_output = TestClient(serverApi(
        ModelsMgr(str(config_path)),
        CaptureHandler("bad", status="error", error_type="tool_output"),
    ).app).post("/v1/chat/completions", json=base)
    assert bad_output.status_code == 502

    incomplete = messages[:-1]
    try:
        validate_tool_request(incomplete, TOOLS, None)
    except ValueError as exc:
        assert "未补齐" in str(exc)
    else:
        raise AssertionError("未补齐的工具调用应被拒绝")
