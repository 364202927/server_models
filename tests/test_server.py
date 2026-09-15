import json

from fastapi.testclient import TestClient

from ai.loader.models_mgr import ModelsMgr
from ai.utils.serverApi import ChatRequest, OpenAIChatRequest, serverApi


class CaptureHandler:
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.calls = []

    async def handle(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"status": "ok", "response": self.response,
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


def test_request_defaults() -> None:
    request = ChatRequest(prompt="hello")
    assert request.special == 0 and request.think == 0 and request.stream is False


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
        "model": "demo", "messages": [], "max_tokens": 1,
        "max_completion_tokens": 2,
    })
    assert response.status_code == 400
