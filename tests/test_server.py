import json

from fastapi.testclient import TestClient

from ai.loader.models_mgr import ModelsMgr
from ai.utils.serverApi import ChatRequest, serverApi


def test_request_defaults() -> None:
    request = ChatRequest(prompt="hello")
    assert request.special == 0 and request.think == 0 and request.stream is False


def test_server_interface_switch(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({
        "defaults": {"server": {"interface": "api", "api_port": 8000, "desktop_port": 3000}},
        "models": {"demo": {"path": str(tmp_path / "demo.gguf")}},
    }), encoding="utf-8")
    manager = ModelsMgr(str(config_path))

    monkeypatch.setenv("AI_SERVER_INTERFACE", "desktop")
    desktop = serverApi(manager)
    client = TestClient(desktop.app)
    assert desktop._port() == 3000
    assert "AI Desktop Web UI" in client.get("/").text
    assert client.get("/v1/models").json()["data"][0]["id"] == "demo"

    monkeypatch.setenv("AI_SERVER_INTERFACE", "api")
    api = serverApi(manager)
    assert api._port() == 8000
    assert api.app is not desktop.app


def test_openai_bearer_auth(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps({
        "defaults": {"server": {"interface": "api", "api_key": "secret"}},
        "models": {"demo": {"path": str(tmp_path / "demo.gguf")}},
    }), encoding="utf-8")
    manager = ModelsMgr(str(config_path))
    monkeypatch.delenv("AI_SERVER_INTERFACE", raising=False)
    client = TestClient(serverApi(manager).app)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
