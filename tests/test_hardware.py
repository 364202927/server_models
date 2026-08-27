from ai.hardware import detect_hardware


def test_hardware_shape() -> None:
    info = detect_hardware()
    assert isinstance(info.gpus, list)
    assert info.cpu_cores >= 1
