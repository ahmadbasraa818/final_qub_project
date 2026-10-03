"""The command line and the browser interface."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from blurfft.cli import main
from blurfft.dataset import convolve, gaussian_kernel, load_source
from blurfft.webapp import make_server


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    folder = tmp_path_factory.mktemp("images")
    photo = load_source("camera")[50:350, 40:440]
    Image.fromarray((photo * 255).astype(np.uint8)).save(folder / "sharp.png")
    Image.fromarray((convolve(photo, gaussian_kernel(3.0)) * 255).astype(np.uint8)).save(folder / "blurred.jpg", quality=92)
    return folder


def test_analyse_prints_a_verdict_per_image(images, capsys):
    assert main(["analyse", str(images)]) == 0
    out = capsys.readouterr().out
    assert "sharp.png: sharp" in out
    assert "blurred.jpg: BLURRED" in out


def test_analyse_json(images, capsys):
    assert main(["analyse", str(images / "blurred.jpg"), "--json", "--precision", "float16"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert reports[0]["blurry"] is True and reports[0]["precision"] in ("float16", "e5m10")


def test_compare_and_formats(images, capsys):
    assert main(["compare", str(images / "sharp.png"), "--precisions", "float64,bfloat16,e8m4"]) == 0
    out = capsys.readouterr().out
    assert "float64" in out and "e8m7" in out and "e8m4" in out and "(emulated)" in out
    assert main(["formats"]) == 0
    assert "e8m12" in capsys.readouterr().out


def test_map_writes_an_image(images, tmp_path, capsys):
    out = tmp_path / "map.png"
    assert main(["map", str(images / "blurred.jpg"), "--out", str(out)]) == 0
    assert out.exists() and Image.open(out).size[0] > 0


def test_bad_input_is_reported_not_raised(capsys):
    assert main(["analyse", "no/such/file.png"]) == 2
    assert main(["analyse", ".", "--precision", "float128"]) == 2


def test_web_interface(images):
    server = make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Drop an image here" in page
        formats = json.loads(urllib.request.urlopen(base + "/api/formats").read())
        assert any(f["label"] == "bfloat16" for f in formats["formats"])
        body = (images / "blurred.jpg").read_bytes()
        request = urllib.request.Request(base + "/api/analyse?precision=float16", data=body, method="POST")
        result = json.loads(urllib.request.urlopen(request).read())
        assert result["report"]["blurry"] is True
        assert set(result["images"]) == {"input", "map", "spectrum"}
        request = urllib.request.Request(base + "/api/compare", data=body, method="POST")
        rows = json.loads(urllib.request.urlopen(request).read())["rows"]
        assert len(rows) >= 6 and rows[0]["blurry"] is True
        bad = urllib.request.Request(base + "/api/analyse", data=b"not an image", method="POST")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
