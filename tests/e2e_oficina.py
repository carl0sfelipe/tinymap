"""Teste de ponta a ponta da oficina num navegador de verdade (Playwright + Chrome), na cidade demo.

    python tests/e2e_oficina.py [--chrome /usr/bin/google-chrome] [--shots pasta]

Passos: abre a oficina, mede a saúde, seleciona uma área, pule com prévia (antes/depois), aplica,
confere o histórico, desfaz, e roda "Consertar e polir o mapa inteiro" até a nota subir.
Falha se a página tiver erro de JavaScript.
"""
import argparse
import json
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_estudio import _cidade, _porta, _quiet  # noqa: E402

from tinymap.mapedit import MapEdit, serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--shots", default=None)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as t:
        run, rf = _quiet(_cidade, Path(t))
        port = _porta()
        threading.Thread(target=serve, args=(MapEdit(run, rf), "127.0.0.1", port), daemon=True).start()
        time.sleep(0.5)
        url = f"http://127.0.0.1:{port}/"
        erros = []
        with sync_playwright() as p:
            b = p.chromium.launch(**({"executable_path": a.chrome} if a.chrome else {}))
            pg = b.new_page(viewport={"width": 1440, "height": 900})
            pg.on("pageerror", lambda e: erros.append(str(e)))
            pg.on("dialog", lambda d: d.accept())
            pg.goto(url)
            pg.wait_for_timeout(1500)
            if pg.is_visible("#welcome-ok"):
                pg.click("#welcome-ok")

            # 1. saúde do mapa
            pg.click("#b-scan2")
            pg.wait_for_function("document.querySelector('#scanst2').textContent.includes('achados')", timeout=120000)
            nota0 = int(pg.inner_text("#health .score b"))
            print("nota inicial", nota0)

            # 2. polir uma área com prévia e aplicar
            pg.evaluate("setSel([1024, 1024, 1536, 1536])")
            assert not pg.is_disabled("#b-polsel"), "botão de polir deveria ligar com seleção"
            pg.click("#b-polsel")
            pg.wait_for_function("document.querySelector('#polinfo').textContent.includes('pixel de')", timeout=60000)
            assert "3 px" in pg.inner_text("#polinfo"), pg.inner_text("#polinfo")
            assert pg.evaluate("!!(edit && edit.result)"), "prévia deveria estar na edição"
            if a.shots:
                pg.screenshot(path=f"{a.shots}/e2e_previa.png")
            pg.click("#b-apply")
            pg.wait_for_function("!edit", timeout=60000)
            hist = json.load(urllib.request.urlopen(url + "api/history"))
            assert hist and hist[0]["box"] == [1024, 1024, 1536, 1536], hist[:1]
            assert hist[0]["note"].startswith("polimento"), hist[0]["note"]
            print("aplicado:", hist[0]["note"])

            # 3. desfazer pelo histórico
            pg.click("#tabs [data-tab='hist']")
            pg.wait_for_selector("#histlist [data-undo]")
            pg.click("#histlist [data-undo]")
            pg.wait_for_timeout(1500)
            hist = json.load(urllib.request.urlopen(url + "api/history"))
            assert hist[0]["undone"], hist[0]

            # 4. consertar e polir o mapa inteiro
            pg.click("#tabs [data-tab='pol']")
            pg.click("#b-polall")
            pg.wait_for_function("document.querySelector('#polall').textContent.includes('Pronto')", timeout=600000)
            pg.wait_for_function("document.querySelector('#scanst2').textContent.includes('achados') && "
                                 "!document.querySelector('#health').textContent.includes('medindo')", timeout=120000)
            pg.wait_for_timeout(1000)
            nota1 = int(pg.inner_text("#health .score b"))
            print("nota final", nota1)
            if a.shots:
                pg.screenshot(path=f"{a.shots}/e2e_final.png")
            b.close()
        assert not erros, erros
        assert nota1 >= nota0 + 30, (nota0, nota1)
        print("ok  e2e da oficina")


if __name__ == "__main__":
    main()
