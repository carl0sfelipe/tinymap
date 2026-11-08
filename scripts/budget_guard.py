#!/usr/bin/env python3
"""Teto de gasto da GPU alugada (ideia do Isometric SF: um guarda para o agente rodar a noite toda).

    python scripts/budget_guard.py --max-usd 15            # preço por hora lido da vast.ai
    python scripts/budget_guard.py --max-usd 15 --usd-hour 0.35

Rode em tmux. A cada minuto soma o gasto desde o início. Em 80% escreve AVISO no log; no teto, mata os
jobs do tinymap e do ai-toolkit (o trabalho feito fica salvo: checkpoint por passo e .done por par)
e para a instância com `vastai stop instance $CONTAINER_ID` (parar mantém o disco; destruir não).
Sem a CLI da vast.ai, só mata os jobs e avisa. Log em runs/budget.log.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "runs" / "budget.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def vast_price(cid: str | None) -> float | None:
    if not (cid and shutil.which("vastai")):
        return None
    try:
        out = subprocess.run(["vastai", "show", "instance", cid, "--raw"], capture_output=True, text=True,
                             timeout=60).stdout
        return float(json.loads(out).get("dph_total") or 0) or None
    except (subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return None


def _jobs() -> list[int]:
    """PIDs dos jobs: o comando `tinymap ...`, `python -m tinymap.cli` e o treino do ai-toolkit.
    Casa por token do comando (não por trecho do caminho: `pkill -f tinymap` mataria qualquer coisa
    rodando dentro da pasta tinymap, inclusive este guarda)."""
    me, out = {os.getpid(), os.getppid()}, []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in me:
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        toks = [t.decode(errors="ignore") for t in argv if t]
        names = [Path(t).name for t in toks]
        if any(n == "tinymap" for n in names[:2]) or "tinymap.cli" in toks \
                or ("run.py" in names and any("ai-toolkit" in t for t in toks)):
            out.append(int(d.name))
    return out


def stop_jobs() -> None:
    import signal
    for sig, wait in ((signal.SIGINT, 30), (signal.SIGKILL, 0)):
        for pid in _jobs():
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--usd-hour", type=float, help="preço por hora; sem isto, lido da vast.ai")
    ap.add_argument("--spent", type=float, default=0.0, help="gasto já feito antes deste guarda")
    ap.add_argument("--every", type=int, default=60)
    a = ap.parse_args()
    cid = os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL", "").removeprefix("C.") or None
    price = a.usd_hour or vast_price(cid)
    if not price:
        raise SystemExit("sem preço por hora: passe --usd-hour (ou instale a CLI vastai com a API key)")
    t0, warned = time.time(), False
    log(f"guarda ligado: US$ {price:.3f}/h, teto US$ {a.max_usd:.2f}, já gasto US$ {a.spent:.2f}, instância {cid}")
    while True:
        spent = a.spent + (time.time() - t0) / 3600 * price
        if spent >= a.max_usd:
            log(f"TETO: US$ {spent:.2f}. Parando os jobs.")
            stop_jobs()
            if cid and shutil.which("vastai"):
                log("parando a instância (disco preservado)")
                subprocess.run(["vastai", "stop", "instance", cid], check=False)
            else:
                log("sem CLI vastai: pare a instância no painel")
            return
        if spent >= 0.8 * a.max_usd and not warned:
            log(f"AVISO: 80% do teto (US$ {spent:.2f} de {a.max_usd:.2f})")
            warned = True
        time.sleep(a.every)


if __name__ == "__main__":
    main()
