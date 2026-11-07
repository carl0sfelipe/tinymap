"""Testa latent_pin e held_windows (models.py) sem torch: extrai as funções do código-fonte."""
import ast
from pathlib import Path

import numpy as np

src = (Path(__file__).resolve().parent.parent / "tinymap" / "models.py").read_text()
mod = ast.parse(src)
ns = {"np": np}
for node in mod.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("latent_pin", "held_windows"):
        exec(compile(ast.Module([node], []), "models.py", "exec"), ns)
latent_pin, held_windows = ns["latent_pin"], ns["held_windows"]


def test():
    f = 16
    pin = np.zeros((2048, 2048), bool)
    pin[:, :1024] = True                      # metade esquerda já pintada
    sem_faixa = latent_pin(pin, f, 0)
    assert sem_faixa[:, :64].all() and not sem_faixa[:, 64:].any()
    com_faixa = latent_pin(pin, f, 64)        # 64 px = 4 células livres junto da borda
    assert com_faixa[:, :60].all() and not com_faixa[:, 60:].any()
    assert com_faixa[0, 0] and com_faixa[-1, 0], "borda da imagem não é divisa: continua fixa"

    pin2 = pin.copy()
    pin2[0, 5] = False                         # um pixel faltando: a célula não fica fixa
    assert not latent_pin(pin2, f, 0)[0, 0]

    lwin = 1024 // f
    wins = [(x, y) for y in (0, 768, 1024) for x in (0, 768, 1024)]
    held = held_windows(sem_faixa, wins, lwin, f)
    assert held == {(0, 0), (0, 768), (0, 1024)}, held
    held64 = held_windows(com_faixa, wins, lwin, f)
    assert held64 == set(), held64  # toda janela da esquerda encosta na faixa livre: trabalha
    print("ok", len(held), len(held64))


if __name__ == "__main__":
    test()
