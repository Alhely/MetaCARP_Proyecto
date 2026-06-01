"""
Approach pr_aislado_20260531 para tabu_reactiva.

Aisla el efecto de Path Relinking LIMPIO (módulo path_relinking_limpio_20260531,
sin frame-hacks ni kick aleatorio) sobre el selector elegible (p_inter/binario),
bajo el evaluador greedy nativo. Grid en dos fases. Lógica en
_pr_aislado_20260531_common.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pr_aislado_20260531_common import main

if __name__ == "__main__":
    main(mh_fijo="tabu_reactiva")
