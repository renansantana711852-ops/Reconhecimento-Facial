"""Armazenamento simples em arquivos (sem banco de dados).

dados/pessoas.json   -> lista de pessoas: id, nome, tipo, ativo, criado_em e os embeddings
dados/fotos/<id>.jpg -> miniatura do rosto (só para exibir na tela)

Contrato para a portaria: ler `todas_amostras()` (lista de (pessoa_id, vetor)) e
`obter_pessoa(id)` (campo `ativo` = acesso liberado ou bloqueado).
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

PASTA = Path(__file__).parent / "dados"
ARQ_PESSOAS = PASTA / "pessoas.json"
PASTA_FOTOS = PASTA / "fotos"


class Armazenamento:
    def __init__(self):
        PASTA_FOTOS.mkdir(parents=True, exist_ok=True)
        self._trava = threading.Lock()
        self._pessoas = json.loads(ARQ_PESSOAS.read_text("utf-8")) if ARQ_PESSOAS.exists() else []

    def _salvar(self):
        tmp = ARQ_PESSOAS.with_suffix(".tmp")  # grava num temporário e troca: não corrompe
        tmp.write_text(json.dumps(self._pessoas, ensure_ascii=False), "utf-8")
        os.replace(tmp, ARQ_PESSOAS)

    def _publico(self, p):
        return {k: p[k] for k in ("id", "nome", "tipo", "ativo", "criado_em")} | {
            "amostras": len(p["embeddings"])}

    def criar_pessoa(self, nome, tipo, foto_jpeg, embeddings):
        with self._trava:
            pid = max((p["id"] for p in self._pessoas), default=0) + 1
            self._pessoas.append({
                "id": pid, "nome": nome, "tipo": tipo, "ativo": True,
                "criado_em": datetime.now().isoformat(timespec="seconds"),
                "embeddings": [e.astype(float).tolist() for e in embeddings]})
            if foto_jpeg:
                (PASTA_FOTOS / f"{pid}.jpg").write_bytes(foto_jpeg)
            self._salvar()
        return pid

    def listar_pessoas(self):
        with self._trava:
            return sorted((self._publico(p) for p in self._pessoas), key=lambda p: p["nome"].lower())

    def obter_pessoa(self, pid):
        with self._trava:
            return next((self._publico(p) for p in self._pessoas if p["id"] == pid), None)

    def foto(self, pid):
        arq = PASTA_FOTOS / f"{pid}.jpg"
        return arq.read_bytes() if arq.exists() else None

    def definir_ativo(self, pid, ativo):
        with self._trava:
            for p in self._pessoas:
                if p["id"] == pid:
                    p["ativo"] = bool(ativo)
            self._salvar()

    def excluir_pessoa(self, pid):
        with self._trava:
            self._pessoas = [p for p in self._pessoas if p["id"] != pid]
            (PASTA_FOTOS / f"{pid}.jpg").unlink(missing_ok=True)
            self._salvar()

    def todas_amostras(self):
        with self._trava:
            return [(p["id"], np.array(e, dtype=np.float32))
                    for p in self._pessoas for e in p["embeddings"]]
