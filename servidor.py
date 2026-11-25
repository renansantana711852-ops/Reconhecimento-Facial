"""Servidor web do controle de acesso por reconhecimento facial.

Tela de cadastro da facial (captura guiada, várias poses). A portaria vem em outra parte.

Rodar:  python servidor.py   e abrir  http://localhost:5000
(`localhost` é tratado pelo navegador como origem segura, então a câmera funciona.)
"""
import hmac
import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, Response

import motor_facial as mf
from armazenamento import Armazenamento

app = Flask(__name__)

# Proteção para o link público: se SENHA_ACESSO estiver definida, o navegador pede
# usuário (qualquer um) + essa senha antes de mostrar qualquer página ou API.
SENHA_ACESSO = os.environ.get("SENHA_ACESSO", "")


@app.before_request
def exigir_senha():
    if not SENHA_ACESSO or request.path == "/saude":
        return None
    auth = request.authorization
    if auth and hmac.compare_digest(auth.password or "", SENHA_ACESSO):
        return None
    return Response("Acesso restrito", 401, {"WWW-Authenticate": 'Basic realm="Cadastro facial"'})


@app.get("/saude")
def saude():
    return "ok"

motor = mf.MotorFacial()
banco = Armazenamento()
base = mf.BaseDeRostos()


def recarregar_base():
    base.carregar(banco.todas_amostras())


recarregar_base()

# ------------------------------------------------------------------------------
# Parâmetros do fluxo
# ------------------------------------------------------------------------------

# Cadastro: poses necessárias e quantas amostras de cada
POSES = [("frente", 3), ("esquerda", 2), ("direita", 2)]
POSE_FRENTE_MAX = 0.12          # |pose| abaixo disso = olhando de frente
POSE_LADO_MIN, POSE_LADO_MAX = 0.20, 0.55
INTERVALO_AMOSTRAS = 0.5        # segundos entre capturas
MENSAGENS_POSE = {
    "frente": "Olhe de frente para a câmera",
    "esquerda": "Vire um pouco o rosto para um dos lados",   # as duas poses laterais são
    "direita": "Agora vire um pouco para o lado oposto",     # aceitas em qualquer ordem
}


def decodificar():
    dados = np.frombuffer(request.get_data(), dtype=np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("imagem inválida")
    return img


def msg_problema(problemas):
    if "pequeno" in problemas:
        return "Aproxime-se da câmera"
    if "escuro" in problemas:
        return "Pouca luz: vire-se para uma fonte de luz"
    if "claro" in problemas:
        return "Luz forte demais no rosto"
    if "desfocado" in problemas:
        return "Imagem tremida/desfocada: fique parado"
    return ""


def pose_do_rosto(pose):
    if abs(pose) <= POSE_FRENTE_MAX:
        return "frente"
    if POSE_LADO_MIN <= pose <= POSE_LADO_MAX:
        return "esquerda"
    if -POSE_LADO_MAX <= pose <= -POSE_LADO_MIN:
        return "direita"
    return None  # meio termo / virou demais


# ------------------------------------------------------------------------------
# Páginas
# ------------------------------------------------------------------------------
@app.get("/")
def inicio():
    return render_template("home.html")


@app.get("/cadastro")
def pagina_cadastro():
    return render_template("cadastro.html")


@app.get("/reconhecimento")
def pagina_reconhecimento():
    return render_template("reconhecimento.html")


# ------------------------------------------------------------------------------
# API de cadastro
# ------------------------------------------------------------------------------
sessoes = {}  # id -> dict(amostras=[(pose, emb)], ultima=float, foto=bytes|None)


def _progresso(sessao):
    cont = {p: 0 for p, _ in POSES}
    for pose, _ in sessao["amostras"]:
        cont[pose] += 1
    return [{"pose": p, "tem": cont[p], "precisa": n} for p, n in POSES]


def _faltam(sessao):
    return [i["pose"] for i in _progresso(sessao) if i["tem"] < i["precisa"]]


def _proxima_pose(sessao):
    """Próxima pose a pedir. Os dois lados podem ser feitos em qualquer ordem."""
    faltam = _faltam(sessao)
    if not faltam:
        return None
    if "frente" in faltam:
        return "frente"
    return "esquerda" if len(faltam) == 2 else "direita"  # 2 lados faltando / só 1


@app.post("/api/cadastro/iniciar")
def cadastro_iniciar():
    sid = uuid.uuid4().hex
    sessoes[sid] = {"amostras": [], "ultima": 0.0, "foto": None}
    return jsonify(sessao=sid, progresso=_progresso(sessoes[sid]), proxima="frente",
                   mensagem=MENSAGENS_POSE["frente"])


@app.post("/api/cadastro/frame")
def cadastro_frame():
    sid = request.args.get("sessao", "")
    sessao = sessoes.get(sid)
    if sessao is None:
        return jsonify(erro="sessão inexistente"), 404
    img = decodificar()
    info = motor.analisar(img)
    resp = {"rosto": False, "capturou": False}

    proxima = _proxima_pose(sessao)
    resp["progresso"] = _progresso(sessao)
    resp["proxima"] = proxima
    resp["completo"] = proxima is None

    if info["rosto"] is None:
        resp["mensagem"] = "Nenhum rosto detectado"
        return jsonify(resp)
    resp.update(rosto=True, caixa=info["caixa"], largura=info["largura"])

    if info["n_rostos"] > 1:
        resp["mensagem"] = "Mais de um rosto na imagem: só a pessoa sendo cadastrada"
        resp["alerta"] = True
        return jsonify(resp)
    if proxima is None:
        resp["mensagem"] = "Captura completa! Preencha os dados e salve."
        resp["ok"] = True
        return jsonify(resp)

    problemas = motor.problemas_reconhecimento(info, mf.LARGURA_MIN_CADASTRO)
    if problemas:
        resp["mensagem"] = msg_problema(problemas)
        resp["alerta"] = True
        return jsonify(resp)

    pose = pose_do_rosto(info["pose"])
    if pose not in _faltam(sessao):
        resp["mensagem"] = MENSAGENS_POSE[proxima]
        return jsonify(resp)

    agora = time.time()
    if agora - sessao["ultima"] < INTERVALO_AMOSTRAS:
        resp["mensagem"] = "Boa! Segure assim..."
        resp["ok"] = True
        return jsonify(resp)

    emb = motor.embedding(img, info["rosto"])
    sessao["amostras"].append((pose, emb))
    sessao["ultima"] = agora
    if sessao["foto"] is None and pose == "frente":
        x, y, w, h = [int(v) for v in info["rosto"][:4]]
        m = int(w * 0.25)
        recorte = img[max(y - m, 0):y + h + m, max(x - m, 0):x + w + m]
        recorte = cv2.resize(recorte, (200, int(200 * recorte.shape[0] / recorte.shape[1])))
        sessao["foto"] = cv2.imencode(".jpg", recorte)[1].tobytes()

    proxima = _proxima_pose(sessao)
    resp.update(
        capturou=True,
        progresso=_progresso(sessao),
        proxima=proxima,
        completo=proxima is None,
        ok=True,
        mensagem=("Captura completa! Preencha os dados e salve." if proxima is None
                  else MENSAGENS_POSE[proxima]),
    )
    return jsonify(resp)


@app.post("/api/cadastro/finalizar")
def cadastro_finalizar():
    dados = request.get_json(force=True)
    sessao = sessoes.get(dados.get("sessao", ""))
    nome = (dados.get("nome") or "").strip()
    tipo = (dados.get("tipo") or "Morador").strip()
    if sessao is None or _proxima_pose(sessao) is not None:
        return jsonify(erro="Captura incompleta."), 400
    if len(nome) < 3:
        return jsonify(erro="Informe o nome completo."), 400

    # Evita cadastrar a mesma pessoa duas vezes
    for _, emb in sessao["amostras"]:
        pid, score, segundo = base.comparar(emb)
        if pid is not None and base.aceito(score, segundo):
            existente = banco.obter_pessoa(pid)
            if existente:
                return jsonify(erro=f"Esse rosto já está cadastrado como “{existente['nome']}”."), 409

    pid = banco.criar_pessoa(nome, tipo, sessao["foto"], [e for _, e in sessao["amostras"]])
    recarregar_base()
    sessoes.pop(dados["sessao"], None)
    return jsonify(id=pid)


# ------------------------------------------------------------------------------
# API de pessoas
# ------------------------------------------------------------------------------
@app.get("/api/pessoas")
def api_pessoas():
    return jsonify(banco.listar_pessoas())


@app.get("/api/pessoas/<int:pid>/foto")
def api_foto(pid):
    foto = banco.foto(pid)
    if not foto:
        return "", 404
    return Response(foto, mimetype="image/jpeg")


@app.post("/api/pessoas/<int:pid>/ativo")
def api_ativo(pid):
    banco.definir_ativo(pid, bool(request.get_json(force=True).get("ativo")))
    return jsonify(ok=True)


@app.delete("/api/pessoas/<int:pid>")
def api_excluir(pid):
    banco.excluir_pessoa(pid)
    recarregar_base()
    return jsonify(ok=True)


# ------------------------------------------------------------------------------
# API de reconhecimento (portaria)
# ------------------------------------------------------------------------------
# Cada chamada analisa UM frame. A regra "3 frames seguidos com a mesma pessoa" fica
# na tela (templates/reconhecimento.html), como recomendado no README.
@app.post("/api/reconhecimento/frame")
def reconhecimento_frame():
    img = decodificar()
    info = motor.analisar(img)
    resp = {"rosto": False, "estado": "sem_rosto", "mensagem": "Aproxime-se da câmera"}
    if info["rosto"] is None:
        return jsonify(resp)

    resp.update(rosto=True, caixa=info["caixa"], largura=info["largura"])
    if info["n_rostos"] > 1:
        resp.update(estado="varios", mensagem="Mais de uma pessoa na imagem: uma por vez")
        return jsonify(resp)

    problemas = motor.problemas_reconhecimento(info)
    if problemas:
        resp.update(estado="problema", mensagem=msg_problema(problemas))
        return jsonify(resp)

    vetor = motor.embedding(img, info["rosto"])
    pid, score, segundo = base.comparar(vetor)
    resp["score"] = round(float(score), 3)
    if pid is not None and base.aceito(score, segundo):
        pessoa = banco.obter_pessoa(pid)
        if pessoa:
            resp.update(
                estado="reconhecido",
                mensagem="Rosto reconhecido",
                pessoa={k: pessoa[k] for k in ("id", "nome", "tipo", "ativo")},
            )
            return jsonify(resp)
    resp.update(estado="desconhecido", mensagem="Rosto não reconhecido")
    return jsonify(resp)


# A portaria (tela que simula o aparelho) usa `motor`, `base` e `banco` daqui: ver
# `MotorFacial.analisar/embedding` e `BaseDeRostos.comparar/aceito`.


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
            threaded=True)
