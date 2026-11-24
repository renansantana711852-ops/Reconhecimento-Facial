# Controle de acesso por reconhecimento facial (simulação)

Este trecho cobre o **motor de reconhecimento (A)** e a **tela de cadastro (B)**.
A tela da portaria (simulação do aparelho) fica com outra parte do grupo.

## Rodando

```bash
pip install -r requirements.txt
python servidor.py
```

Abra `http://localhost:5000/cadastro` (tem que ser `localhost` ou HTTPS para o navegador liberar a câmera).
Os modelos (`modelos/`) já vêm no repositório; se faltarem: `python baixar_modelos.py`.

## Arquivos

| Arquivo | Função |
|---|---|
| `motor_facial.py` | detecta rosto (YuNet), gera o vetor do rosto (SFace) e compara com os cadastrados |
| `armazenamento.py` | guarda as pessoas em `dados/pessoas.json` + miniaturas em `dados/fotos/` (sem banco de dados) |
| `servidor.py` | Flask: tela de cadastro e API |
| `templates/`, `static/` | tela de cadastro e utilitários de câmera (reaproveitáveis na portaria) |

## Para a portaria usar

```python
import motor_facial as mf
from armazenamento import Armazenamento

motor, banco, base = mf.MotorFacial(), Armazenamento(), mf.BaseDeRostos()
base.carregar(banco.todas_amostras())              # recarregar quando alguém for cadastrado/excluído

info = motor.analisar(frame_bgr)                   # info["rosto"] é None se não achou rosto
problemas = motor.problemas_reconhecimento(info)   # [] ou "pequeno", "desfocado", "escuro", "claro"
vetor = motor.embedding(frame_bgr, info["rosto"])
pessoa_id, score, segundo = base.comparar(vetor)
if base.aceito(score, segundo):
    pessoa = banco.obter_pessoa(pessoa_id)         # pessoa["ativo"] False = cadastro bloqueado
```

Recomendação: só liberar depois de **3 frames seguidos** com a mesma pessoa, não em um único frame.

## Ajustes (variáveis de ambiente)

| Variável | Padrão | Para quê |
|---|---|---|
| `LIMIAR` | 0.40 | similaridade mínima para reconhecer (maior = mais rígido) |
| `MARGEM` | 0.05 | diferença mínima para o 2º colocado |
| `LARGURA_MIN_RECONHECER` | 60 | largura mínima do rosto (px) para tentar reconhecer |
| `NITIDEZ_MIN` / `BRILHO_MIN` | 25 / 55 | rejeita imagem tremida / escura |

Os scripts antigos (`cadastrar_rosto.py`, `reconhecimento_facial.py`, ...) foram mantidos como protótipo.

## Publicando online (link público)

A câmera do navegador só funciona em HTTPS, e hospedagens como o Render já dão HTTPS automático.

1. Suba o código para o GitHub (branch `master`).
2. No [Render](https://render.com): **New → Blueprint** (ou *Web Service*), escolha este repositório. O `render.yaml` já configura tudo.
   Se criar manualmente: build `pip install -r requirements-deploy.txt`, start
   `gunicorn servidor:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.
3. Em **Environment**, defina `SENHA_ACESSO` com uma senha. O navegador vai pedir usuário (qualquer um) e essa senha.
4. Abra o link `https://<nome>.onrender.com/cadastro`.

Pontos de atenção:
- **Use sempre 1 worker** (sessões de cadastro ficam em memória).
- **Os cadastros somem** quando o serviço grátis reinicia/dorme (disco efêmero). Para manter, use um disco
  persistente e aponte `DADOS_DIR` para ele. Para a apresentação, cadastre as pessoas pouco antes.
- O plano grátis dorme após ~15 min sem uso (o primeiro acesso demora ~1 min) e tem pouca CPU: o
  reconhecimento fica mais lento que no notebook. Abra o link antes de apresentar.
- São **dados biométricos**: cadastre só quem concordou (o grupo), mantenha a senha ligada e exclua os
  cadastros depois da apresentação.
