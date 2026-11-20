// Utilidades de câmera compartilhadas pelas duas telas.
const Camera = {
  stream: null,

  async listar() {
    const devs = await navigator.mediaDevices.enumerateDevices();
    return devs.filter(d => d.kind === 'videoinput');
  },

  // Pede 1280x720 (o padrão do navegador costuma ser 640x480, o que prejudica
  // MUITO o reconhecimento à distância).
  async abrir(video, deviceId) {
    this.parar();
    const v = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
    if (deviceId) v.deviceId = { exact: deviceId }; else v.facingMode = 'user';
    this.stream = await navigator.mediaDevices.getUserMedia({ video: v, audio: false });
    video.srcObject = this.stream;
    await video.play();
    const palco = video.closest('.palco');
    if (palco && video.videoWidth) palco.style.aspectRatio = video.videoWidth + '/' + video.videoHeight;
    return { w: video.videoWidth, h: video.videoHeight };
  },

  parar() {
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.stream = null;
  },

  // Captura o frame atual como JPEG (resolução cheia, no máximo 1280 de largura).
  capturar(video, canvas) {
    const esc = Math.min(1, 1280 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * esc);
    canvas.height = Math.round(video.videoHeight * esc);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.9));
  },

  // Desenha a caixa do rosto (coordenadas 0-1) sobre o canvas de sobreposição.
  caixa(canvas, video, cx, cor) {
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const g = canvas.getContext('2d');
    g.clearRect(0, 0, canvas.width, canvas.height);
    if (!cx) return;
    g.strokeStyle = cor; g.lineWidth = Math.max(3, canvas.width / 300);
    g.strokeRect(cx[0] * canvas.width, cx[1] * canvas.height, cx[2] * canvas.width, cx[3] * canvas.height);
  },
};

const esperar = ms => new Promise(r => setTimeout(r, ms));
