let sessionId = null;
let naturalWidth = 0;
let naturalHeight = 0;

const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const inpaintBtn = document.getElementById('inpaintBtn');
const exportBtn = document.getElementById('exportBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const layersEl = document.getElementById('layers');

const baseImage = new Image();

async function upload() {
  const file = fileInput.files?.[0];
  if (!file) return alert('PNGを選択してください');

  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/session', { method: 'POST', body: formData });
  if (!res.ok) return alert(await res.text());
  const data = await res.json();
  sessionId = data.session_id;

  const src = URL.createObjectURL(file);
  baseImage.src = src;
  await baseImage.decode();

  naturalWidth = baseImage.naturalWidth;
  naturalHeight = baseImage.naturalHeight;
  canvas.width = naturalWidth;
  canvas.height = naturalHeight;
  ctx.drawImage(baseImage, 0, 0);
  layersEl.innerHTML = '';
}

canvas.addEventListener('click', async (e) => {
  if (!sessionId) return;

  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width));
  const y = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height));

  const layerName = prompt('レイヤー名 (例: hair_front):', `part_${layersEl.children.length + 1}`) || undefined;

  const res = await fetch(`/api/session/${sessionId}/mask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x, y, positive: true, layer_name: layerName }),
  });

  if (!res.ok) return alert(await res.text());
  const data = await res.json();

  const img = new Image();
  img.src = data.preview;
  img.title = layerName || `part_${data.layer_index + 1}`;
  layersEl.appendChild(img);
});

inpaintBtn.addEventListener('click', async () => {
  if (!sessionId) return;

  const res = await fetch(`/api/session/${sessionId}/inpaint`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expand_px: 18, radius: 5 }),
  });

  if (!res.ok) return alert(await res.text());
  const data = await res.json();

  const img = new Image();
  img.src = data.preview;
  await img.decode();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, naturalWidth, naturalHeight);
});

exportBtn.addEventListener('click', async () => {
  if (!sessionId) return;
  const res = await fetch(`/api/session/${sessionId}/export`, { method: 'POST' });
  if (!res.ok) return alert(await res.text());

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'live2d.psd';
  a.click();
  URL.revokeObjectURL(url);
});

uploadBtn.addEventListener('click', upload);
