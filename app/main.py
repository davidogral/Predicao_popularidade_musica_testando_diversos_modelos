from pathlib import Path
from typing import List, Tuple
import tempfile

import joblib
import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


class TrackFeatures(BaseModel):
    danceability: float = Field(..., ge=0, le=1, description="0-1; quão dançante é a faixa")
    energy: float = Field(..., ge=0, le=1, description="0-1; intensidade da música")
    acousticness: float = Field(..., ge=0, le=1, description="0-1; probabilidade de ser acústica")
    speechiness: float = Field(..., ge=0, le=1, description="0-1; presença de fala")
    valence: float = Field(..., ge=0, le=1, description="0-1; humor/positividade")
    tempo: float = Field(..., gt=0, description="BPM da faixa")
    duration_ms: float = Field(..., gt=0, description="Duração em milissegundos")
    instrumentalness: float = Field(..., ge=0, le=1, description="0-1; probabilidade de ser instrumental")
    liveness: float = Field(..., ge=0, le=1, description="0-1; probabilidade de ser ao vivo")
    loudness: float = Field(..., description="Volume médio em dB (tipicamente negativo)")


class PredictionResponse(BaseModel):
    cluster: int
    distances: List[float]
    model_file: str
    features_order: List[str]


def _load_latest_bundle() -> Tuple[List[str], object, object, str]:
    """Carrega o pacote (scaler + modelo) mais recente salvo em nao_supervisionado/models."""
    models_dir = Path(__file__).resolve().parents[1] / "nao_supervisionado" / "models"
    if not models_dir.exists():
        raise RuntimeError("Diretório de modelos não encontrado. Treine o KMeans antes de servir a API.")

    model_files = sorted(models_dir.glob("kmeans_k*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_files:
        raise RuntimeError("Nenhum modelo encontrado em nao_supervisionado/models.")

    bundle_path = model_files[0]
    bundle = joblib.load(bundle_path)
    features = bundle["features"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    return features, scaler, model, bundle_path.name


FEATURES, SCALER, MODEL, MODEL_FILE = _load_latest_bundle()

app = FastAPI(
    title="Clusterização de Faixas - Spotify",
    description="Serviço FastAPI para prever o cluster de uma música usando o KMeans treinado.",
    version="1.0.0",
)


def _describe_cluster_features(feat: dict) -> Tuple[str, str]:
    """Gera um rótulo curto e um resumo textual para um cluster baseado no centróide."""
    energy = feat.get("energy", 0)
    dance = feat.get("danceability", 0)
    acoustic = feat.get("acousticness", 0)
    speech = feat.get("speechiness", 0)
    valence = feat.get("valence", 0)
    tempo = feat.get("tempo", 0)
    duration_ms = feat.get("duration_ms", 0)

    def level(val: float, hi: float, lo: float = 0.33) -> str:
        if val >= hi:
            return "alto"
        if val <= lo:
            return "baixo"
        return "médio"

    energy_lvl = level(energy, 0.67)
    dance_lvl = level(dance, 0.67)
    acoustic_lvl = level(acoustic, 0.6)
    mood = "mais alegre" if valence > 0.6 else "mais sombrio" if valence < 0.4 else "neutro"
    vocal = "voz em destaque" if speech > 0.25 else "instrumental" if feat.get("instrumentalness", 0) > 0.5 else "misto"

    label_parts = []
    if energy_lvl == "alto" and dance_lvl != "baixo":
        label_parts.append("Energético e dançante")
    elif acoustic_lvl == "alto":
        label_parts.append("Acústico/intimista")
    elif speech > 0.35:
        label_parts.append("Falado/rap")
    else:
        label_parts.append(f"Clima {mood}")

    summary = (
        f"{label_parts[0]}; energia {energy_lvl}, dança {dance_lvl}, acústico {acoustic_lvl}, "
        f"{vocal}; BPM ~{tempo:.0f}, duração ~{duration_ms/60000:.1f} min."
    )
    return label_parts[0], summary


def _build_cluster_profiles() -> List[dict]:
    centers = MODEL.cluster_centers_
    raw_centers = SCALER.inverse_transform(centers)
    profiles = []
    for idx, center in enumerate(raw_centers):
        centroid = {feat: float(center[i]) for i, feat in enumerate(FEATURES)}
        if "duration_ms" in centroid:
            centroid["duration_ms"] = float(np.expm1(centroid["duration_ms"]))
        label, summary = _describe_cluster_features(centroid)
        profiles.append(
            {
                "index": idx,
                "label": label,
                "summary": summary,
                "centroid": centroid,
            }
        )
    return profiles


CLUSTER_PROFILES = _build_cluster_profiles()


def _transform_payload(payload: TrackFeatures) -> np.ndarray:
    """Aplica o mesmo pré-processamento usado no treino (log1p em duration_ms + StandardScaler)."""
    values = []
    data = payload.model_dump()
    for feat in FEATURES:
        if feat not in data:
            raise HTTPException(status_code=400, detail=f"Campo ausente: {feat}")
        val = data[feat]
        if feat == "duration_ms":
            if val <= 0:
                raise HTTPException(status_code=400, detail="duration_ms precisa ser positivo para aplicar log1p.")
            val = np.log1p(val)
        values.append(val)

    arr = np.array(values, dtype=float).reshape(1, -1)
    return SCALER.transform(arr)


def _predict_from_features(features: TrackFeatures) -> dict:
    transformed = _transform_payload(features)
    cluster = int(MODEL.predict(transformed)[0])
    distances = MODEL.transform(transformed)[0].round(5).tolist()

    dists_arr = np.array(distances, dtype=float)
    max_dist = float(dists_arr.max()) if dists_arr.size else 1.0
    proximity = []
    if max_dist == 0:
        proximity = [{"cluster": int(i), "score": 1.0} for i in range(len(distances))]
    else:
        scores = 1 - (dists_arr / max_dist)
        order = list(np.argsort(scores))[::-1]
        proximity = [{"cluster": int(i), "score": round(float(scores[i]), 3)} for i in order]

    profile = CLUSTER_PROFILES[cluster] if 0 <= cluster < len(CLUSTER_PROFILES) else None

    return {
        "cluster": cluster,
        "distances": distances,
        "proximity": proximity,
        "cluster_label": profile["label"] if profile else f"Cluster {cluster}",
        "cluster_summary": profile["summary"] if profile else "",
        "cluster_centroid": profile["centroid"] if profile else {},
        "model_file": MODEL_FILE,
        "features_order": FEATURES,
    }


def _extract_features_from_audio(audio_path: Path) -> TrackFeatures:
    """Extrai features aproximadas do áudio para alimentar o modelo de clusters."""
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as exc:  # erros de leitura
        raise HTTPException(status_code=400, detail=f"Não foi possível ler o áudio: {exc}") from exc

    if y.size == 0:
        raise HTTPException(status_code=400, detail="Áudio vazio ou inválido.")

    duration_ms = float(librosa.get_duration(y=y, sr=sr) * 1000)
    tempo_arr = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
    tempo = float(np.median(tempo_arr)) if tempo_arr.size else 60.0

    rms = librosa.feature.rms(y=y).squeeze()
    if rms.size == 0:
        raise HTTPException(status_code=400, detail="Não foi possível calcular RMS do áudio.")
    rms_mean = float(np.mean(rms))
    loudness_db = float(librosa.amplitude_to_db([rms_mean], ref=1.0).mean()) if rms_mean > 0 else -60.0
    energy = float(np.clip((loudness_db + 60) / 60, 0, 1))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_norm = float(np.clip(np.mean(onset_env) / (np.max(onset_env) + 1e-6), 0, 1))
    tempo_norm = float(np.clip((tempo - 60) / 120, 0, 1))
    danceability = float(np.clip(0.6 * tempo_norm + 0.4 * onset_norm, 0, 1))

    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    acousticness = float(np.clip(1 - flatness * 2, 0, 1))

    # Detecção simples de voz: usa hpss para medir conteúdo harmônico vs percussivo + zcr
    y_h, y_p = librosa.effects.hpss(y)
    rms_h = float(np.mean(np.abs(y_h)))
    harmonic_ratio = rms_h / (rms_mean + 1e-6) if rms_mean > 0 else 0.0
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    speechiness_raw = zcr * 3 * (1 - harmonic_ratio * 0.5)
    speechiness = float(np.clip(speechiness_raw, 0, 1))
    instrumental_boost = 0.4 if harmonic_ratio < 0.3 else 0.0
    instrumentalness = float(np.clip(1 - speechiness * 1.2 + instrumental_boost, 0, 1))

    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    centroid_norm = float(np.clip(centroid / 5000, 0, 1))
    valence = float(np.clip(0.5 * tempo_norm + 0.5 * centroid_norm, 0, 1))

    dynamic_range = float(np.clip((np.percentile(rms, 95) - np.percentile(rms, 5)) * 10, 0, 1))
    liveness = dynamic_range

    features = {
        "danceability": danceability,
        "energy": energy,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "valence": valence,
        "tempo": max(tempo, 1.0),
        "duration_ms": max(duration_ms, 1.0),
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "loudness": loudness_db,
    }

    return TrackFeatures(**features)


def _build_track_insights(features: TrackFeatures) -> List[str]:
    data = features.model_dump()
    insights = []
    tempo = data["tempo"]
    bpm_zone = "lento" if tempo < 90 else "moderado" if tempo < 120 else "rápido"
    insights.append(f"BPM ~{tempo:.0f} ({bpm_zone})")

    energy = data["energy"]
    if energy > 0.7:
        insights.append("Energia alta")
    elif energy < 0.4:
        insights.append("Energia baixa/suave")
    else:
        insights.append("Energia moderada")

    dance = data["danceability"]
    if dance > 0.65:
        insights.append("Bastante dançante")
    elif dance < 0.35:
        insights.append("Pouco dançante")

    acoustic = data["acousticness"]
    if acoustic > 0.6:
        insights.append("Caráter acústico/intimista")
    elif acoustic < 0.2:
        insights.append("Som eletrônico/produzido")

    speech = data["speechiness"]
    if speech > 0.35:
        insights.append("Fala/rap em destaque")

    valence = data["valence"]
    if valence > 0.65:
        insights.append("Clima alegre/positivo")
    elif valence < 0.35:
        insights.append("Clima mais sombrio/sério")

    duration_min = data["duration_ms"] / 60000
    insights.append(f"Duração ~{duration_min:.1f} min")
    return insights


def _summarize_track(features: TrackFeatures) -> str:
    data = features.model_dump()
    tempo = data["tempo"]
    energy = data["energy"]
    dance = data["danceability"]
    acoustic = data["acousticness"]
    speech = data["speechiness"]
    valence = data["valence"]
    duration_min = data["duration_ms"] / 60000

    tempo_desc = "lento" if tempo < 90 else "moderado" if tempo < 120 else "rápido"
    mood = "alegre" if valence > 0.65 else "sombrio" if valence < 0.35 else "neutro"
    vibe = "acústico/intimista" if acoustic > 0.6 else "eletrônico/produzido" if acoustic < 0.2 else "misto"
    vocal = "voz em destaque" if speech > 0.35 else "instrumental" if data["instrumentalness"] > 0.6 else "voz discreta"
    energy_txt = "energia alta" if energy > 0.7 else "energia baixa" if energy < 0.4 else "energia moderada"
    dance_txt = "bem dançante" if dance > 0.65 else "pouco dançante" if dance < 0.35 else "dança moderada"

    return (
        f"{vibe}, {vocal}; {energy_txt}, {dance_txt}; BPM ~{tempo:.0f} ({tempo_desc}), "
        f"{mood}; duração ~{duration_min:.1f} min."
    )


@app.get("/health")
def healthcheck():
    """Retorna informações básicas do modelo carregado."""
    return {
        "status": "ok",
        "model_file": MODEL_FILE,
        "n_clusters": int(MODEL.n_clusters),
        "features_order": FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_cluster(features: TrackFeatures):
    """Recebe os atributos da faixa e devolve o cluster previsto e distâncias aos centróides."""
    try:
        return _predict_from_features(features)
    except HTTPException:
        raise
    except Exception as exc:  # captura erros inesperados de shape/validação
        raise HTTPException(status_code=500, detail=f"Erro na predição: {exc}") from exc


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Recebe um arquivo de áudio, extrai features aproximadas e devolve o cluster."""
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        features = _extract_features_from_audio(tmp_path)
        prediction = _predict_from_features(features)
        insights = _build_track_insights(features)
        track_summary = _summarize_track(features)
        return {
            "features": features.model_dump(),
            "track_insights": insights,
            "track_summary": track_summary,
            **prediction,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
def landing_page():
    """Página simples para subir áudio e ver a previsão do cluster."""
    return """
    <!doctype html>
    <html lang=\"pt-BR\">
    <head>
      <meta charset=\"utf-8\">
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
      <title>Cluster de Música</title>
      <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
      <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
      <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap\" rel=\"stylesheet\">
      <style>
        :root { --bg: #0b1221; --panel: #0f1b32; --accent: #7cf6b9; --accent-2: #7fc6f6; --text: #eaf3ff; --muted: rgba(234,243,255,0.7); }
        * { box-sizing: border-box; }
        body { margin:0; min-height:100vh; font-family:'Space Grotesk', system-ui; background:radial-gradient(circle at 20% 20%, #10203d 0, #0b1221 40%), radial-gradient(circle at 80% 0%, #0f2b4a 0, #0b1221 35%), #0b1221; color:var(--text); }
        .shell { max-width: 960px; margin: 0 auto; padding: 32px 18px 48px; display:flex; justify-content:center; }
        .card { width: 100%; background:var(--panel); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:24px 24px 30px; box-shadow:0 20px 60px rgba(0,0,0,0.35); }
        h1 { margin:0 0 8px; font-size:28px; letter-spacing:-0.5px; }
        p { margin:0 0 18px; color:var(--muted); line-height:1.5; }
        .drop { position:relative; border:1.5px dashed rgba(255,255,255,0.25); border-radius:12px; padding:18px 16px; display:flex; align-items:center; gap:12px; transition: border-color 0.2s ease, background 0.2s ease; cursor:pointer; }
        .drop:hover { border-color:var(--accent); background:rgba(124,246,185,0.05); }
        .drop input[type=file] { position:absolute; inset:0; width:100%; height:100%; opacity:0; cursor:pointer; }
        .file-name { font-weight:600; color:var(--text); word-break: break-all; }
        .hint { color:var(--muted); font-size:14px; }
        button { margin-top:16px; width:100%; background:linear-gradient(120deg, var(--accent), var(--accent-2)); color:#041021; border:none; border-radius:10px; padding:14px; font-weight:700; letter-spacing:0.3px; cursor:pointer; transition:transform 0.1s ease, box-shadow 0.1s ease; }
        button:hover { transform:translateY(-1px); box-shadow:0 8px 18px rgba(124,246,185,0.25); }
        .grid { margin-top:20px; display:grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap:14px; }
        .panel { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px 14px 16px; }
        .title { display:flex; justify-content:space-between; align-items:center; font-weight:700; margin:0 0 6px; }
        .tag { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; background:rgba(124,246,185,0.12); color:var(--accent); font-weight:700; font-size:14px; }
        ul { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:8px; }
        li { color:var(--muted); font-size:14px; }
        .raw { margin-top:14px; background:#0c1629; border-radius:12px; padding:12px; font-family:'SFMono-Regular', Consolas, monospace; font-size:13px; color:#cfe4ff; white-space:pre-wrap; display:none; }
        .btn-small { margin-top:10px; background:none; border:1px solid rgba(255,255,255,0.12); color:var(--text); width:auto; padding:8px 12px; border-radius:8px; font-weight:600; }
        .btn-small:hover { box-shadow:none; transform:none; border-color:var(--accent); }
        @media (max-width: 640px) { h1 { font-size:24px; } }
      </style>
    </head>
    <body>
      <main class=\"shell\">
        <div class=\"card\">
          <h1>Clusterize seu áudio</h1>
          <p>Envie um trecho de música (mp3/wav). Extraímos features de áudio e rodamos o K-Means para te dizer o cluster mais próximo com uma leitura rápida.</p>

          <label class=\"drop\">
            <div>
              <div class=\"file-name\" id=\"file-name\">Selecione um arquivo de áudio</div>
              <div class=\"hint\">MP3/WAV • use 5–20s para resposta rápida</div>
            </div>
            <input id=\"file\" type=\"file\" accept=\"audio/*\" />
          </label>
          <button id=\"send\">Analisar e prever</button>

          <div class=\"grid\">
            <div class=\"panel\">
              <div class=\"title\">
                <span>Seu áudio</span>
                <span id=\"status\" class=\"tag\" style=\"display:none;\"></span>
              </div>
              <div id=\"summary\" style=\"color:var(--muted); line-height:1.5;\">Envie um áudio para ver a leitura rápida aqui.</div>
            </div>
            <div class=\"panel\">
              <div class=\"title\"><span>Cluster atribuído</span></div>
              <div id=\"cluster-label\" style=\"font-weight:700; margin-bottom:4px; color:var(--text);\">-</div>
              <div id=\"cluster-summary\" style=\"color:var(--muted); line-height:1.4;\">Sem dados ainda.</div>
            </div>
            <div class=\"panel\">
              <div class=\"title\"><span>Insights do áudio</span></div>
              <ul id=\"insights\"><li style=\"opacity:0.6;\">Sem dados ainda.</li></ul>
            </div>
            <div class=\"panel\">
              <div class=\"title\"><span>Proximidade com clusters</span></div>
              <ul id=\"proximity\"><li style=\"opacity:0.6;\">Sem dados ainda.</li></ul>
            </div>
          </div>

          <button class=\"btn-small\" id=\"toggle-raw\">Ver JSON bruto</button>
          <pre id=\"raw\" class=\"raw\"></pre>
        </div>
      </main>

      <script>
        const btn = document.getElementById('send');
        const fileInput = document.getElementById('file');
        const fileName = document.getElementById('file-name');
        const statusEl = document.getElementById('status');
        const summary = document.getElementById('summary');
        const clusterLabel = document.getElementById('cluster-label');
        const clusterSummary = document.getElementById('cluster-summary');
        const insights = document.getElementById('insights');
        const proximity = document.getElementById('proximity');
        const raw = document.getElementById('raw');
        const toggleRaw = document.getElementById('toggle-raw');

        fileInput.addEventListener('change', () => {
          const file = fileInput.files[0];
          fileName.textContent = file ? file.name : 'Selecione um arquivo de áudio';
        });

        toggleRaw.addEventListener('click', () => {
          const isHidden = raw.style.display === 'none' || raw.style.display === '';
          raw.style.display = isHidden ? 'block' : 'none';
          toggleRaw.textContent = isHidden ? 'Ocultar JSON' : 'Ver JSON bruto';
        });

        btn.addEventListener('click', async () => {
          const file = fileInput.files[0];
          if (!file) {
            alert('Escolha um arquivo de áudio');
            return;
          }
          statusEl.style.display = 'inline-flex';
          statusEl.textContent = 'Processando…';
          summary.textContent = 'Lendo e extraindo features...';
          insights.innerHTML = '<li>...</li>';
          proximity.innerHTML = '<li>...</li>';
          raw.style.display = 'none';
          toggleRaw.textContent = 'Ver JSON bruto';

          const form = new FormData();
          form.append('file', file);

          try {
            const res = await fetch('/analyze', { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Erro na API');

            statusEl.textContent = data.cluster_label || `Cluster ${data.cluster}`;
            summary.textContent = data.track_summary || 'Leitura indisponível.';
            clusterLabel.textContent = data.cluster_label || '-';
            clusterSummary.textContent = data.cluster_summary || 'Sem resumo do cluster.';

            insights.innerHTML = '';
            (data.track_insights || []).forEach(item => {
              const li = document.createElement('li');
              li.textContent = item;
              insights.appendChild(li);
            });
            if (!insights.children.length) insights.innerHTML = '<li>Sem insights.</li>';

            proximity.innerHTML = '';
            (data.proximity || []).forEach(item => {
              const li = document.createElement('li');
              li.textContent = `Cluster ${item.cluster}: afinidade ${(item.score * 100).toFixed(0)}%`;
              proximity.appendChild(li);
            });
            if (!proximity.children.length) proximity.innerHTML = '<li>Sem dados.</li>';

            raw.textContent = JSON.stringify(data, null, 2);
          } catch (err) {
            statusEl.textContent = 'Falhou';
            summary.textContent = err.message;
            insights.innerHTML = '<li>Erro ao processar.</li>';
            proximity.innerHTML = '<li>Erro ao processar.</li>';
            raw.textContent = err.message;
          }
        });
      </script>
    </body>
    </html>
    """
