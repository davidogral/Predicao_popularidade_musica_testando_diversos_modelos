# Resultados

## Visão geral do dataset
- Base SpotifyFeatures com 232,7k faixas; após limpeza/seleção de colunas numéricas ficaram 176.774 exemplos para modelagem.
- Target de popularidade binária gerou proporção ~63,5% não-popular vs. 36,5% popular.

## Não supervisionado (K-means)
- Features: `danceability`, `energy`, `acousticness`, `speechiness`, `valence`, `tempo`, `duration_ms` (log1p), `instrumentalness`, `liveness`, `loudness`, escaladas com StandardScaler.
- K escolhido: 3 (melhor equilíbrio pelo silhouette/elbow). Modelo salvo em `nao_supervisionado/models/kmeans_k3.joblib`.
- Tamanhos dos clusters: 120.975 (C1), 45.591 (C2), 10.208 (C0).
- Perfis e popularidade média:
  - C1 — Hit dançante/urbano (pop/rap/reggaeton): popularidade 40,5; alta danceability/energy; tempo ~123 BPM; loudness -7 dB.
  - C2 — Acústico/Orquestral (clássico/trilha): popularidade 28,5; alta acousticness e instrumentalness (~0,46); tempo ~105 BPM; loudness -18 dB.
  - C0 — Falado/Narrativo (comedy/movie/infantil): popularidade 20,9; speechiness ~0,87; liveness ~0,73; tempo ~98 BPM.
- Gêneros dominantes: C1 — Alternative/Reggae/Reggaeton; C2 — Opera/Soundtrack/Classical; C0 — Comedy/Movie/Children’s Music.

## Supervisionado (classificação de popularidade)
- Base balanceada para treino/teste com as mesmas features numéricas + dummies de `key`, `mode`, `time_signature`.
- Desempenho em teste:
  - Linear SVM: acurácia 0,852.
  - Random Forest: acurácia 0,856 (matriz confusão — verdadeiros: 40.427/19.303; falsos: 3.886/6.202).
  - Decision Tree: acurácia 0,787 (matriz confusão — verdadeiros: 36.771/18.166; falsos: 7.542/7.339).
  - XGBoost: acurácia 0,856 (matriz confusão — verdadeiros: 40.324/19.463; falsos: 3.989/6.042).
- Modelos mais fortes: Random Forest e XGBoost empatados em acurácia e com confusões semelhantes; SVM próximo e mais leve.
