## Objetivo
- Agrupar faixas do Spotify em arquétipos sonoros (ex.: “pop dançante energético”, “acústico intimista”, “falado/hip‑hop”, “eletrônico acelerado”) para apoiar curadoria de playlists, direcionamento de marketing e entendimento dos drivers de popularidade.

## Sinais observados na EDA
- 232.7k músicas; variáveis de áudio majoritariamente em [0,1], prontas para normalização.
- Correlação com popularidade: `danceability` (~0.26) e `energy` (~0.25) são positivas; `acousticness` é negativa (~-0.38); `speechiness` negativa (~-0.15). `tempo` é fraca (~0.08); `duration_ms` quase neutra e muito assimétrica (skew ~9.9).
- Pairplots/dispersion sugerem faixas espalhadas em múltiplas combinações de energia/dançabilidade/acústica, o que é propício para segmentação não supervisionada.
- Pop/Rock/Hip-Hop concentram popularidade média mais alta; Comedy e Soundtrack têm popularidade baixa — reforça a utilidade de clusters que capturem perfis sonoros além do gênero.

## Features para o K-means
- Numéricas e contínuas: `danceability`, `energy`, `acousticness`, `speechiness`, `valence`, `tempo`, `duration_ms`, `instrumentalness`, `liveness`, `loudness`.
- Pré-processamento: remover `track_id`/texto; dropar duplicatas; `duration_ms` com log1p para reduzir cauda longa; `tempo` opcionalmente padronizado z-score; `loudness` já em dB, apenas padronizar.
- Padronizar todas as features (StandardScaler) antes do K-means para igualar magnitude.
- Opcional: PCA/UMAP só para visualização (não para clustering final, a menos que melhoria de separabilidade seja clara).

## Escolha de K e justificativas
- Testar K em {4, 5, 6, 8, 10, 12}.
  - K baixos (4–6) capturam macro-arquétipos facilmente acionáveis.
  - K médios (8–12) permitem nuances de ritmo/energia/acústica sem gerar clusters minúsculos num dataset grande.
- Critérios de seleção:
  - Elbow (inertia) para ver ganho marginal.
  - Silhouette e Davies-Bouldin para separação/compacidade.
  - Estabilidade por random_state (rodar 3–5 seeds) e distribuição de tamanhos (evitar clusters <<5% do total).
  - Interpretabilidade: inspecionar centróides e distribuição de popularidade/gênero por cluster; priorizar K que produza rótulos acionáveis.

## Por que tende a funcionar
- As variáveis já são densas e numéricas; K-means se beneficia de escala uniforme após padronização.
- A EDA mostrou correlações moderadas entre energia/dançabilidade/acústica e popularidade, sugerindo eixos naturais de variação que podem ser capturados em centróides.
- Volume de dados alto garante centróides estáveis e clusters robustos mesmo para K médios.
- Clusters baseados em áudio complementam gênero e popularidade, revelando perfis sonoros reutilizáveis em recomendação e posicionamento.

## Passos de implementação
1. Carregar CSV bruto (`data/raw/SpotifyFeatures.csv`) e remover colunas não numéricas (id, texto) e linhas duplicadas.
2. Aplicar log1p em `duration_ms`; opcionalmente winsorizar extremos de `tempo`/`loudness` se necessário.
3. Dividir dataset em amostra de desenvolvimento (ex.: 50k linhas estratificadas por gênero) para sintonizar K; manter full set para treinamento final.
4. Padronizar features com StandardScaler; ajustar K-means para cada K em {4,5,6,8,10,12} com 10 init e max_iter generoso.
5. Registrar métricas (inertia, silhouette, Davies-Bouldin), tamanho dos clusters e centróides ordenados por cada feature.
6. Escolher K final balanceando métrica + interpretabilidade; rotular clusters com descrições curtas (ex.: “alto energy + alta danceability + baixa acousticness”).
7. Treinar modelo final no conjunto completo, salvar scaler e K-means (pkl ou joblib) e um dicionário de descrições dos clusters.
8. Documentar exemplos de uso: sugestão de playlist por cluster, análise de popularidade média por cluster, e segmentação por gênero dentro de cada cluster.

## Benefício esperado
- Curadoria e recomendação: playlists e rádios mais coerentes por mood/energia do que apenas por gênero.
- Produto e marketing: identificar arquétipos de alta popularidade para promoções e entender lacunas no catálogo.
- Artistas/ops: direcionar lançamentos para nichos sonoros identificados, aumentando chance de aderência e descoberta.
