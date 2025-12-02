# Dicionário de Dados – SpotifyFeatures

| Coluna            | Português       | Descrição |
|-------------------|---------------------------------|-----------------------------|
| `genre`           | Gênero musical                  | Categoria atribuída pelo Spotify (ex.: Movie, Pop, Rock) usada para segmentações por estilo. |
| `artist_name`     | Artista                         | Nome do(a) intérprete principal; permite identificar autores e cruzar discografias. |
| `track_name`      | Nome da faixa                   | Título oficial exibido ao usuário e útil para buscas manuais. |
| `track_id`        | ID da faixa no Spotify          | Identificador único alfanumérico usado em APIs e para garantir unicidade mesmo com nomes repetidos. |
| `popularity`      | Popularidade (0–100)            | Índice calculado pelo Spotify com base em plays e engajamento recentes; valores altos indicam faixas atuais e muito ouvidas. |
| `acousticness`    | Caráter acústico (0–1)          | Probabilidade de a faixa ser predominantemente acústica; perto de 1 indica instrumentação orgânica, perto de 0 indica produção eletrônica/amplificada. |
| `danceability`    | Bailabilidade (0–1)             | Mede quão fácil é dançar a música considerando ritmo, estabilidade e regularidade dos batimentos; valores altos têm groove consistente. |
| `duration_ms`     | Duração em ms                   | Tempo total em milissegundos; divida por 60.000 para minutos. Valores altos representam faixas longas. |
| `energy`          | Energia (0–1)                   | Intensidade sonora baseada em volume, densidade e dinamismo; valores altos indicam músicas agitadas, baixos indicam faixas calmas. |
| `instrumentalness`| Instrumentalidade (0–1)         | Probabilidade de ser instrumental; acima de ~0.5 sugere ausência de vocais, próximo de 0 aponta presença marcante de voz. |
| `key`             | Tom principal                   | Tonalidade estimada (C, C#, D, …, B). Útil para análises harmônicas; cada valor representa a nota raiz. |
| `liveness`        | "Ao vivo" (0–1)                 | Detecta ambiência com público; acima de 0.8 geralmente indica gravação ao vivo, valores baixos sugerem estúdio. |
| `loudness`        | Volume médio (dB)               | Decibéis relativos (sempre negativos); quanto mais próximo de 0, mais alta a faixa. Valores como -15 dB indicam músicas silenciosas. |
| `mode`            | Modo musical                    | `Major` (maior) tende a soar alegre; `Minor` (menor) traz sensação mais melancólica. |
| `speechiness`     | Presença de fala (0–1)          | >0.66 quase falado (podcasts, rap declamado); 0.33–0.66 mistura fala/canto; <0.33 predomina canto. |
| `tempo`           | Andamento (BPM)                 | Batidas por minuto; valores altos representam músicas rápidas, baixos indicam baladas/lentas. |
| `time_signature`  | Fórmula de compasso             | String como 4/4 ou 3/4 informando quantos tempos há por compasso e qual figura recebe o tempo. |
| `valence`         | Valência (0–1)                  | Pontuação de positividade/humor; valores altos são faixas alegres/otimistas, baixos representam músicas tristes ou tensas. |
