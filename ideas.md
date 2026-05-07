# Lost Cities Deep CFR — Ideas 압축 정리

## Active Research Threads

- [Lost Cities selectivity 가설 / 4-model 분석](docs/research/lost_cities_selectivity.md)
- [Julia 포팅 검토 — 단일/멀티스레드/ML 평가](docs/research/julia_port_evaluation.md)

## 진단 가설들

- Variance 가설 — Draw variance가 entry regret signal 압도. Reward smoothing 1순위.
- Architecture 가설 — Flat MLP가 action-conditional advantage 분리 못 함.
- Open and recover 가설 — 정책이 entry filter 대신 수습 학습 수렴.
- Action-local credit assignment 가설 — Skip color action 부재로 regret 분산. Bad_open metric 자체 mismatch 가능.
- Average / League inertia 가설 — Current policy는 selectivity 학습해도 average/league가 끌고 있음.
- All-negative fallback 가설 — Uniform fallback이 학습 dynamic hole. 검증됨, 부분 풀림.
- Self-play attractor 가설 — 5-color stable equilibrium.
- Lost Cities NE = 5-color 가설 — 가능성 낮지만 0 아님.

## 개입 아이디어들

### Diagnostic (cheap, 정보량 큼)

- Current vs Average vs League policy 분리 측정
- Empirical r̃ by action class
- BR-to-current 진단
- Tabular Lost Cities oracle
- 색별 opening rate, unopened color score

### Architectural

- Open-gate (latent skip color)
- Action-factorized scorer + per-action features
- Color permutation equivariance/augmentation
- Slot-shared encoder
- Dueling head
- Two-headed MLP

### Training dynamics

- ✅ Argmax_tiebreak fallback (적용됨, mechanism 작동)
- LCFR / DCFR weighting (reservoir inertia 직접 공격)
- Reward smoothing
- Open-action regret sign auxiliary
- First-open replay reweighting
- Type-balanced RM epsilon
- League weight 조정 (older 비중 감소)
- Memory discounting

### Framework level

- PSRO-style population

## 새 measurement 항목

- Δ_open 기반 calibration metric
- Policy 분리 (current/average/league) opened_colors
- Fallback breakdown (rate, action 분포, opened_colors bucket, tie rate)
- Empirical r̃ by action class
- 색별 opening rate, unopened rate
- Avoided open penalty proxy
