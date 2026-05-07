# Deep CFR `opponent_policy: network` 발산 (Policy Collapse) 분석

**Date:** 2026-05-07
**Author:** Claude Code 세션 기록
**Status:** Definitive — `opponent_policy: network`는 안정적인 학습을 보장하지 않음

## TL;DR

`traversal.opponent_policy: network` 옵션은 **자기 자신의 현재 네트워크**를
opponent로 사용한다. 이 설정으로 1000-iteration 실험을 두 번 (512x3 / 1024x4)
돌린 결과, **두 실험 모두 학습 초기 (peak) 이후 명백한 policy collapse가
발생**했다. 큰 네트워크는 발산을 늦추기는 하지만 막지는 못한다.

**권고:** 향후 학습은 `opponent_policy: self_play_league`를 기본값으로 유지할 것.

## 실험 설정

두 실험 모두 동일한 구조:

| 항목 | 값 |
|------|-----|
| `traversal.opponent_policy` | `network` |
| `traversal.traversals_per_iteration` | 2 |
| `traversal.traversals_per_player` | 70 |
| `optimization.advantage_updates_per_iteration` | 512 |
| `optimization.strategy_updates_per_iteration` | 512 |
| `optimization.learning_rate` | 3e-5 |
| `regret_matching.all_negative_fallback` | argmax_tiebreak |
| `self_play.max_snapshots` | 0 (snapshot pool 비활성) |
| Eval | 매 5 iter, 6 opponents × 100 games |
| Max iterations | 1000 |

차이점:
- **512x3**: `network.hidden_size=512, num_layers=3`
  config: `configs/deep_cfr/deep_cfr_opponent_network_512x3_1000iter.yaml`
- **1024x4**: `network.hidden_size=1024, num_layers=4`
  config: `configs/deep_cfr/deep_cfr_opponent_network_1024x4_1000iter.yaml`

두 실험 모두 동일 머신 (AMD Ryzen 5 5600X, 12 threads, CUDA GPU)에서
약간의 시간차 (4분)를 두고 병렬 실행. 병렬 실행으로 iteration time이
평균 ~2배 느려졌으나 결과 패턴(발산)에는 영향 없음.

## 결과

### 512x3 (iteration 363까지 관찰 후 중단)

| Phase | Iter | Random WR | Random Δ | Safe Heur WR | Safe Heur Δ |
|-------|------|-----------|----------|---------------|-------------|
| 초기 | 5 | 53.0% | -0.7 | 4.0% | -78.1 |
| **Peak** | **15** | **85.0%** | **+32.2** | 5.0% | -47.2 |
| 발산 시작 | 20 | 67.0% | +14.4 | 2.0% | -62.3 |
| 발산 진행 | 30 | 36.0% | -14.3 | 3.0% | -106.2 |
| 수렴 (망함) | 100 | 41.0% | -12.9 | 0.0% | -111.6 |
| 수렴 (망함) | 200 | 33.0% | -17.1 | 1.0% | -114.1 |
| 마지막 | 360 | 34.0% | -12.7 | 2.0% | -101.3 |

- **Peak는 iter 15** — Random 상대 85% win rate.
- iter 20부터 급격히 무너짐.
- iter 30 이후 Random WR이 **Random 자신(50%)보다 낮음** → 모델이
  random보다 못 한 상태로 수렴.
- Safe Heuristic 상대로는 처음부터 끝까지 ~0–8% (전혀 학습 안 됨).

### 1024x4 (iteration 231까지 관찰 후 중단)

| Phase | Iter | Random WR | Random Δ | Safe Heur WR | Safe Heur Δ |
|-------|------|-----------|----------|---------------|-------------|
| 초기 | 5 | 35.0% | -14.9 | 1.0% | -91.8 |
| Plateau | 30 | 62.0% | +15.0 | 3.0% | -82.2 |
| Plateau | 55 | 63.0% | +16.0 | 5.0% | -69.2 |
| **Best Random** | **70** | **70.0%** | **+17.6** | 6.0% | -74.1 |
| **Best Safe Heur** | **85** | 63.0% | +11.4 | **13.0%** | **-52.3** |
| 발산 시작 | 100 | 49.0% | +3.8 | 3.0% | -89.7 |
| 발산 진행 | 150 | 35.0% | -17.4 | 5.0% | -78.1 |
| 수렴 (망함) | 200 | 24.0% | -24.6 | 9.0% | -71.3 |
| 마지막 | 230 | 36.0% | -19.9 | 4.0% | -95.6 |

- Plateau가 **iter 30–95** (60+ iter)로 길게 유지됨.
- **Best Safe Heuristic WR = 13% (iter 85)** — 512x3의 best (8%, iter 10)
  보다 명확히 우수. 큰 capacity가 다양성 표현에 도움.
- iter 100 이후부터 발산 시작, iter 150 이후 망가짐.

### 두 실험 비교

| 항목 | 512x3 | 1024x4 |
|------|-------|--------|
| Best Random WR | **85%** (iter 15) | 70% (iter 70) |
| Best Safe Heur WR | 8% (iter 10) | **13%** (iter 85) |
| Plateau 길이 | ~10 iter | ~65 iter |
| 발산 시작 시점 | iter 20 | iter 100 |
| 수렴 시 Random WR | ~30–40% | ~30–40% |

**관찰:**
- 큰 네트워크는 **plateau를 5x 이상 길게** 유지함.
- 큰 네트워크는 Safe Heuristic 상대로도 plateau 구간에 학습 가능 (13%).
- 그러나 **두 실험 모두 결국 발산**. capacity로 발산을 막지 못함.

## 원인 분석

### 1. Moving Target (가장 큰 원인)

정상적인 외부 샘플링 CFR은 traversal 동안 opponent가 **고정된** policy를
사용한다고 가정한다. `opponent_policy: network`는 **학습 중인 네트워크**를
opponent로 사용하기 때문에:

- 매 iteration마다 opponent의 policy가 바뀜.
- 이전 iteration에서 추정한 advantage가 outdated 됨.
- advantage memory의 (state, action, regret) 샘플들이 서로 다른 opponent
  policy 하에서 측정됨 → 일관성 없음.
- 결과: 학습이 자기 자신을 쫓는 무한 루프.

### 2. Echo Chamber (다양성 부재)

traverser와 opponent가 **같은 네트워크**를 공유하므로:

- 같은 약점 공유. 예를 들어 모델이 Safe Heuristic 스타일의 조심스러운
  상대를 처리하지 못하면, 그 상대를 self-play에서 만날 일이 없음.
- 약점이 advantage estimation에 표현되지 않음 → regret 신호로 학습되지
  않음.
- 결과: 모델이 "자기 자신을 이기는 데 특화된" 좁은 strategy로 수렴.
  Random 같은 다른 distribution을 만나면 처참히 패배.

### 3. CFR 수렴 보장 깨짐

Deep CFR의 이론적 수렴은 **average strategy**가 Nash에 가까워진다는
보장이며, opponent가 fixed 또는 average policy일 때 성립한다. network
policy를 매번 바뀌는 traversal opponent로 사용하면:

- external sampling의 unbiased estimate 가정 위반.
- no-regret 보장 사라짐.
- 수렴이 이론적으로 보장되지 않음 → 실제로 발산 관찰됨.

### 4. Non-stationary Regret 추정

같은 (state, action) 샘플에서 측정한 regret이 시간에 따라 다름 (opponent
가 변하니까). advantage 네트워크는:

- 새 데이터와 옛 데이터가 다른 distribution.
- 학습이 진동.
- advantage_loss가 안정적으로 줄지 않고 plateau 또는 증가.

### 5. Strategy Mode Collapse

같은 네트워크끼리 self-play하면 mixed strategy가 deterministic-like한
하나의 mode로 수렴하기 쉽다 (game-theoretic 의미에서 Nash가 mixed인 경우
에도). 이는 imperfect information game (Lost Cities 포함) 에서 본질적으로
suboptimal — exploitable.

## 왜 self_play_league는 안정적인가

| | opponent_network | self_play_league |
|---|---|---|
| Opponent policy | **moving** (현재 네트워크) | **fixed** (snapshot pool) |
| Diversity | 없음 (echo) | 다수 snapshot에서 sampling |
| Regret estimation | non-stationary | quasi-stationary |
| 이론적 수렴 | 보장 없음 | average strategy → Nash |

`self_play.max_snapshots > 0` 으로 과거 정책의 스냅샷을 pool에 저장하고
weighted sampling으로 opponent를 선택하면, 위 문제 4개 (1, 2, 3, 4) 모두
완화되거나 해결된다.

## 왜 큰 네트워크가 plateau를 늘렸나

가설: 큰 capacity는 더 다양한 strategy mode를 표현할 수 있음. echo
chamber가 단일 mode로 collapse되는 데 더 오래 걸림. 하지만 일단
collapse가 시작되면 큰 네트워크도 동일하게 발산. **근본 원인 (moving
target) 은 capacity로 해결 불가.**

부수 효과: 큰 네트워크 + plateau 동안 **Safe Heuristic 상대 13% WR**은
다른 어떤 self-play 실험에서도 보지 못한 수치. 이 단계의 checkpoint는
별도로 보존할 가치가 있을 수 있음 (단, 1024x4 실험은 발산 후 last 만 저장
되어 iter 85 checkpoint가 archive되지 않았다면 복구 불가).

## 결론 및 권고

1. `traversal.opponent_policy: network`는 **단독으로 사용하지 말 것.**
   학습 초기에 잘 되는 것처럼 보이다가 발산하므로 짧은 실험으로 위험성을
   놓치기 쉽다.

2. **기본값은 `self_play_league`** 유지. snapshot pool로 fixed/diverse
   opponent를 제공해야 안정적.

3. 그래도 `network`를 시도하고 싶다면:
   - **early stopping** 필수 (eval WR 기준 best checkpoint 보존).
   - `save_iteration_interval`을 짧게 (예: 5 iter) 설정해서 plateau 시점을
     archive.
   - peak 이후 곧바로 중단.

4. 연구 가치 있는 후속 실험:
   - `opponent_policy: average_strategy` (학습 중인 average 정책 사용).
     CFR 이론과 더 잘 부합할 가능성.
   - `opponent_policy: hybrid` (probabilistic mix of network + snapshot).
     diversity와 simplicity 절충.
   - 1024x4의 iter 85 plateau 패턴을 self_play_league에서 재현 가능한지.

## 사용된 Config 파일

- `configs/deep_cfr/deep_cfr_opponent_network_512x3_1000iter.yaml`
- `configs/deep_cfr/deep_cfr_opponent_network_1024x4_1000iter.yaml`

## Run 디렉토리

- `runs/deep_cfr/deep_cfr_opponent_network_512x3_1000iter/` (363 iter에서
  중단)
- `runs/deep_cfr/deep_cfr_opponent_network_1024x4_1000iter/` (231 iter에서
  중단)

각 디렉토리의 `metrics.jsonl`이 본 분석의 raw source.
