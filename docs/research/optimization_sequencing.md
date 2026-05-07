# Optimization Sequencing

여러 최적화 lever 사이의 의존성과 권장 진행 순서. 시간 지나면서 update.

## Pending levers

| Lever | 어디서 nail되는지 | 현재 상태 |
| --- | --- | --- |
| **Model size growth** (hidden ≥ 1024 / layers ≥ 6) | `docs/plans/model_size_experiment.md` | 인프라 미설치 |
| **Option B** (per-worker interleaved traversal) | plan 미작성 | 미시작 |
| **AMP** trainer | `docs/plans/archive/amp_trainer.md` (구현 됨, default off) | 모델 키운 후 재측정 |
| **torch.compile** trainer | `docs/plans/torch_compile.md` | 모델 키운 후 재측정 |
| **TensorRT** inference | plan 미작성 | 모델 키운 + eval dense 시점 |
| **Option A re-enable** | 코드 있음 (default off) | 모델 키운 후 또는 Option B 후 |
| **Julia port** | `docs/research/julia_port_evaluation.md` | Torch.jl 결과 대기 중. Flux FAIL. |

## 의존성 그래프

```
Julia decision (Torch.jl pending)
   │
   ├── PASS → port 작업 (큰 분기)
   └── FAIL → Python/Cython 유지
                │
                ▼
         Option B 인프라 작업
                │
                ▼
         모델 크기 실험 (200 iter × 4 config)
                │
                ▼
   ┌────────────┼────────────┬─────────────┐
   ▼            ▼            ▼             ▼
   AMP 재측정   compile      TensorRT     Option A
   재측정       (eval-heavy)  재활성
```

## 권장 순서와 사유

**1. Julia 결정 확정 (지금 진행 중)** — Torch.jl criterion 4 결과 도착하면 Julia GO/NO-GO 자동 판정. 이게 "어떤 런타임 위에서 최적화하나"를 결정.

**2. 인프라 최적화 먼저, 모델 크기 실험 뒤에** — 핵심 원칙.
   - 모델 크기 실험은 1회성 아님. 4 config × 200 iter = 800 iter의 한 번이 아니라 **여러 변수 바꾸면서 다회 재측정**.
   - 인프라 1.3× 빨라지면 모든 미래 학습 실험에 복리 적용. 모델 크기 실험만의 누적 절약 + 그 후 ablation/seed 다중 실행 + selectivity 연구 등 전부 혜택.
   - 거꾸로 가면 (모델 크기 → 최적화) 모델 크기 실험 다시 돌려야 할 때 기존 측정값이 인프라 다른 상태에서 나온 거라 비교 신뢰도 낮음.

**3. AMP/compile/TRT는 모델 키운 후** — 이건 doc 곳곳에 이미 박힘. 작은 모델에서 dispatch 오버헤드 > kernel 이득. `docs/performance.md` "Post-A Optimization Calculus" 참조.

**4. Option A 재활성은 옵션 B 또는 모델 키움 후** — `docs/performance.md` "Option A Bench Result" 참조.

## 현재 진행 상태 (2026-05-07)

- ⏳ Julia Torch.jl criterion 4 측정 중
- ⏸ Option B: plan 미작성, Julia 결과 후
- ⏸ 모델 크기 실험 인프라: 미시작
- ✅ Cython 봇 (~2.55× on opponent_act)
- ✅ Option A 코드: 머지됨, default off
- ✅ AMP 코드: 머지됨, default off

## 결정 점들

언제 이 doc 다시 보는가:
- Torch.jl 결과 들어올 때 (Julia 트랙 closed/branched)
- Option B plan 작성 시점
- 모델 크기 실험 결과 받을 때 (downstream lever 4개 모두 trigger 결정)
