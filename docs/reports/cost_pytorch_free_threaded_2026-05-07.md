# Cost Report: Adopting Free-Threaded Python (3.13t / 3.14t) for PyTorch

**Date**: 2026-05-07
**Scope**: Evaluate the cost of migrating this project's Deep CFR training stack
to free-threaded Python (PEP 703 / `python3.13t` / `python3.14t`) so that the
multiprocessing traversal workers + inference server can be replaced by
threads, escaping the structural batching ceiling documented in
`docs/performance.md` ("Option A Bench Result and Structural Ceiling").

## 한 줄 결론

**WAIT (3-6개월).** PyTorch 측 free-threaded 지원은 이미 production 수준
(2.10 cp314t wheel, 2.11 experimental)이지만, **우리 코드 측 진짜 비용은
PyTorch가 아니라 Cython 게임 엔진의 `nogil` 정합성 작업**이다. 그 작업을
하지 않으면 free-threaded Python으로 전환해도 traversal 핫패스에서 GIL이
없어진 효과를 못 본다 — 단지 단일-스레드 perf 회귀(현재 ~20-40%, Python
3.14에서 ~5-10%로 개선 예정)만 떠안게 된다. 더 직접적이고 더 작은 우회로
(Option B-shape per-worker interleaving)가 같은 batch-size 목표를 더 적은
ecosystem-risk로 달성한다.

---

## PyTorch 측 상태 (2026-05 기준)

### Wheel 가용성

- **`torch` 2.10.0** (2026-01): cp313t / cp314t 정식 wheel을 PyPI에 게시.
  Linux x86_64 / aarch64, Windows, macOS 모두 커버.
  ([PyPI torch](https://pypi.org/project/torch/),
  [PyTorch Issue #156856](https://github.com/pytorch/pytorch/issues/156856))
- **`torch` 2.11.0** (2026-05 직전): cp314t를 "experimentally supported"로
  명시. cp313t는 stable.
  ([PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/),
  [PyTorch 2.11 Release Notes](https://github.com/pytorch/pytorch/releases))
- `pip install torch` on `python3.13t` 환경에서 실제로 동작한다는 트래킹
  페이지 보고가 있음.
  ([Free-threading Compatibility Tracking](https://py-free-threading.github.io/tracking/))

### CUDA wheel 주의사항

- **Python 3.14 (정규/free-threaded)** 의 CUDA wheel 배포 늦음 보고가 있다.
  ([PyTorch Issue #169929](https://github.com/pytorch/pytorch/issues/169929))
  3.14t에서 CUDA를 쓰려면 wheel 인덱스/플랫폼을 확인해야 한다. 우리 RTX 3090
  로컬은 cp313t + CUDA가 안전한 조합.

### 안전한 영역

- 기본 텐서 연산 (`torch.as_tensor`, `nn.Linear`, ReLU 등 우리 MLP에서
  쓰는 모든 op)은 GIL을 이미 release한다. Quansight/Meta 보고에서
  멀티스레드에서 잘 도는 것으로 확인.
  ([Quansight: free-threaded rollout](https://labs.quansight.org/blog/free-threaded-python-rollout))
- `torch.inference_mode()` 컨텍스트, `state_dict` load, MLP forward —
  우리 inference server가 쓰는 모든 경로가 잘 알려진 멀티스레드-가능 영역.

### 위험 영역 / gated

- **`torch.compile`**: free-threaded 빌드에서 "기본 동작은 가능, 진짜
  멀티스레드 사용은 미지원"으로 명시.
  ([PyTorch Issue #156856](https://github.com/pytorch/pytorch/issues/156856))
  우리 프로젝트는 `torch.compile` 회귀가 이미 확인되어 비활성 상태이므로
  이 제약은 영향 없음 (`docs/performance.md` "torch.compile 실험" 절).
- **`DataLoader`**: 멀티스레드 DataLoader는 SPDL/Meta 측에서 prototype 단계
  시연이 있을 뿐, stable contract 아님. 우리 프로젝트는 DataLoader를 쓰지
  않으므로 영향 없음.
- **CUDA 멀티스레드 컨텍스트 공유**: long-standing 주의 사항. 한 프로세스
  안에서 여러 스레드가 같은 CUDA stream에 일을 던지면 race/순서 문제 가능.
  현재 우리 inference server는 단일-스레드 디스패치 루프 (`run_inference_server`,
  `inference_server.py:218-246`)이므로 free-threaded로 전환해도 forward
  자체는 단일 스레드가 처리하면 안전.
  ([PyTorch Forums: thread safety + multiprocessing CUDA](https://discuss.pytorch.org/t/thread-safety-in-multiprocessing-cuda-tensors-dont-update-asynchronously/160151),
  [CUDA semantics 2.11](https://docs.pytorch.org/docs/2.11/notes/cuda.html))
- **Autograd**: 멀티스레드 backward는 PyTorch에서 historically 락이 많음.
  우리 trainer는 단일 스레드에서 backward를 돌릴 것이므로 영향 없음 —
  단, 멀티스레드 inference + 동시 backward를 같은 모델에 시도하지 않아야
  한다는 일반 원칙은 그대로.

### 단일-스레드 perf 회귀

- **3.13t**: specializing adaptive interpreter 비활성화로 단일-스레드
  코드가 ~20-40% 느림.
  ([CodSpeed: State of 3.13](https://codspeed.io/blog/state-of-python-3-13-performance-free-threading))
- **3.14t**: specializing interpreter 재활성화. 회귀가 ~5-10%로 축소
  (예상). 3.14는 PEP 779 통과 후 free-threaded가 "non-experimental,
  officially supported"로 격상됨.
- **PyTorch 자체**: native code는 GIL을 이미 release하므로 free-threaded
  빌드에서도 텐서 연산 perf는 거의 동일 (Trent Nelson 보고).
  ([Trent Nelson: PyTorch + free-threading](https://trent.me/articles/pytorch-and-python-free-threading/))
- 우리 traversal hot path는 Cython이라 Python 인터프리터 회귀의 직접
  영향이 작지만, Python 콜백 (정책 호출 → numpy → torch) 빈도가 매우
  높으므로 (~205k calls/iter) 회귀가 곱 effect로 누적될 수 있다.

### 알려진 이슈 / 트래커

- [pytorch#130249 — Python 3.13 support](https://github.com/pytorch/pytorch/issues/130249) —
  지속 업데이트되는 메타 이슈.
- [pytorch#156856 — Python 3.14 support](https://github.com/pytorch/pytorch/issues/156856) —
  3.14 / 3.14t 진행 상황.
- [pytorch#169929 — Python 3.14 CUDA wheel 누락 보고](https://github.com/pytorch/pytorch/issues/169929)

### 생태계 신호 (실제로 쓰는 사람들)

- **Optuna**: 3.13t를 정식 지원, 멀티스레드 trial 실행 검증.
  ([Optuna 3.13t support](https://medium.com/optuna/overview-of-python-free-threading-v3-13t-support-in-optuna-ad9ab62a11ba))
- **Meta SPDL** (DataLoader 대체): ImageNet 이터레이터에서 process →
  thread 전환으로 +74% throughput / -50GB 메모리 보고 (8x A100).
  단, 이건 고정-비용 비교가 아닌 cherry-picked benchmark.
- **SGLang**: 3.14t 지원 요청 issue가 열려 있음 (open).
  ([sglang#22889](https://github.com/sgl-project/sglang/issues/22889))
  → 즉, **메이저 LLM serving 프레임워크조차 아직 production 도입을 안
  했다**는 신호.
- **Lightning / RLlib**: free-threaded 도입 공식 발표 없음 (검색 시점).
- 일반 보고: PyO3 dependent Rust extensions (pydantic, tiktoken 등) 일부가
  free-threaded wheel 없음 → setup이 "fiddly". 우리 프로젝트는 이런
  의존성이 사실상 없음 (Cython만 있음 — 이건 free-threaded 지원 wheel
  배포 진행 중).

---

## 우리 코드 측 작업량

다음은 multiprocessing → threading 전환 시 만져야 할 곳 / 동시성 가정을
재검토해야 할 곳을 파일·함수 단위로 정리한 것.

### 파일·함수 인벤토리

| 파일 | 역할 | 현재 동시성 가정 | 스레드化 비용 |
| --- | --- | --- | --- |
| `src/coolrl_lost_cities/games/classic/deep_cfr/trainer.py` (`_run_traversal_iteration`, `_evaluate_iteration` 부근, line 461-525, 818-855) | `ProcessPoolExecutor(mp_context=spawn)`로 worker batch dispatch | 워커는 별 프로세스, fork-safe 가정 없음 (spawn) | `ThreadPoolExecutor`로 교체. 각 worker가 trainer의 model state에 read-only 접근 → state_dict copy 시점만 lock으로 보호. 중간. |
| `src/coolrl_lost_cities/games/classic/deep_cfr/workers.py` (`run_traversal_worker_batch`, `_configure_worker_torch_threads` line 28-46) | per-process `torch.set_num_threads(1)`, networks를 매 batch마다 `state_dict`로 load 후 eval | 프로세스마다 격리된 torch state, model copy 1쌍 | thread-shared model로 단순화 가능. `torch.set_num_threads(1)` 호출은 process-global이라 thread 환경에서는 1번만 호출하면 됨 — 약간 작업. **MODEL을 공유하는 순간 weight-update 동시성 문제 신규 발생** (현재는 매 batch 시작 시 state_dict 복사라 자연스레 안전). 중간-높음. |
| `src/coolrl_lost_cities/games/classic/deep_cfr/inference_server.py` (`InferenceServerController`, `run_inference_server`, line 218-326) | 별 프로세스 + spawn context, shared-memory tensor pool | 프로세스 격리 → thread 환경에서는 server 자체가 불필요. 같은 주소 공간에서 직접 호출. | 서버 자체 삭제 또는 in-process thread-pool 디스패처로 변환. **다만 이 변환이 free-threaded migration의 진짜 목적이므로 비용이라기보다 보상.** 중간. |
| `src/coolrl_lost_cities/games/classic/deep_cfr/inference_client.py` (`InferenceClient.forward`, `NetworkProxy.__call__`) | shared-memory slot 잡고 queue post → event wait | 슬롯 = 워커 1개 가정. 슬롯 free pool은 `mp.Queue` 기반. | 스레드化하면 그냥 직접 model forward 호출 가능. 작은 인터페이스 어댑터만 필요. 작음. |
| `src/coolrl_lost_cities/games/classic/deep_cfr/inference_buffers.py` (`InferenceBuffers`, line 45) | `mp.get_context("spawn")` 기반 shared memory + queues | mp 전용 | 스레드 환경에서는 통째로 불필요. 작음 (삭제). |
| `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx` (1179줄) | Cython 재귀 traversal, 정책 호출, regret 누적 | **GIL 보유 가정**. Python object 접근 다수. 자유 스레드化 시 race 위험 미평가. | **이게 진짜 비용.** `nogil` 클린업 audit 필요. 1179줄 + `cfr_math.pyx` + `encoding.pyx` 전부. **수일~수주 작업, high risk** (`docs/performance.md`도 같은 결론). |
| `src/coolrl_lost_cities/games/classic/deep_cfr/memory.py` (73줄, `TrainingSample` 추가 경로) | 현재 워커가 결과를 list로 반환, trainer가 단일 스레드에서 `memory.add(...)` | 스레드 추가 시 add 호출이 동시 발생 → 락 필요 | replay buffer add 경로에 `threading.Lock` 추가. 작음. |
| `src/coolrl_lost_cities/games/classic/deep_cfr/inference_server.py` (`run_inference_server` 디스패치 루프, `torch.inference_mode()` + `torch.as_tensor` line 231-246) | 단일 프로세스 단일 스레드 디스패치 | 스레드化 후에도 단일 디스패처 thread 1개로 유지 가능 (CUDA stream 안전) | 단일 thread 보장. 작음. |
| Cython `.pyx` (encoding, cfr_math, traversal) | numpy + Python object 빈번 | `nogil` cleanup 비용 매우 큼 | **자유 스레드化의 critical path.** |

### 새로 필요한 동시성 primitive 위치 추정

다음 곳에 명시적 락 또는 per-thread 격리가 필요해진다:

1. `trainer.py`: 모델 weight 업데이트 ↔ inference 디스패처 read 사이 —
   현재는 `weight_queue`가 알아서 sync. 스레드化 후엔 RWLock 또는 epoch
   기반 swap 필요. **1곳, 패턴 명확.**
2. `memory.py`: replay buffer `add` 경로. **1곳, 평범한 lock으로 해결.**
3. `traversal_stats.py`: stats 누적 — thread-local 후 머지가 깔끔. **1곳.**
4. `traversal.pyx`: Python-level state mutation (regret/strategy
   accumulator dict 등) — `nogil` 영역에서 건드리면 안 됨. **이게 가장
   많고 가장 어려운 곳.** 정확한 location 수는 audit 전엔 미상이지만
   재귀 호출마다 등장.

### 요약 추정

- **PyTorch 사용 위치만 카운트하면**: 명시적 locking이 새로 필요한 자리는
  3-5곳 정도로 매우 작다 (weight swap, replay add, stats merge).
- **진짜 작업량**: Cython 엔진 `nogil` cleanup. 이건 PyTorch 측 free-
  threaded 지원과 무관하게 필요한 상수 비용이고, `docs/performance.md`
  Decision A의 "free-threaded는 옵션 있지만 cleanup 비용 때문에 deferred"
  와 일치한다.

---

## 위험 표

| # | 위험 | 발생 확률 | 영향도 | 검증 방법 |
| -: | --- | --- | --- | --- |
| 1 | Cython traversal 핫패스가 `nogil`-clean이 아니어서 스레드 추가가 곧 GIL 직렬화로 환원 | **높음 (사실상 확정)** | High — 마이그레이션 목적 자체가 사라짐 | `traversal.pyx` 함수에 `nogil` 어노테이션을 시험 적용 → 컴파일 에러로 Python-object 접근 위치 enumerate. 1-2일. |
| 2 | 단일-스레드 perf 회귀 (3.13t: 20-40%) → traversal 절대 시간이 미세하게 더 느려짐 | 높음 (3.13t에서) / 중간 (3.14t에서 ~5-10%) | Medium | 같은 머신에서 `python3.13` vs `python3.13t`로 현 traversal 벤치 (`scripts/bench_inference_backend.py`) 비교. 반나절. |
| 3 | PyTorch CUDA wheel이 cp314t에서 누락/늦음 → CUDA 트레이너에서 import 실패 | 중간 | High (이라면 즉시 블로커) | `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126` on `python3.14t` 시도. |
| 4 | Cython 의존성 (numpy 등)이 free-threaded wheel 부재 또는 thread-unsafe op | 낮음-중간 | Medium | `pip install` smoke + import 테스트. numpy 2.3+는 free-threaded compat. |
| 5 | 같은 모델에 multi-thread inference + concurrent backward에서 autograd 락 경합 | 중간 (구조에 따라) | Medium | 우리 구조는 단일-스레드 backward라 회피 가능. 디자인 단계에서 "trainer step 동안 inference 중지" 약속만 지키면 됨. |
| 6 | 멀티스레드 CUDA stream 사용 시 race | 낮음 (단일 디스패처 thread 유지하면 0) | High if hit | inference server를 thread 1개로 제한. 검증: `torch.cuda.synchronize()` + functional test. |
| 7 | 생태계 미성숙 — wandb / pytest / ruff / 기타 dev 툴 free-threaded 호환성 | 중간 | Low (개발 환경만 영향, 학습은 OK) | 새 venv 만들어 `uv sync` 시도. |
| 8 | PEP 703 정책 변경 / Python 측 backout (낮지만 0 아님) | 매우 낮음 | High | 트래커 모니터. 3.14에서 phase 2 (officially supported)로 격상되어 위험 감소. |
| 9 | 우리가 적용한 시점이 아직 너무 일러 회귀 발생 시 upstream에 patch 못 받음 | 중간 | Medium | SGLang / Lightning 등 production 도입 신호 대기. 현재 미도입. |

---

## 권고

### 지금 시도하지 말 것

이유 3줄:

1. **PyTorch는 충분히 준비됐지만, 우리 병목은 PyTorch가 아니다.**
   `docs/performance.md` Option A 분석은 batching ceiling이 *Cython
   traversal의 sync-blocking 구조* 때문이라고 명시. free-threaded Python은
   "워커를 스레드로 바꿀 수 있게" 해주지만, Cython이 `nogil`이 아니면
   스레드끼리 GIL을 직렬-획득하므로 ceiling이 안 올라간다.
2. **단일-스레드 perf 회귀 (3.13t 20-40%)** 가 traversal 절대 시간을
   악화시킬 수 있다. 3.14t에서 5-10%로 개선될 때까지 기다리는 편이 비용
   대비 안전.
3. **Production 도입 신호가 아직 약하다**. SGLang은 issue가 open이고,
   Lightning/RLlib는 발표 없음. 우리가 early adopter가 되어 디버깅 비용을
   짊어질 가치는 단일 프로젝트 입장에서 낮다.

### 더 작은 우회로 (이미 plan에 있음)

`docs/performance.md` "Re-enable A when one of these holds" 항목 #2
**per-worker interleaved traversal (Option B-shape)** 가 같은 batch=64
목표를 free-threaded migration 없이 달성한다. Cython 재귀를 resumable
state machine으로 바꾸는 작업은 `nogil` audit보다 **로컬 영역이고 risk가
낮다** (단일 worker scope, 검증 가능, ecosystem 의존성 0). 자유-스레드
이주의 대안으로 Option B를 먼저 시도하는 것을 권장.

### 다시 보는 조건 (트리거)

다음 중 하나라도 충족되면 재평가:

- **(a) Cython 엔진을 다른 이유로 `nogil`-clean 화한다** — 그 경우
  free-threaded Python은 거의 무료 부산물이 된다. 비용 대부분이 그쪽
  작업에 흡수됨.
- **(b) Python 3.14.x patch release에서 free-threaded가 stable로 격상되고
  단일-스레드 회귀가 ≤5%로 측정됨** + **메이저 ML 프레임워크 (Lightning,
  RLlib, vLLM, SGLang 중 2개 이상)** 가 production 도입 발표.
- **(c) 모델이 1024-hidden / 6-layer 이상으로 커져서** GPU forward가
  IPC 오버헤드를 흡수할 수 있는 영역으로 들어가면 — Option A 자체가
  다시 살아나므로 free-threaded migration이 필요 없어진다.
- **(d) eval이 dominant phase가 됨** (eval_every=5, games=1000). eval은
  이미 batch=64이므로 free-threaded와 무관하게 Option A 재활성화로 충분.

### 액션

1. **지금 (0 비용)**: `docs/performance.md`의 free-threaded note를 이
   리포트로 cross-link. 트래킹 페이지
   ([py-free-threading.github.io/tracking](https://py-free-threading.github.io/tracking/))
   를 분기별로 확인.
2. **다음 작업 후보**: Option B-shape per-worker interleaving 설계 검토.
3. **장기**: 모델 사이즈 결정이 끝나면 (`docs/performance.md` 권장
   sequencing #2) Option A의 batch ceiling이 자연 해소되는지 재측정.

---

## 참고

- [PyTorch Issue #130249 — Python 3.13 support](https://github.com/pytorch/pytorch/issues/130249)
- [PyTorch Issue #156856 — Python 3.14 support](https://github.com/pytorch/pytorch/issues/156856)
- [PyTorch Issue #169929 — Python 3.14 CUDA wheel](https://github.com/pytorch/pytorch/issues/169929)
- [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- [PyTorch 2.11 Release Notes](https://github.com/pytorch/pytorch/releases)
- [py-free-threading Compatibility Tracking](https://py-free-threading.github.io/tracking/)
- [Quansight Labs — Free-threaded rollout](https://labs.quansight.org/blog/free-threaded-python-rollout)
- [CodSpeed — State of Python 3.13 free-threading perf](https://codspeed.io/blog/state-of-python-3-13-performance-free-threading)
- [Trent Nelson — PyTorch + Free-Threading 실전](https://trent.me/articles/pytorch-and-python-free-threading/)
- [Optuna — 3.13t 지원](https://medium.com/optuna/overview-of-python-free-threading-v3-13t-support-in-optuna-ad9ab62a11ba)
- [SGLang Issue #22889 — 3.14t 지원 요청 (open)](https://github.com/sgl-project/sglang/issues/22889)
- [PyTorch CUDA semantics 2.11](https://docs.pytorch.org/docs/2.11/notes/cuda.html)
- 사내: `docs/performance.md` "Option A Bench Result and Structural Ceiling",
  "Free-threaded Python (3.13t) note"
