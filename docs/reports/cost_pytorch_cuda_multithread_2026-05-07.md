# Cost / Risk Report: PyTorch CUDA from Multiple Threads in One Process

작성일: 2026-05-07
대상 질문: Deep CFR 파이프라인을 multiprocessing → threading(또는 free-threaded
Python)으로 옮길 경우, 같은 프로세스 안에서 여러 스레드가 동시에 PyTorch CUDA
연산을 호출하는 것이 얼마나 비싸고 위험한가?

---

## 한 줄 결론

**Risky — separate-module + 적절한 stream/lock 규율이 있으면 "safe with care",
공유 모듈에 대한 동시 forward/backward/load_state_dict 조합은
serialization 없이는 silent wrong outputs 또는 segfault를 일으킬 수 있음.**
지금 코드 형태(공유 advantage/strategy network + trainer가 매 N step마다
weight push) 그대로 threading으로 옮기면 weight-sync 경계에서 데이터 레이스가
거의 확실히 발생한다.

---

## CUDA 멀티스레드 모델 (PyTorch 2.x 기준, 2026 초 시점)

### 1) Current stream은 thread-local

PyTorch C++ 코어(`c10/cuda/CUDAStream.h`)가 명시적으로 보장한다:

> "the notion of 'current stream for device' is thread local (every OS thread
> has a separate current stream, as one might expect)"
> — [pytorch/c10/cuda/CUDAStream.h](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAStream.h)

즉 thread A가 `with torch.cuda.stream(s):` 안에서 띄운 커널은 thread B의
current stream 설정과 독립이다. 하지만 *기본* current stream은 **device의 default
stream**이며, 모든 스레드가 명시적으로 `set_stream`을 하지 않으면 같은 default
stream을 공유한다. 이 경우 GPU 측에서는 직렬화된다(병렬 launch가 안 됨).

### 2) Per-thread default stream(PTDS)는 PyTorch에서 *enable되지 않음*

CUDA driver-level의 `--default-stream per-thread` 컴파일 옵션은 PyTorch
배포 빌드에서 켜져 있지 않다. 관련 트래킹 이슈
[pytorch#25540](https://github.com/pytorch/pytorch/issues/25540) 은 2019년
오픈 이후 미해결 상태로 남아 있고, 결과적으로 "여러 스레드가 default stream을
사용하면 모두 같은 legacy default stream에 직렬화된다"가 현행 동작이다.
([cuda streams run sequentially #59692](https://github.com/pytorch/pytorch/issues/59692),
[default stream is not synchronous #101300](https://github.com/pytorch/pytorch/issues/101300)).

→ **결론: 진짜 GPU 동시성이 필요하면 각 스레드가 명시적으로
`torch.cuda.Stream()`을 만들고 `with torch.cuda.stream(s):` 컨텍스트로
감싸야 한다.** 그렇지 않으면 멀티스레드 = 멀티프로세스 대비 GIL/IPC만
줄어들 뿐 GPU 자체는 직렬화된다.

### 3) Python-level forward/backward 호출의 thread safety

공식 문서/포럼 발언을 종합하면:

- **Tensor 자체는 read-only thread-safe, write는 직렬화 책임이 사용자.**
  Edward Yang: "PyTorch underlying C++ library is expected to be thread safe
  (although the Tensor object is not thread-safe for multiple writers; you need
  to synchronize that yourself)."
  ([forum #36540](https://discuss.pytorch.org/t/is-pytorch-supposed-to-be-thread-safe/36540))
- **Inference만 한다면 같은 nn.Module 인스턴스를 여러 스레드에서 동시
  `forward()` 호출해도 OK** — 단 module state를 mutate하지 않는다는 전제.
  ([forum #88583](https://discuss.pytorch.org/t/is-inference-thread-safe/88583)).
  주의: BatchNorm `train()` 모드, dropout state, lazy module init,
  `register_buffer`로 EMA 갱신 같은 건 mutate에 해당.
- **TorchScript/JIT module은 단일 인스턴스를 동시에 forward하면 안 됨**
  ([pytorch#15210](https://github.com/pytorch/pytorch/issues/15210),
  [pytorch#51452](https://github.com/pytorch/pytorch/issues/51452)).
  우리는 JIT을 안 쓰지만, `torch.compile`로 래핑된 모듈은 내부 캐시 일관성에서
  비슷한 위험을 가질 수 있다
  ([dev-discuss: compile + multithreading](https://dev-discuss.pytorch.org/t/impact-of-multithreading-and-local-caching-on-torch-compile/2498)).
- **C++ custom 모듈은 여러 스레드에서 동시 호출 시 segfault 보고**
  ([pytorch#19029](https://github.com/pytorch/pytorch/issues/19029)) — 우리
  코드엔 직접 해당하지 않지만, 의존성 라이브러리에 비슷한 게 끼어들 수 있음.

### 4) Backward / autograd

- Autograd 엔진은 **device당 1 스레드**의 worker pool로 backward를 실행한다
  ([forum #36824](https://discuss.pytorch.org/t/only-1-thread-for-backward/36824),
  [autograd notes](https://docs.pytorch.org/docs/stable/notes/autograd.html)).
  즉 두 스레드가 *서로 다른* graph에 대해 동시에 `.backward()`를 호출하면
  엔진은 receive-side에서 큐잉/locking으로 처리한다 — 코어는 thread-safe.
- 그러나 **두 스레드가 share된 graph 부분을 동시에 backward하면
  파괴(graph free) 레이스로 다른 스레드가 crash**한다(같은 forum 답변).
  Deep CFR에서 traversal worker는 backward를 안 부르므로 직접 적용은 적지만,
  trainer + 동시 inference + replay sample이 동일 텐서를 retain하는 경우
  주의해야 한다.
- `torch.no_grad()` / `torch.inference_mode()`는 **thread-local TLS 플래그**다.
  스레드별로 따로 켜야 한다. trainer 스레드가 backward 도중에 inference 스레드가
  같은 모듈을 forward하더라도 TLS가 분리되므로 모드 자체의 충돌은 없다.
  하지만 module state(파라미터)는 공유되므로 아래 weight-update 위험은 그대로다.

### 5) CUDA caching allocator

`CUDACachingAllocator`는 **per-device mutex**로 보호된다
([zdevito guide](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html),
[CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)).
`cudaEventCreate` 같은 비싼 호출은 EventPool을 둬서 멀티스레드
allocation rate가 높아도 안전하지만, 락 경합은 존재한다. 작은 모델 + 고빈도
forward 패턴(우리 traversal과 정확히 일치)은 allocator lock contention이
스루풋의 실질적 ceiling이 될 수 있다는 점이 위험 항목에 들어간다.

### 6) Free-threaded Python (3.13t / 3.14t) 상황

- PyTorch는 3.13t 빌드(`cp313t` nightly wheels)를 제공하지만 *partial
  support*. 트래킹 이슈 [pytorch#130249](https://github.com/pytorch/pytorch/issues/130249).
- DataLoader가 thread-based로 가서 ImageNet iter +74% 같은 사례가 있지만
  ([Trent Nelson's notes](https://trent.me/articles/pytorch-and-python-free-threading/)),
  **"competing Python threads feeding the same CUDA stream still need explicit
  synchronization"** — GIL이 사라져도 stream/모듈 동기화 책임은 동일하다.
- 우리 의존성 중 **Cython 확장(`game.pyx`, `traversal.pyx`, `encoding.pyx`,
  `cfr_math.pyx`)**은 `nogil` 정합성 감사가 안 된 상태(performance.md "Why not
  Cython nogil + threading" 섹션). free-threaded 빌드에서 굴리는 것은
  threading 이전 단계의 별도 risk.

---

## 우리 시나리오 매핑

현재 (multiprocessing 기준) GPU touchpoint 위치:

| 위치 | 파일:라인 | 역할 | 현재 격리 수준 |
| --- | --- | --- | --- |
| Trainer forward+backward+optimizer (advantage) | `trainer.py:981-987` | 매 iteration 학습 | 메인 프로세스, 단일 스레드 |
| Trainer forward+backward+optimizer (strategy) | `trainer.py:1028-1034` | 매 iteration 학습 | 메인 프로세스, 단일 스레드 |
| Trainer device 결정 + eval 모듈 deepcopy | `trainer.py:862-870` | eval용 모듈 복제 | 메인 프로세스 |
| Imitation 학습 step | `imitation.py:87-89` | 옵션 path | 메인 프로세스 |
| Policy gradient 학습 step | `policy_gradient.py:69-71` | 옵션 path | 메인 프로세스 |
| Eval forward (batched, inference_mode) | `evaluate.py:211, 249` | eval phase | 메인 프로세스 |
| Inference server forward (batched, inference_mode) | `inference_server.py:231-249` | option A 워커 inference | **별도 spawn 프로세스** |
| Inference server weight load | `inference_server.py:61-67, 173-192` | trainer→server weight push | **별도 프로세스, 큐 직렬화** |
| Worker network reconstruct (CPU only today) | `workers.py:119, 141` | per-worker state_dict 재생성 | 별도 fork/spawn 프로세스 |

핵심 관찰: **모든 GPU 쓰기 경로(optimizer.step, load_state_dict)는 오늘 단일
프로세스 내에서 단일 스레드에 의해 직렬로 발생**한다. Multiprocessing이
"무료로" 보장해주던 격리다.

### Hypothetical threaded design (worst case에 가까운 단순 변환)

| 스레드 | 호출하는 CUDA 연산 | 모듈 공유? | 위험 |
| --- | --- | --- | --- |
| Trainer thread | advantage/strategy forward+backward+`optimizer.step()` | 본인이 own | weight write |
| N traversal threads | advantage/strategy `forward()` (`inference_mode`) | trainer와 share | read-during-write race |
| Eval thread (옵션) | strategy `forward()`, deepcopy | trainer와 share | deepcopy 중 write |
| Weight-sync (없어짐 — 같은 in-process 객체) | `load_state_dict` (만약 league snapshot용으로 남으면) | share | 같음 |

가장 무서운 조합: **trainer가 `optimizer.step()` 도중**(파라미터 텐서가
in-place로 부분 갱신되는 시점)에 traversal 스레드가 같은 파라미터에 대해
`forward()`를 돌리는 경우. PyTorch는 파라미터 텐서 write에 대한 user-side
동기화를 요구하므로(위 §3) **결과는 silent wrong outputs**다 — segfault도
없고 에러도 없고, 그냥 일부 파라미터가 step 전, 일부는 step 후 값으로 섞여
forward가 진행된다. CFR regret 추정 자체에 노이즈를 더해서 학습 발산/품질
저하로 나타난다.

`load_state_dict`도 동일하게 **부분 갱신 + 동시 read** 위험이다.
[forum #224131](https://discuss.pytorch.org/t/thread-safety-between-model-state-dict-and-optimizer-step/224131)
의 동일한 우려가 그대로 적용된다.

---

## 위험 표

| # | 항목 | 종류 | 현실성 (우리 코드) | 증상 |
| --- | --- | --- | --- | --- |
| R1 | 공유 모듈에 대한 forward vs optimizer.step race | silent wrong outputs | **매우 높음** | 학습 noisy, 발산 가능. crash 없음. |
| R2 | 공유 모듈 forward vs `load_state_dict` race | silent wrong outputs | 높음 (league snapshot 갱신 시) | 부분 weight forward, sample contamination |
| R3 | 모든 스레드 default stream → GPU 직렬화 | perf collapse | 매우 높음 (default 동작) | 멀티스레드인데 GPU utilization 그대로 |
| R4 | Allocator lock contention (작은 텐서, 고빈도) | perf collapse | 중간 | trace상 `cudaMalloc`/free 락 대기 증가 |
| R5 | 두 스레드가 공유 autograd graph 일부에 backward | crash / wrong grad | 낮음 (우리는 backward를 trainer만 호출) | graph free race → segfault |
| R6 | `torch.compile`된 모듈 + 멀티스레드 캐시 일관성 | wrong output / 재컴파일 폭주 | 낮음 (현재 main에 compile 없음) | unexpected recompile, dispatcher TLS 충돌 |
| R7 | Cython 확장 `nogil` 미감사 | crash / heap 손상 | 매우 높음 (free-threaded 한정) | segfault, 재현 불가 버그 |
| R8 | CUDA context 상호작용 (단일 context는 OK, 그래도 set_device 누락 시 wrong device) | wrong device error | 낮음 | runtime error |
| R9 | Pinned-memory / `non_blocking=True` H2D 카피 + 다른 스레드의 source tensor mutate | data race on source | 중간 | non-deterministic input bytes ([forum #182924](https://discuss.pytorch.org/t/is-it-safe-to-use-tensor-cuda-non-blocking-true-in-a-thread/182924)) |
| R10 | TLS 누수: `inference_mode`/`no_grad`가 trainer 스레드에서 켜져 있는데 backward 호출 | 잘못된 grad 누락 | 낮음 (코드 명시적으로 with-block 사용) | grad가 안 나와서 학습 정지 |

가장 위험한 건 R1, R2, R3, R7. 이 넷은 "그냥 옮기기"의 직접 결과다.

---

## Mitigations

### M1. Per-thread module copy (제일 안전, 메모리 비용)

각 traversal 스레드가 모듈 deepcopy를 들고 있고, weight-sync 시점에만
`load_state_dict`로 갱신. trainer는 자기 모듈만 만진다.

- 비용: 모듈 사이즈 × N_threads VRAM. 우리 모델(512×3)은 작아서 무시 가능.
- 동기화 지점: weight-sync 때 traversal 스레드를 **잠시 quiesce**해야 안전.
  현재 multiprocessing 패턴(`weight_sync_event`)이 그대로 매핑됨.
- 결과: R1, R2 제거. R5도 자동 제거.

### M2. Reader-writer lock on the shared module

Trainer write(step/load_state_dict)는 writer lock, traversal forward는 reader
lock. Pythonic하게는 `threading.RLock` + writer가 모든 reader 종료 대기.

- 비용: writer가 풀릴 때까지 모든 traversal 스레드 정지 → tail latency 증가.
  CFR처럼 "조금 stale한 weight도 OK" 알고리즘에선 OK.
- 결과: R1, R2 제거. M1 대비 메모리 절약 / latency 손해.

### M3. Per-thread CUDA stream

각 스레드가 `s = torch.cuda.Stream(); with torch.cuda.stream(s):`로 forward를
감싼다. trainer도 본인 stream에서 step. weight-sync 시 `torch.cuda.synchronize()`
또는 stream event로 ordering 보장.

- 비용: 거의 없음 (stream 객체는 cheap). 코드 변경은 호출 사이트 추가.
- 결과: R3 제거 — 진짜 GPU 동시성 가능. **R1/R2는 해결하지 못함**(stream은
  ordering이지 mutual exclusion이 아니다). 반드시 M1 또는 M2와 같이 써야 한다.

### M4. Serialize at boundary (가장 단순, 거의 multiprocessing 효과)

Trainer forward/backward/step 전체를 큰 lock으로 감싸고, traversal forward도
같은 lock으로 감싼다. = 사실상 GIL 흉내.

- 비용: 멀티스레드 의미 사라짐. GPU 사용률 = single thread.
- 결과: 모든 race 제거되지만 free-threaded Python으로 갈 이유가 없어짐.
  *threading 도입 자체의 가치가 사라지는 시그널.*

### M5. 현재 inference-server 패턴을 프로세스 → 스레드로 단순 치환

현 `inference_server.py`는 1 spawn process + 워커가 큐로 RequestMessage 송신.
이를 1 server thread + N requester threads로 바꾸는 건 가장 minimal한 변경.

- 모든 GPU 호출이 server thread 1개로 모이므로 R1/R2/R5 자동 회피.
- IPC 비용 사라짐(shared memory가 그냥 메모리). performance.md "Option A
  Bench Result"의 IPC 오버헤드(~수백 μs/call) 제거가 가능.
- 단점: server thread가 여전히 single GPU executor → R3와 같은 GPU 직렬화는
  유지되나, 그게 *batching의 목표*이기 때문에 해롭지 않음. realized batch
  size가 ceiling 8(현재) → free-threaded로 64+ 스레드면 ceiling 64로 올라감 →
  performance.md가 기대했던 bs=64 regime에 진입.

이 시나리오에선 **GPU 호출은 여전히 1 스레드만 한다**. 멀티스레드 → CUDA의
복잡도 대부분이 사라진다. *권고 핵심.*

### M6. (보조) `torch.compile` 모듈을 공유 forward 대상에서 제외

향후 trainer가 compile을 다시 쓴다면 inference 스레드들에 노출하지 말 것
([dev-discuss: compile + multithreading](https://dev-discuss.pytorch.org/t/impact-of-multithreading-and-local-caching-on-torch-compile/2498)
의 캐시 일관성 이슈 회피).

---

## 권고

### 지금 위험 수준

코드 그대로(공유 모듈 + 동시 forward + 동시 step) threading 전환하면:

- R1, R2가 거의 확실히 발생 → 학습 품질에 silent regression.
- R3 때문에 GPU 활용도는 multiprocessing 대비 거의 개선 없음.
- Cython 코드(R7) 때문에 free-threaded 빌드에서 segfault 위험.

따라서 "**그냥 threading으로 옮기기**"는 **추천하지 않음**.

### 안전한 경로 (선호 순)

1. **M5 (server-thread pattern) + 기존 multiprocessing 워커 유지**.
   가장 작은 변경으로 GPU 호출 스레드를 1개로 묶고, IPC 비용만 줄인다.
   하지만 이건 threading으로의 *전환*이 아니라 "Option A의 in-process variant"
   라는 점 명심.

2. **Free-threaded Python으로 가야 한다면**: 그 결정은
   - (a) Cython 코드의 `nogil` 감사를 끝내고
   - (b) 적어도 traversal worker가 thread가 되어 game logic을 동시에 굴리고
   - (c) GPU 호출은 M5 패턴으로 단일 server-thread에 위임
   세 조건이 동시에 충족될 때만 가치가 있다. performance.md "Free-threaded
   Python 노트"의 결론(현재 미적용)과 일치.

3. **공유 모듈을 어쩔 수 없이 여러 스레드에서 호출해야 한다면** M1
   (per-thread copy, 모델이 작으니 비용 미미) + M3 (per-thread stream) 조합이
   안전. 단순히 lock(M2/M4)만 걸면 GPU 활용도는 single-thread와 동일해진다.

### 안전 검증 체크리스트 (전환 전)

- [ ] Module 공유 여부를 모든 forward 호출 사이트에 대해 표로 만들기.
      "이 forward는 trainer가 weight write하는 모듈인가?"가 yes면 M1/M2 필수.
- [ ] 각 GPU-호출 스레드가 `torch.cuda.set_device` 명시.
- [ ] 각 GPU-호출 스레드가 자신의 `torch.cuda.Stream` 보유 + `with` 감싸기.
- [ ] Weight push 경로에서 reader fence (모든 inflight forward 완료 대기) 보장.
      현 `weight_sync_event`와 동일한 의미를 in-process로 구현.
- [ ] `torch.compile`된 모듈은 공유 forward 대상에서 제외.
- [ ] Cython 확장이 free-threaded 빌드에서 동작/safe함을 회귀 테스트로 확인.
- [ ] `CUDA_LAUNCH_BLOCKING=1`로 한 번 돌려 silent wrong-output을 동기 에러로
      변환해보고 race 미존재 확인.
- [ ] 결정성 테스트: 동일 seed로 multiprocessing 버전과 threading 버전의
      iteration 1 forward outputs bit-exact 비교 (allocator/stream 비결정 제외).
- [ ] 부하 테스트: 동시 forward+step을 10⁵ 회 돌려 weight checksum 변화가
      예상 범위 내인지 (R1 검출).

### 한 줄 정리

**현재 모델 사이즈에서는 multiprocessing → threading의 위험·복잡도 vs 이득
비율이 나쁘다.** 정말 단일 프로세스가 필요하다면 *모든* GPU 연산을 한
"server thread"로 모으는 M5 형태가 거의 모든 위험을 회피한다 — 그리고 그건
이미 우리가 가진 inference_server.py 구조의 thread 버전일 뿐이다.

---

## 참고 자료

- [PyTorch CUDA semantics docs](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [c10/cuda/CUDAStream.h — current stream is thread-local](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAStream.h)
- [c10/cuda/CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)
- [zdevito: A guide to PyTorch's CUDA caching allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)
- [pytorch#25540 — per-thread default stream feature request, unresolved](https://github.com/pytorch/pytorch/issues/25540)
- [pytorch#59692 — streams sequentially serialized](https://github.com/pytorch/pytorch/issues/59692)
- [pytorch#101300 — default stream not synchronous](https://github.com/pytorch/pytorch/issues/101300)
- [pytorch#15210 — torch::jit::script::Module not thread-safe](https://github.com/pytorch/pytorch/issues/15210)
- [pytorch#19029 — C++ custom module not thread safe](https://github.com/pytorch/pytorch/issues/19029)
- [pytorch#51452 — JIT module forward thread safety](https://github.com/pytorch/pytorch/issues/51452)
- [pytorch#130249 — Python 3.13t free-threaded support tracking](https://github.com/pytorch/pytorch/issues/130249)
- [forum: Is inference thread-safe?](https://discuss.pytorch.org/t/is-inference-thread-safe/88583)
- [forum: Is PyTorch supposed to be thread-safe?](https://discuss.pytorch.org/t/is-pytorch-supposed-to-be-thread-safe/36540)
- [forum: Only 1 thread for backward?](https://discuss.pytorch.org/t/only-1-thread-for-backward/36824)
- [forum: state_dict vs optimizer.step thread safety](https://discuss.pytorch.org/t/thread-safety-between-model-state-dict-and-optimizer-step/224131)
- [forum: Tensor.cuda(non_blocking=True) in a thread](https://discuss.pytorch.org/t/is-it-safe-to-use-tensor-cuda-non-blocking-true-in-a-thread/182924)
- [forum: Threaded inference c10::CuDNNError](https://discuss.pytorch.org/t/threaded-inference-c10-cudnnerror/182191)
- [dev-discuss: torch.compile + multithreading caches](https://dev-discuss.pytorch.org/t/impact-of-multithreading-and-local-caching-on-torch-compile/2498)
- [Trent Nelson — PyTorch and free-threading](https://trent.me/articles/pytorch-and-python-free-threading/)
- 본 저장소: `docs/performance.md` "Option A Bench Result and Structural
  Ceiling", "Free-threaded Python (3.13t) 노트"
- 본 저장소: `src/coolrl_lost_cities/games/classic/deep_cfr/inference_server.py`,
  `trainer.py`, `workers.py`, `evaluate.py`
