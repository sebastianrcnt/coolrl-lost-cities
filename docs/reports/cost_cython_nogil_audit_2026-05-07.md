# Cython `nogil`-cleanliness Audit

Date: 2026-05-07
Scope: `game.pyx`, `traversal.pyx`, `encoding.pyx`, `cfr_math.pyx` (+ matching `.pxd`)
Trigger: free-threaded Python (3.13t) + threaded traversal as an alternative
to multiprocessing for Deep CFR. See
`docs/performance.md` "Option A Bench Result and Structural Ceiling" and
"Free-threaded Python (3.13t) note".

---

## 한 줄 결론

**Medium effort.** 게임 엔진(`game.pyx`)과 보조 산술(`cfr_math.pyx`,
`encoding.pyx` core encode 함수)은 이미 거의 nogil-clean이다. 진짜
blocker는 한 곳에 모여 있다: **`traversal.pyx`의 `_traverse` 재귀 본체가
PyTorch forward, NumPy 배열 할당, Python `TraversalStats`/`TrainingSample`
객체 mutation, f-string, Python list/dict bucket 누적**을 모두 직접
한다는 것. 이걸 두 단계로 분리(순수 C 시뮬레이션 + Python에서 후처리)하지
않으면 `with nogil:` 블록을 의미 있게 키울 수 없다. 비용은 traversal 한
파일의 mid-scale 리팩터(추정 1–2주, 회귀 위험 큼) + 작은 게임 엔진 정리
(1–2일)이다.

---

## 파일별 현황

### 1. `game.pyx` / `game.pxd` — **거의 nogil-clean (small effort)**

게임 상태 데이터는 모두 raw `int*` (C 힙)에 살고, 핵심 동작(`_apply_*_c`,
`_undo_*_c`, `_legal_actions_c`, `_can_play_encoded_card_c`,
`_score_from_summary_c`, `_recompute_score_caches`,
`_swap_deck_cards_c`, `_push_action_c`, `_pop_action_c`)이 전부
`cdef ... noexcept`/`except *` 시그니처에 C 산술만 한다.

이미 nogil-callable 후보 (시그니처에 `nogil` 키워드만 추가하면 되는 것):

- `_is_legal_action_c` (game.pyx:869)
- `_legal_actions_c` (game.pyx:896)
- `_unified_legal_actions_c` (game.pyx:924)
- `_can_play_encoded_card_c` (game.pyx:953)
- `_fill_undo_c` (game.pyx:964)
- `_score_from_summary_c` (game.pyx:1267)
- `_has_any_legal_draw` (game.pyx:1281)
- `_hand_index` / `_expedition_len_index` / `_expedition_index` /
  `_discard_index` / `_encode_card` / `_card_color` / `_card_rank`
  (game.pyx:1293–1312)
- `_recompute_score_caches` (game.pyx:1229)

부분 GIL 필요 (작은 수정으로 nogil 가능):

- `_apply_action_unchecked_c` / `_apply_card_action` / `_apply_draw_action`
  (game.pyx:1008, 1099, 1140) — 본문은 순수 C이지만 `except *`라
  exception propagation을 위해 GIL이 필요. nogil 컨텍스트에서 호출하려면
  `noexcept` 또는 명시적 `nogil` + `with gil` 예외 블록이 필요. 본문에는
  실제로 raise할 곳이 없으므로 시그니처를 `noexcept`로 바꾸는 게 가장 싸다.
  단, `_apply_card_action`/`_apply_draw_action`은 invariant를 깨는 입력이
  들어와도 silently 진행하게 되므로 호출 전 검증을 강화해야 한다.
- `_apply_action_with_undo_c`, `_push_action_c`, `_pop_action_c`,
  `_swap_deck_cards_c`, `_ensure_undo_capacity_c` (game.pyx:1004–1055) —
  `_ensure_undo_capacity_c`만 `realloc` 실패 시 `MemoryError`를 raise.
  `with gil:` 짧은 블록으로 분리하거나, traversal 진입 시
  capacity를 미리 키워두면 nogil-clean하게 만들 수 있다.
- `_undo_*_c` (game.pyx:1161, 1169, 1203) — `raise ValueError("undo
  ... mismatch")` 가드가 들어 있음. Production 경로에서는 fire되지 않으므로
  guard를 `assert` 또는 디버그 빌드 한정으로 빼면 nogil 가능.

완전 GIL 함수 (nogil 변환 비대상; 호출자가 GIL 가진 채로 부른다):

- `__init__`, `_configure`, `from_snapshot`, `to_snapshot`, `validate_invariants`
  (Python config object/dict touch, dataclass, `Counter`, yaml, etc.)
- 모든 property: `phase`, `deck`, `hands`, `expeditions`, `discards` —
  Python list-of-Card 생성. 전부 reporting/serialization용이라 hot path
  아님.
- `clone()` (game.pyx:550) — `GameState(self.config)` 생성자 호출이
  Python object instantiation. nogil 안에서 부르려면 별도 `cdef
  GameState _clone_into(self, GameState dst) nogil` 같은 C-only fast clone을
  추가해야 한다 (전부 `memcpy`이므로 trivial하지만 새 entry point 필요).
- `to_unified_action`, `hand_slots`, `sort_hand` 등 Python 인터페이스 —
  hot path 아님.

요약: game.pyx는 **시그니처 정리 + 작은 helper 추가**로 hot path를 통째로
`nogil` 안에 넣을 수 있다. 게임 엔진 자체는 큰 비용이 아니다.

### 2. `cfr_math.pyx` / `cfr_math.pxd` — **이미 nogil-clean (zero effort)**

`regret_matching_c`, `normalize_legal_policy_c`, `sample_policy_c` 모두
`noexcept` + 순수 C 산술. `nogil` 키워드만 시그니처에 추가하면 끝.

(file:1–89). 파이썬 wrapper 3개(`regret_matching` 등, file:92–146)는
NumPy 인터페이스라 GIL 필요하지만 hot path 아님 — traversal은 이미 C
함수 직접 호출 (traversal.pyx:11, `from ... cimport regret_matching_c`).

### 3. `encoding.pyx` / `encoding.pxd` — **거의 nogil-clean (small effort)**

C-only encoders (전부 `noexcept`/`except -1`, raw float buffer 출력):

- `_base_input_dim_c`, `input_dim_c`, `_input_dim_with_flags_c`,
  `_numeric_value_c`, `_max_numeric_sum_c`, `_max_score_estimate_c`
  (encoding.pyx:12–57)
- `_color_playability_summary_c` (encoding.pyx:60–) — 본문은 순수 C
  (state의 C 필드 참조 + `abs()`), nogil 가능.
- `_append_derived_playability_features_c`,
  `_append_slot_aware_playability_features_c` (encoding.pyx:156, 209) —
  본문은 순수 C 산술. nogil 가능.
- `encode_info_state_c`, `_encode_info_state_with_flags_c`
  (encoding.pyx:287–413) — `except -1`로 Python `ValueError`를 raise할 수
  있는 두 곳(file:319, 321)이 있지만 둘 다 정적 sanity 체크
  (`player < 0`, `action_size > 64`). 호출 전에 검증되면 제거해도 안전.

`abs(state.expedition_penalty)` (encoding.pyx:53, 95): Cython이 `int`에
대해 C `abs`로 lower하므로 nogil-safe. `bool(encoding.derived_playability)`
(file:420, 432) 같은 건 Python wrapper에서만 호출되므로 무관.

요약: `_encode_info_state_with_flags_c`를 `noexcept nogil`로 바꾸고
input validation을 호출자로 옮기면 nogil-clean. 변환 매우 쉬움.

### 4. `traversal.pyx` / `traversal.pxd` — ****진짜 blocker가 모두 여기 있다 (medium-large effort)****

이미 nogil-callable인 helper들:

- `_next_u32`, `_next_double` (traversal.pyx:24, 29) — `noexcept`,
  raw uint32 LCG. 사실상 nogil이지만 키워드 빠짐.
- `_sample_policy_from_actions_c` (traversal.pyx:33) — `noexcept`,
  raw pointer.
- `_sampling_policy` (traversal.pyx:582) — `noexcept`, raw pointer.
- `_from_unified_action_c`, `_to_unified_action_c` (traversal.pyx:992, 997)
- `_opened_color_count` (traversal.pyx:766)
- `_self_play_bucket` (traversal.pyx:774) — `noexcept`이지만 `len(self.
  league_advantage_networks)`를 본다 → Python list `__len__` (PyObject_Size).
  이건 GIL 필요. 단순한 fix: 별도 `cdef int _league_size`를 캐싱.
- `_depth_bucket_start` (traversal.pyx:58)
- `random_rollout_value_c` (traversal.pyx:1092) — 본문은 순수 C이지만
  `_push_action_c`, `_pop_action_c`, `_legal_actions_c`가 nogil이 되면
  자동으로 nogil-callable. raise 두 줄(file:1106, 1108)을 호출 전 검증으로
  옮기면 끝.

반면 hot path인 `_traverse` (traversal.pyx:259) 본문에는 다음과 같은
Python-object touch가 깔려 있다 (per-node, per-iteration):

1. **PyTorch forward 호출** — `_policy_from_networks` (file:439) /
   `_policy_from_strategy_network` (file:519). NumPy `np.empty`,
   `torch.as_tensor`, `networks[player](x)`, `.detach().cpu().numpy().
   astype(np.float32)`. 이게 모든 `_policy` 호출(노드당 1회)에서 일어남.
2. **`stats` mutation** — 모든 카운터 증가가 Python attr 접근:
   `stats.nodes += 1`, `stats.terminals += 1`, `stats.max_depth_reached`,
   `stats.regret_fallback_*` 등 (file:289, 290, 298, 302, 386, 695–752).
3. **f-string + dict bucket** — `_record_endpoint` (file:980), `_record_
   fallback_depth_bucket` (file:753): `f"{start}_{start + width - 1}"`,
   `stats.endpoint_depth_buckets[key] = ... .get(key, 0) + 1`. Python
   string format + dict lookup.
4. **NumPy 배열 할당 per leaf** — `_record_strategy` (file:877), `_record_
   advantage` (file:914), `_record_external_advantage` (file:948): `np.empty(
   self.action_size, dtype=np.float32)`, `.append(TrainingSample(...))`.
   Sample마다 두 개의 작은 NumPy array + dataclass 인스턴스화.
5. **Python list `.append`** — `self.advantage_samples.append(...)`,
   `self.strategy_samples.append(...)` (file:903, 937, 969). list의
   PyObject reference 갱신은 free-threaded Python에서도 atomic refcount
   비용을 추가로 부담한다.
6. **`HeuristicBot.act(state)`** — `_fixed_opponent_action`
   (file:633, 652), `_rollout_value` (file:841). Python class
   메서드 호출. `heuristic_balanced` 옵션 사용 시만 핫.
7. **`league_advantage_networks` indexing** — `_self_play_snapshot_
   networks` (file:802), `[-recent_count:]`, `[:max(0, ...)]` slicing
   = Python list slicing.
8. **`f"invalid ..."` raises** — game state 검증 실패 시.

`_traverse`는 game state mutation(전부 C struct 통한
`_push_action_c`/`_pop_action_c`)과 위 Python object 작업을 한 함수에서
교차해서 한다. 즉 `with nogil:`로 감쌀 수 있는 자연스러운 chunk가
없다 — recursion 한 단계 안에서 GIL을 ~6번 release/re-acquire해야
하는데, 그 비용이 forward latency보다 크다.

---

## 주요 blocker 카탈로그

### B1. PyTorch forward 호출 (가장 큰 단일 blocker)

```python
# traversal.pyx:473-475
with torch.inference_mode():
    x = torch.as_tensor(info_state, dtype=torch.float32, device=self.device).unsqueeze(0)
    advantages = networks[player](x).squeeze(0).detach().cpu().numpy().astype(np.float32)
```

- 빈도: 노드당 1회 (~205k/iter, performance.md 참조).
- 변환 난이도: **High (구조 변경 필수)**. 핵심 통찰은: **이걸 nogil 만들
  필요 없다.** PyTorch CUDA 호출 자체가 internally GIL을 잠깐 잡지만
  `inference_mode` + CUDA dispatch는 잘 알려진 GIL-friendly 영역이다.
  진짜 문제는 *traversal recursion이 forward 호출에서 sync-block*해서
  배치가 안 모이는 것 (performance.md "Option B-shape refactor"). nogil로
  단일 thread를 빠르게 만들기보다 **traversal을 resumable state machine
  으로 깨고 N개 thread를 띄워 동시에 sync-block시키면**, free-threaded
  Python 하에서 batch=N forward로 자연 합쳐진다. 즉 nogil-cleaning은
  Option B/C와 같은 작업의 일부이지 독립 작업이 아니다.
- 권고: 이 blocker는 nogil audit 단독으로 고치지 말고, "traversal을
  state-machine으로 해체" 작업 안에 묶는다.

### B2. Python `TraversalStats` attribute mutation (전 노드 핫)

```python
# traversal.pyx:289-294
stats.nodes += 1
if depth > stats.max_depth_reached:
    stats.max_depth_reached = depth
if self.has_max_nodes and stats.nodes >= self.max_nodes:
    stats.node_limit_cutoffs += 1
```

- 빈도: 매 노드. 합쳐서 노드당 5–15회 attr access.
- 변환 난이도: **Low–Medium**. `TraversalStats`를 `cdef class`로 바꾸고
  필드를 `cdef public long long`로 선언하면 attr access가 C struct field
  store가 된다. 단, `stats.regret_fallback_depth_buckets` 같은 dict
  필드는 별도로 처리(아래 B3).
- 위치: traversal.pyx:289, 290, 293, 298, 302, 331, 386, 695–752, 853, 856,
  912, 946, 978, 985–990 + `_record_*` 전체.

### B3. dict bucket + f-string key (depth/color buckets)

```python
# traversal.pyx:986-990, 753-764, 702-705, 723-725, 742-744
key = f"{start}_{start + width - 1}"
stats.endpoint_depth_buckets[key] = stats.endpoint_depth_buckets.get(key, 0) + 1
```

- 빈도: 매 leaf/cutoff/regret-fallback 노드.
- 변환 난이도: **Medium**. dict + str key + format은 nogil 불가. 해법:
  - bucket 인덱스로 미리 정해진 정수 array를 쓴다 (`endpoint_depth_bucket_max
    / endpoint_depth_bucket_width + 1` slot의 `cdef long[:]` 또는 raw
    int64 array). string key는 마지막 reporting 단계에서만 생성.
  - color/opened_color bucket도 모두 5–6개 정해진 슬롯이므로 `cdef
    long[5]`로 충분.

### B4. NumPy 배열 + dataclass 인스턴스 per training sample

```python
# traversal.pyx:896-911, 926-945, 959-977
target = np.empty(self.action_size, dtype=np.float32)
legal_mask = np.empty(self.action_size, dtype=np.bool_)
...
self.advantage_samples.append(TrainingSample(info_state=..., target=..., ...))
```

- 빈도: leaf마다 1개 advantage sample + 노드별 strategy sample
  (interval-gated).
- 변환 난이도: **Medium-High**. 두 가지 옵션:
  - **(a) Buffer pre-allocate**: traverser가 큰 `cdef float[:, ::1]
    advantage_targets`, `cdef uint8[:, ::1] advantage_legal`,
    `cdef long[:] advantage_iteration` 등을 미리 잡아두고 row index만 늘린다.
    drain 시점에 `TrainingSample` Python 객체로 wrap. **추천**.
  - **(b)** PyObject 그대로 두고 `with gil:` 짧게 — sample 누적이
    노드당 ~1회라 IPC overhead 분석 그대로 적용된다 (작은 hold라도 thread
    contention 발생).
- 추가 고려: `info_state`(`np.empty(input_dim, dtype=np.float32)`)도 노드당
  새 NumPy. buffer-pool 또는 batched encoder로 묶어야 한다.

### B5. PyTorch `state_dict()`-share, league list slicing

```python
# traversal.pyx:802-817
candidates = self.league_advantage_networks[-recent_count:]
...
candidates = self.league_advantage_networks[:max(0, len(self.league_advantage_networks) - recent_count)]
```

- 빈도: traversal 진입 시 한 번 (`traverse`에서 미리 픽), 재귀 안에서는
  `active_self_play_networks`만 본다. 따라서 cold path. 변환 불필요.

### B6. `HeuristicBot.act(state)` — Python bot

```python
# traversal.pyx:633, 652, 841
return int(self.heuristic_opponent_bot.act(state))
```

- 빈도: `opponent_policy=heuristic_balanced` 또는 `cutoff_rollout_policy=
  heuristic_balanced`일 때만. 현 default는 self_play_league + score_diff
  cutoff (per memory의 opponent_policy_network_divergence note + AGENTS).
- 변환 난이도: **Medium-High** (Python class 전체를 cython화). 현 default
  config에서는 핫 아님 — 시도하지 않는 게 합리.

### B7. `len(self.league_advantage_networks)`

```python
# traversal.pyx:782, 783, 803, 806, 811, 814
recent_count = min(len(self.league_advantage_networks), self.self_play_recent_window)
```

- 빈도: `_self_play_bucket`가 traversal 진입에 한 번, `_self_play_snapshot_
  networks`가 한 번. 노드당이 아님 → cold path. 무시 가능 (단, 두 함수가
  `_traverse` 안에서 직접 불리지 않음을 확인했음, file:244–251).

### B8. `_apply_action_unchecked_c` `except *`

게임 엔진 쪽 game.pyx:1008. 본문에 raise 없음 → `noexcept`로 강등하면
`_traverse`의 `state._push_action_c` (game.pyx:1029, `except *`) 호출도
`noexcept`로 만들 수 있다. 단, `_ensure_undo_capacity_c`의 `MemoryError`만
별도 처리 필요.

### B9. Recursion이 그 자체로 `_traverse` (cdef method `except *`)

`_traverse`는 `cdef float ... except *` (file:259). nogil로 만들려면
재귀 호출도 nogil 컨텍스트여야 하고, 모든 파이썬 touch가 제거되어야 한다.
즉 **B1–B4가 전부 해결되기 전엔 `_traverse` 본체를 `nogil`로 못 만든다.**

---

## 작업 단계 (안전한 순서)

1. **단계 0 — 측정 인프라.** Cython annotate (`cython -a`)를 빌드 스크립트에
   추가. `.html`에서 노란/빨간 줄 = Python interaction. 반복적으로 본다.
2. **단계 1 — 무비용 청소 (1–2일):**
   - `cfr_math.pyx`의 3개 C 함수에 `nogil` 키워드 추가.
   - `encoding.pyx`의 `_encode_info_state_with_flags_c`와 모든 helper의
     검증을 호출자로 옮기고 `noexcept nogil`로.
   - `game.pyx`의 `_legal_actions_c`, `_unified_legal_actions_c`,
     `_can_play_encoded_card_c`, `_score_from_summary_c`,
     `_has_any_legal_draw`, 모든 `_*_index`/`_card_*` 함수에 `nogil` 추가.
   - 회귀 테스트: `uv run pytest -q`.
3. **단계 2 — 게임 엔진 mutation을 nogil로 (2–3일):**
   - `_apply_card_action`, `_apply_draw_action`, `_apply_action_unchecked_c`
     를 `noexcept`로 강등 (호출 전 legality check가 이미 `_traverse`에서
     수행되므로 안전).
   - `_undo_*_c`의 `ValueError` mismatch 가드를 debug 빌드 한정 (`IF
     DEBUG:` 컴파일 디렉티브 또는 release 시 제거).
   - `_ensure_undo_capacity_c`: traversal 진입 시점에 한 번 큰 capacity로
     `realloc`해두고, hot path의 `_push_action_c`는 capacity 체크만
     (`assert undo_stack_len < undo_stack_capacity` debug only)하게 분리.
   - 결과: `_push_action_c`/`_pop_action_c`/`_swap_deck_cards_c` 모두
     `nogil`.
4. **단계 3 — TraversalStats를 cdef class로 (3–5일):**
   - 모든 정수 카운터를 `cdef public long long` 필드로.
   - depth bucket / color bucket dict들을 fixed-size `cdef long[N]` array로
     교체하고 reporting 단계에서만 dict로 변환.
   - `_record_endpoint`, `_record_fallback_depth_bucket`,
     `_record_regret_matching_decision`을 `noexcept nogil`로 다시 작성.
   - 회귀 테스트: `metrics.jsonl`의 모든 키가 동일한 값으로 나오는지 비교.
5. **단계 4 — Sample buffer pre-allocate (3–5일):**
   - traverser에 `cdef float[:, ::1] advantage_targets`,
     `cdef uint8[:, ::1] advantage_legal_masks`,
     `cdef float[:, ::1] advantage_info_states`,
     `cdef long[:] advantage_iterations`, `cdef int[:] advantage_players`
     등을 chunk-grow array로. row index만 nogil에서 늘림.
   - `drain_samples()`에서만 GIL 잡고 `TrainingSample` 리스트로 wrap.
   - 회귀 테스트: trainer가 받는 sample 분포 동일해야 함.
6. **단계 5 — Forward 호출 분리 (large, 다른 작업과 묶음):**
   - `_traverse`를 "forward 직전까지" + "forward 결과 받은 후" 두 구간의
     resumable state machine으로 재구성. forward 호출은 외부 batcher가
     수행. 이게 Option B/C 본체이므로 별도 design doc 필요.
   - 그제서야 `_traverse` 자체를 `nogil`로 선언할 의미가 생긴다.
7. **단계 6 — 검증:**
   - `cython -a`로 hot path가 모두 흰색인지 시각 확인.
   - micro-bench: 단일 thread에서 traversal 시간이 회귀 없는지.
   - free-threaded Python (`uv run --python python3.13t ...`) 또는
     `nogil`-제어 micro-bench로 N=2/4/8 thread scaling 확인.

---

## 위험

- **Silent slowdown (GIL re-acquisition)**: `with nogil:` 블록 안에서
  Python 객체를 무심코 건드리면 Cython이 `with gil:` 블록을 자동 삽입
  (또는 `noexcept nogil` 위반 시 컴파일 에러). 작은 attr touch 하나가
  re-acquisition 비용을 부르고, 멀티스레드에선 contention으로 single-thread
  대비 더 느려질 수 있다. 검증: `cython -a`가 진실의 원천. 모든 hot 경로가
  흰색이어야 함. 추가로 `python -X dev`나 `PYTHONDEVMODE=1`로 thread state
  체크.
- **Correctness regression on undo path**: 단계 2의 `_undo_*` 가드 제거가
  invariant를 silently 위반시킬 수 있음. 검증: `tests/games/classic/test_
  deep_cfr_trainer.py` + `validate_invariants()`를 `--set debug=true` 같은
  모드에서 매 100노드마다 호출.
- **Sample buffer overflow**: 단계 4의 chunk-grow가 race 없는지
  (single-traverser-per-thread 구조 유지) 확인. 두 thread가 같은 traverser
  객체를 공유하면 안 됨.
- **`TraversalStats` API 변경**: `metrics.jsonl` 형식 변경 가능성. 단계
  3에서 reporting 어댑터를 명시적으로 보존. 기존 dict 형식과 byte-wise
  동일한 테스트 추가.
- **Cython `nogil` + cdef class 라이프타임**: `cdef class` 인스턴스의
  refcount는 free-threaded Python에서 atomic이지만 deallocation이 nogil
  컨텍스트 안에서 트리거되면 안 됨. 모든 cdef object는 함수 시작에 GIL
  잡힌 채로 acquire, nogil 블록 안에서는 raw pointer/struct만 접근.
- **CUDA forward thread-safety**: PyTorch는 같은 device 위 동시
  forward에 대해 internal lock을 사용한다. N=64 thread가 동시에 forward를
  치면 합쳐주지 않으면 lock contention만 늘 수 있다. 단계 6의 batcher가
  필수.

---

## 권고

**현 시점에는 단계 1–3까지만 기회 봐서 진행하고, 단계 4 이상은 보류.**

이유:

1. 단계 1–3은 **나중 단계와 무관하게 단일-thread traversal도 살짝 빠르게**
   만들고, `cython -a`상의 visible Python interaction을 줄여 다음 작업의
   기반이 된다. 비용 작음(~1주), 회귀 위험 낮음(테스트 충분).
2. 단계 4부터는 free-threaded Python이나 Option B/C 같은 호출자 측 변경이
   같이 와야 의미가 있다. 현재 `default.yaml`은 single-process local
   backend로 잘 돌고 있고(performance.md), 모델 크기·eval 비중·python3.13t
   생태계 모두 트리거가 안 와 있음.
3. 단계 5/6는 Option B-shape refactor와 사실상 같은 작업이므로 **별도
   설계 문서가 먼저** 필요하다. nogil audit이 그걸 정당화하는 근거는
   되지만 단독 추진 사유는 안 된다.

**다시 볼 트리거** (둘 중 하나라도 만족):

- (a) **Free-threaded Python (3.13t)이 mainstream** 으로 가서 PyTorch 공식
  지원이 stable이 되고, `uv`가 3.13t를 1차 시민으로 다룬다.
- (b) **Model이 커진다** — hidden=1024 / depth=6 등으로 forward가
  단일 호출 ~수백 μs 영역에 들어가서, traversal 한 번에 한 forward를 GIL
  잡고 부르는 게 명백히 bottleneck이 된다.
- (c) **Eval 비중이 dominant**해진다 (`eval_every=5`, `evaluation.games=
  1000+`). Eval은 이미 batch-friendly이라 thread pool + nogil game engine만
  으로도 큰 win.

위 세 가지가 모두 멀어 보일 때(현 상황)는 단계 1–3만 chip away 하고,
설계 측면에서는 Option B-shape (per-worker interleaved traversal) 쪽이
ROI가 더 높다 (performance.md "Re-enable A when one of these holds" 참조).

### 빠른 우선순위 1순위 (지금 당장 1일)

`cython -a` 빌드 옵션 추가 + cfr_math와 encoding hot path에 `nogil`
키워드만 다는 것. 이건 아무것도 안 깨고 tooling 인프라가 생긴다.
다음 nogil 작업할 때 진단 출발점이 됨.

---

## 참조

- `docs/performance.md`:571 (Option A Bench Result)
- `docs/performance.md`:672 (Free-threaded Python note)
- `src/coolrl_lost_cities/games/classic/game.pxd`
- `src/coolrl_lost_cities/games/classic/game.pyx`:550, 869, 924, 953, 1004,
  1099, 1140, 1161, 1229, 1281, 1293
- `src/coolrl_lost_cities/games/classic/deep_cfr/traversal.pyx`:259 (`_traverse`),
  439 (`_policy_from_networks`), 519 (`_policy_from_strategy_network`),
  582 (`_sampling_policy`), 753 (`_record_fallback_depth_bucket`),
  877 (`_record_strategy`), 914 (`_record_advantage`),
  980 (`_record_endpoint`), 1092 (`random_rollout_value_c`)
- `src/coolrl_lost_cities/games/classic/deep_cfr/encoding.pyx`:287, 291,
  416, 425
- `src/coolrl_lost_cities/games/classic/deep_cfr/cfr_math.pyx`:5, 37, 68
