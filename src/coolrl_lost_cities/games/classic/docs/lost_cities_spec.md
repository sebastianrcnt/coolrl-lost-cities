# Lost Cities — 게임 핵심 로직 명세서

본 문서는 외부 에이전트가 Lost Cities를 플레이하기 위해 알아야 할 모든 규칙과 상태 전이를 기술한다. 구현(Rust core)과 인터페이스(gRPC proto)는 이 명세를 준수해야 한다.

---

## 1. 개요

Lost Cities는 2인 불완전정보 카드게임이다. 각 플레이어는 자신의 손패로 색깔별 탐험대(expedition)를 꾸리고, 게임 종료 시점의 총점으로 승부를 가린다.

- 플레이어 수: 정확히 2 (인덱스 0과 1)
- 상태 종류: 완전히 관찰 가능(공개) + 부분 관찰 가능(자기 손패만)
- 결정성: config의 seed가 주어지면 초기 덱 셔플이 결정적

---

## 2. 구성 요소

### 2.1 Config

게임은 다음 파라미터로 완전히 기술된다.

| 필드 | 의미 | 기본값 예시 |
|---|---|---|
| `n_colors` | 색깔(탐험대) 수 | 5 |
| `n_ranks` | 색깔당 숫자 카드 종류 수 | 9 |
| `min_rank` | 숫자 카드의 최소 face value | 2 |
| `n_handshakes` | 색깔당 handshake 카드 수 | 3 |
| `hand_size` | 각 플레이어의 손패 크기 | 8 |
| `expedition_penalty` | 탐험대 시작 비용 (음수) | -20 |
| `bonus_threshold` | 보너스 자격 카드 수 | 8 |
| `bonus_amount` | 보너스 점수 | 20 |
| `seed` | 덱 셔플 시드 (선택) | — |

제약: 모든 수치는 양수여야 하며, `deck_size >= 2 * hand_size`여야 한다.

### 2.2 Card

각 카드는 `(color, rank)` 쌍으로 식별된다.

- `color`: 0부터 `n_colors - 1`까지의 정수
- `rank`:
  - `0`: handshake (투자) 카드
  - `1`부터 `n_ranks`까지: 숫자 카드
- 숫자 카드의 face value: `min_rank + rank - 1`
  - 예: `min_rank=2, rank=1` 이면 face value는 2
  - 예: `min_rank=2, n_ranks=9` 이면 face value 범위는 2부터 10
- handshake의 face value는 정의상 0 (점수 계산에만 사용)

### 2.3 Deck

덱 구성: 각 색깔마다 `n_handshakes`장의 handshake와 rank 1부터 `n_ranks`까지 숫자 카드 각 1장씩.

총 덱 크기: `n_colors * (n_handshakes + n_ranks)`

---

## 3. 초기화

1. 덱을 config의 seed로 셔플한다.
2. 각 플레이어에게 `hand_size`만큼 카드를 딜한다. 딜 순서는 플레이어를 번갈아가며 1장씩.
3. 각 플레이어의 손패는 항상 정렬 상태로 유지된다 (key: color 오름차순, 같은 color 내에서 rank 오름차순).
4. 각 플레이어의 expedition은 색깔마다 빈 stack으로 초기화된다.
5. 각 색깔의 discard 더미는 빈 상태로 시작한다.
6. `current_player = 0`, `phase = CARD`, `pending_discarded_color = None`, `turn_count = 0`, `terminal = False`.

---

## 4. 턴 구조

한 턴은 정확히 두 phase로 구성된다.

1. **CARD phase** — 손패에서 카드 한 장을 선택해 플레이(play)하거나 버린다(discard).
2. **DRAW phase** — 덱 맨 위 또는 비어 있지 않은 discard 더미 중 하나에서 카드를 손패로 받는다.

DRAW phase가 끝나면 `current_player`가 교체되고 `phase = CARD`로 돌아가며 `turn_count`가 1 증가한다.

단, 아래 6장(종료 조건)에 해당하면 턴 교체 없이 게임이 종료된다.

---

## 5. 액션

### 5.1 CARD phase의 합법 액션

손패의 각 슬롯 `i`에 대해 다음 두 액션이 평가된다.

**PLAY_CARD(slot=i)**

- 대상 카드 `c = hand[i]`
- 해당 색깔의 expedition에서 "마지막 숫자 랭크"를 `last_numeric`이라 하자. expedition에 숫자 카드가 없으면 `last_numeric = 0`.
- `c`가 handshake이면: `last_numeric == 0`일 때만 합법
- `c`가 숫자 카드이면: `c.rank > last_numeric`일 때만 합법
- 합법이면 `c`를 해당 색깔 expedition stack에 push

**DISCARD_CARD(slot=i)**

- 항상 합법
- `c`를 `c.color`의 discard 더미 맨 위에 push
- `pending_discarded_color = c.color`로 기록

CARD phase에서는 DISCARD가 항상 가능하므로 합법 액션이 최소 1개 존재한다.

### 5.2 DRAW phase의 합법 액션

**DRAW_DECK**

- `len(deck) > 0`일 때만 합법
- 덱 맨 위 카드를 손패에 추가한 뒤 손패를 재정렬

**DRAW_DISCARD(color=k)**

- `len(discards[k]) > 0`이고 `k != pending_discarded_color`일 때만 합법
- `discards[k]` 맨 위 카드를 손패에 추가한 뒤 손패를 재정렬

### 5.3 pending_discarded_color

- CARD phase에서 `DISCARD_CARD`로 색깔 `k`를 버렸다면, 같은 턴의 DRAW phase에서 `DRAW_DISCARD(k)`는 금지된다.
- CARD phase에서 `PLAY_CARD`를 했다면 `pending_discarded_color`는 `None`.
- DRAW phase가 끝나면 `pending_discarded_color`는 다음 턴으로 넘어가기 전에 `None`으로 초기화된다.

---

## 6. 종료 조건

게임은 다음 중 하나가 성립하는 즉시 종료된다.

1. DRAW phase에서 카드를 받은 직후 `len(deck) == 0`이 되는 경우. 이 경우 턴 교체 없이 종료.
2. 드물게: DRAW phase에서 합법 액션이 전혀 없는 경우 (덱과 모든 discard가 동시에 사용 불가). 극단적 config에서만 발생.

종료 시 `terminal = True`가 되고 이후 어떤 액션도 적용할 수 없다.

---

## 7. 점수 계산

각 플레이어의 각 색깔 expedition에 대해:

- 카드가 하나도 없으면: 0점
- 카드가 있으면:
  - `numeric_sum` = expedition 내 숫자 카드의 face value 합
  - `handshakes` = expedition 내 handshake 카드 수
  - `score = (numeric_sum + expedition_penalty) * (handshakes + 1)`
  - `len(expedition) >= bonus_threshold`이면 `score += bonus_amount`

플레이어의 총점: 모든 색깔 expedition 점수의 합.

`score_diff(p) = total_score(p) - total_score(1 - p)`

---

## 8. 불변식

구현은 아래 불변식을 모든 상태 전이에서 유지해야 한다.

1. **카드 보존**: `len(deck) + Σ|hand[p]| + Σ|expedition[p][c]| + Σ|discard[c]|` 는 항상 `deck_size`와 같다.
2. **expedition monotonicity**: 각 expedition stack 내 숫자 카드는 bottom→top 방향으로 strictly increasing rank.
3. **handshake 위치**: expedition에 숫자 카드가 하나라도 들어가면 이후로 해당 expedition에 handshake를 더 추가할 수 없다.
4. **phase-scoped invariant**: `pending_discarded_color`는 `phase == DRAW`에서만 의미를 가진다.
5. **terminal 단조성**: `terminal`은 한 번 `True`가 되면 `False`로 되돌아가지 않는다.
6. **legal action 존재**: `terminal == False`이면 합법 액션 집합이 비어 있지 않다. `terminal == True`이면 공집합이다.
7. **손패 정렬**: 손패는 항상 `(color, rank)` 오름차순 정렬 상태.

---

## 9. 외부 에이전트를 위한 계약

gRPC 인터페이스로 플레이하는 에이전트가 알아야 할 계약.

### 9.1 상태는 세션 단위로 서버가 유지

클라이언트는 `session_id`로 게임을 식별한다. 한 세션 = 한 게임.

### 9.2 observation은 항상 특정 플레이어 관점

`observer_player`가 보는 정보:

- 자신의 손패 전체
- 자신과 상대의 expedition 전체 (공개 정보)
- 모든 discard 더미 전체 (공개 정보)
- 상대 손패 크기 (내용은 비공개)
- 남은 덱 크기 (내용은 비공개)

### 9.3 action id는 observation scoped

`Action.id`는 그 observation과 함께 반환된 것에 한해서만 유효하다. 상태가 한 번이라도 진행되면 이전 observation의 id는 재사용 불가.

이를 강제하기 위해 모든 observation에는 `state_version`(세션 내 단조증가)이 실리고, `ApplyAction`은 `expected_state_version`을 함께 받아 mismatch 시 거절한다.

### 9.4 합법 액션은 observation에 임베드되어 옴

별도 RPC 호출 없이 `observation.legal_actions` 안에서 전체 리스트와 마스크가 모두 제공된다. 외부 에이전트는 이 리스트에서 `id`만 골라 `ApplyAction`으로 되돌려주면 된다.

### 9.5 보상

`ApplyAction`의 `reward`는 observer 관점이며, terminal 전이에서만 nonzero이고 값은 observer의 `score_diff`. 비terminal 전이의 reward는 0.

---

## 10. 결정성과 재현

동일한 `(config, seed, action_sequence)`는 동일한 상태 전이와 최종 점수를 재생성한다. 로그/리플레이는 이 튜플을 저장하는 것으로 충분하다.
