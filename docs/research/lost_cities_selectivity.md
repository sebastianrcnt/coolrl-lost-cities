# Lost Cities Deep CFR — 다음 실험 아이디어 노트

## Recap

slot_aware_playability iter 240까지 진행. 광의 selectivity 학습됨 (score per opened color -10 → -7, baseline +14 우위). 협의 selectivity 미해결: opened_colors 4.95+, bad_open_rate 88-92% plateau, calibration gap 6-9 → 2-4 감소. 외부 모델 4개 자문 받음. 진단 다 다르게 옴, 다 mechanism level 합당. **네 번째 모델이 가장 깊은 layer 짚음** — Deep CFR 구현 디테일 (current vs average policy, all-negative fallback, reservoir memory inertia) 측면에서 우리가 본 적 없는 진단들. 코드 수정 (league bucket sampling, AdamW) 진행 중.

## 진단 가설들 (서로 충돌, 중첩 가능)

**Variance 가설 (Gemini)**
Draw variance ±40 vs entry regret ~5 = SNR 0.125. Variance가 entry decision regret signal 압도. Calibration gap 감소 = "variance가 prediction 자체를 무용하게 만듦". 1순위 개입: reward smoothing.

**Architecture / Action representation 가설 (두 번째, 세 번째, 네 번째 모델)**
Flat MLP가 action-conditional advantage 분리 못 함. "Play X" vs "Discard X"가 같은 weights 사용. Selectivity는 action property인데 state encoder 위 flat output으로는 표현 못 함. 1순위 개입: action-factorized scorer 또는 open-gate.

**"Open and recover" 가설 (두 번째 모델)**
정책이 entry filter 학습 대신 "다 열고 수습" 평형으로 수렴. score per opened color 개선이 그 증거. 수습 능력 강해질수록 entry 학습 동기 줄어듦.

**Action-local credit assignment 가설 (세 번째 모델)**
"Bad open이 진짜 bad action인가" 자체 의심. final own expedition negative ≠ counterfactual EV negative. 우리 bad_open_rate metric이 Deep CFR objective와 mismatch 가능. "Skip color" action 부재로 single regret accumulator 없어서 부정 신호가 여러 play-card action에 분산.

**Average strategy / League inertia 가설 (네 번째 모델, 신규)**
Deep CFR은 최종적으로 average strategy가 수렴 대상. current advantage policy는 selectivity 학습 중인데 average strategy / league inertia가 끌고 있을 가능성. 우리가 측정한 metric은 어떤 정책 기준인지 불분명. league `current 0.5 / recent 0.3 / older 0.2`에서 older가 초기 5-open convention 보존 가능.

**All-negative fallback 가설 (네 번째 모델, 신규)**
정보집합에서 모든 predicted regret ≤ 0이면 uniform fallback. Lost Cities에서 uniform이 open_new를 자주 선택. Deep CFR 논문 ablation에서도 fallback 처리가 큰 영향이라고 보고. 우리 코드에서 이게 어떻게 작동하는지 미확인.

**Calibration ceiling 가설 (부분, 여러 모델)**
Visible-score predictor의 ceiling은 있을 수 있음. 다만 selectivity 자체의 ceiling은 아님. selectivity는 visible-score prediction 말고 option value, irreversible cost 회피, opponent dynamics 대응 등 다른 경로로도 emerge 가능.

**Self-play attractor 가설**
5-color가 stable equilibrium. 한쪽이 selectivity 시도하면 즉시 손해. safe_heuristic이 3.7에서 강하면 진짜 NE는 아닐 듯. 다만 self-play 안에서는 stable.

**Lost Cities NE 자체가 5-color 가설**
이론적 가능성 0 아님. self-play가 발견한 게 진짜 NE면 transition 영원히 안 옴. 검증 가능한 형태로는 tabular oracle 또는 BR 진단.

## 개입 아이디어 dump

### Diagnostic (가장 먼저 — 진단 cost 낮음, 정보량 큼)

**Current / Average / League policy 분리 측정 (네 번째 모델)**: opened_colors를 세 정책 (current advantage + RM, average strategy net, league mixture) 별로 따로 측정. 셋 다 5.0이면 정책 자체 stuck, current만 4.0이면 inertia 문제. 즉시 진단 가능.

**All-regrets ≤ 0 fallback 빈도 측정 (네 번째 모델)**: 정보집합 중 fallback 발동 비율, 그때 action_type 분포, 그 결정만으로 opened_colors 측정. open_new 비율 높으면 cheap fix 후보.

**Empirical r̃ by action class (네 번째 모델)**: advantage memory의 target을 action class별로 분류. open_new_safe / open_new_bad / continue / discard / draw로 나눠서 평균과 분위수. `r̃(open_bad) < 0인데 V predicts ≥ 0`이면 representation 문제. `r̃(open_bad) ≥ 0`이면 bad_open metric mismatch.

**BR-to-current 진단 (세 번째, 네 번째 모델)**: frozen current 상대로 pure payoff BR 학습. BR이 selective면 all-open exploitable, BR도 5-open이면 zero-sum objective상 자연스러운 전략.

**Counterfactual open EV audit (세 번째 모델)**: first-open 정보집합에서 force open vs force best non-open 비교. Δ_open 분포 분석. `Δ_open < 0`인데 policy opens면 학습 실패.

**Tabular Lost Cities oracle (네 번째 모델, 우아함)**: 2 colors × 6 ranks 축소판으로 tabular CFR 또는 external-sampling MCCFR. tabular도 5-open이면 게임 자체 5색 선호 가능성, selective면 Deep CFR approximation 문제 확정. Pure self-play 철학 안 깸.

### Architectural

**Open-gate (세 번째, 네 번째 모델)**: legal action set 안 바꾸고 latent에서 skip color 표현. `score[a_first_open_c] = base_score(a) + open_gate(state, color_c)`. 부정 신호가 한 곳에 모임. Action-factorized보다 cheap.

**Hierarchical entry gate target (네 번째 모델 구체화)**: open-gate를 traversal counterfactual value로 학습:
```
target_entry_adv(I,c) = max_a opens_c r̃(I,a) - max_b not_open_c r̃(I,b)
또는: logsumexp over open_c - logsumexp over non_open_c
```
heuristic label 아님. self-play traversal value에서 직접 나옴.

**Action-factorized scorer (full)**: state encoder + per-action feature → score(s, a). 1-2일.

**Color permutation equivariance / augmentation (네 번째 모델)**: 5색 대칭 활용. SharedColorMLP 또는 학습 시 색 순서 permute augmentation. 거의 무료. selectivity 기준 한 번만 배움.

**Slot-action shared encoder**: SharedSlotMLP로 hand slot embedding.

**Dueling head**: A(s,a) = Q(s,a) - mean(Q(s, legal_actions)). Action 간 차이 강조.

**Two-headed MLP**: final layer를 play/discard로 split. capacity 동일. 1-2시간.

**Capacity scaling**: 256x3 → 512x3. 다른 변수와 entangle 위험.

### Training dynamics

**LCFR / DCFR weighting (네 번째 모델)**: 과거 regret discount, late iteration 더 강하게 반영. 초기 5-open snapshot이 memory와 league에 오래 남는 문제 직접 해결. Pure self-play 유지.

**All-negative fallback 수정 (네 번째 모델)**: uniform → argmax regret + tie random. 한 줄 수정. 가장 cheap. discard tie-break는 heuristic 냄새라 자제 권고.

**League weight 조정 (네 번째 모델)**: older weight 0.2 → 0.05~0.1. 초기 convention 영향 감소.

**Burn-in 제외**: 처음 100~200 iter average memory에 약하게 반영. 초기 high-entropy phase 영향 감소.

**Reward smoothing (Gemini)**: `U = sign(Δ)·sqrt(|Δ|)` 또는 `tanh(Δ/20)`. Variance 압축. 5-10분. zero-sum 보존. (네 번째 모델은 권고 안 함, 게임 변경 우려)

**V(S) baseline (state-level aux)**: state value 예측 head. State-level이라 action property와 mismatch 가능.

**Open-action regret sign auxiliary (세 번째 모델)**: first-open action에서 `y = 1[A_open(I,a) > 0]` 예측. Counterfactual regret 자체를 target으로.

**aux_open_adv / aux_delay_value / aux_entry_bucket (네 번째 모델)**: final sign 말고 action-regret 기반 auxiliary. λ 0.03~0.10 작게. 후반 anneal down. shared trunk까지만 gradient 허용 ablation.

**First-open replay reweighting (세 번째 모델)**: minibatch에서 first-open 정보집합 oversample. Pure self-play 안 깸.

**Type-balanced RM epsilon (세 번째 모델)**: action-uniform 대신 type-uniform exploration. 단 이건 exact CFR에서 멀어질 수 있어 fallback/weighting/architecture 먼저.

**DCFR style discount**: regret/strategy memory 시간 가중. Reservoir buffer와 상호작용 검토 필요.

**Memory discounting**: 오래된 all-open snapshot 영향 감소.

### Game-specific inductive bias

**Hypergeometric features (세 번째 모델)**: `P(reach_break_even | public info, random unseen draw)`. recoverable_score보다 직접적.

**Discard alternative quality features**: open이 좋은가가 아니라 "open이 지금 가장 덜 나쁜가" 평가용.

**Option value features**: 미래 draw 수, public dead cards, deck phase pressure 등.

**Risk-sensitive objective**: zero-sum 보존이지만 게임 변경. 후순위.

**Action abstraction (skip color action)**: 게임 자체 변경. 큰 작업. (세 번째 모델이 비추, latent gate가 더 안전)

### Framework level (마지막 옵션)

**PSRO-style population**: strategy portfolio + learned response → empirical game equilibrium. Heuristic anchor 없이 population diversity. Deep CFR 내부 수정이 아니라 훈련 틀 변경.

## 검증 측정 항목

### 기존 metric

- Score per opened color 변동성
- Calibration gap 추세
- Opened_colors 분산
- Bad_open_rate vs good_open_rate 갈라짐
- 4-color vs 5-color frequency

### 새 metric

**Calibration metric 자체 변경 (세 번째 모델)**:
```
기존: E[first_open_recoverable | positive final] - E[... | negative final]
추천: E[Δ_open | policy opens] - E[Δ_open | policy doesn't open]
```
Deep CFR objective와 align.

**Policy 분리 측정 (네 번째 모델)**:
- opened_colors (current advantage policy)
- opened_colors (average strategy net)
- opened_colors (league mixture)

**Fallback 측정 (네 번째 모델)**:
- % infosets where all legal regrets ≤ 0
- among fallback: P(action_type=open_new), P(continue), P(discard)
- avg_opened_colors from fallback decisions only

**Empirical advantage by class (네 번째 모델)**:
- empirical r̃(open_new_safe), r̃(open_new_bad), r̃(continue_safe), r̃(continue_unsafe), r̃(discard_safe), r̃(discard_dangerous), r̃(draw_*)
- 평균과 분위수
- network V vs empirical r̃ 비교

**Action-level metrics**:
- First-open action regret sign accuracy (또는 AUC)
- Open-vs-best-discard margin 분포
- bad_open_EV_rate = 비율(Δ_open < 0)
- Deck phase별 open threshold

### 필요 샘플 수 수학 (네 번째 모델)

```
calibration gap = d, terminal variance = σ²
sign 안정 인식 위해 N ≳ σ²/d²

d 작거나 σ 크면 N 폭발
→ "더 돌리기"의 한계 명확
```
Phase transition은 iter 수가 아니라 "음의 entry regret이 분리되어 누적되는가"에 달림.

## 네 모델 비교 요약

| 항목 | Gemini | 두 번째 | 세 번째 | 네 번째 |
|------|--------|--------|---------|---------|
| Dominant obstacle | Variance × attractor | Architecture | Action-local credit + skip 부재 | **Avg/league inertia + fallback + architecture** |
| Calibration ceiling | Fundamental | "Open and recover" | Predictor ceiling이지 selectivity ceiling 아님 | 부분 동의, predictor ceiling 한정 |
| 1순위 개입 | Reward smoothing | Action-factorized | Open-gate + replay reweight + type-balanced eps | **Diagnostic 먼저, 그다음 fallback fix + LCFR/DCFR + action-factorized** |
| Current vs avg policy 분리 | | | | **있음** |
| Fallback 진단 | | | | **있음** |
| Empirical r̃ by class | | | | **있음** |
| Tabular oracle | | | | **있음** |
| Entry gate target 정의 | | | (언급) | **수학적 정의** |
| Color permutation aug | | (언급) | | **있음** |
| BR 진단 | | | 있음 | 있음 |
| Bad_open metric 의심 | | | 있음 | 있음 |
| Sample complexity 수학 | | | | **있음** |
| Phase transition | 가능 | 거의 불가 | 거의 불가 | 거의 불가, mechanism 정밀 |
| 이론 reference | | | Zinkevich, Brown | **Zinkevich, Brown, MCCFR, DCFR, PSRO** |

## 주관적 인상 / 메모

- 네 모델 다 valuable하지만 깊이가 다름. Gemini < 두 번째 < 세 번째 < 네 번째 순으로 mechanism level 깊어짐.
- 네 번째 모델의 가장 큰 신규 기여: **diagnostic 먼저 하라**는 권고. 우리가 4번 실험 동안 한 번도 안 본 측정들 (current vs avg policy, fallback, empirical r̃ by class). 이거 측정만 해도 나머지 가설들 중 어느 게 맞는지 즉시 판별.
- "Current vs Average vs League policy 분리" — 우리 분석의 진짜 blind spot. 우리는 한 정책의 행동만 봤음. Deep CFR은 average strategy가 수렴 대상인데.
- All-negative fallback 진단 — Deep CFR 논문 ablation에서 큰 영향이라고 보고된 거. 우리 코드에서 어떻게 작동하는지 한 번도 안 본 듯. 진단 후 한 줄 수정으로 큰 변화 가능성.
- Tabular oracle 실험 — 가장 우아함. "Lost Cities 구조에서 selectivity가 자연스러운가" 질문에 데이터로 답함. Pure self-play 철학 안 깸.
- Entry gate target의 수학적 정의 (네 번째 모델) — 세 번째 모델 open-gate 권고의 빈 부분 채움. heuristic label 아니라 traversal counterfactual value 직접 사용.
- "Open and recover" (두 번째) + "Skip color 부재" (세 번째) + "Avg/league inertia" (네 번째) 통합: 정책이 entry filter 학습 대신 수습 학습 가는 mechanism이 (1) skip action 없어서 부정 신호 분산 + (2) average policy가 inertia로 5-open 보존 + (3) function approximation이 action 분리 못 함의 복합.
- Calibration metric 자체를 Δ_open 기반으로 바꾸는 게 진단 정확도 올림. 기존 visible_score 기반은 Deep CFR objective와 mismatch.
- "5000 iter 더 돌리기"는 네 모델 다 회의적. 네 번째 모델이 sample complexity 수학으로 가장 정확히 설명. expectation 낮춰야.
- Reward smoothing은 Gemini만 권고. 다른 모델들은 회의적이거나 비권고. Variance가 진짜 dominant인지 검증 필요.
- 다른 모델들이 architecture 1순위로 갔는데 네 번째 모델은 **diagnostic 1순위, weighting 2순위, architecture 3순위** 순서 권고. 진단 안 하고 architecture 가는 건 비싼 도박. 진단으로 어느 가설이 맞는지 먼저 보는 게 ablation 청결성과 효율성 모두 높음.
- Pure self-play 철학과 self-play trajectory label aux loss는 양립 (두, 세, 네 번째 모델 다 권고).
- "Discard tie-break" 같은 selectivity 방향 휴리스틱은 자제 권고 (네 번째 델). pure self-play 철학 보존.
