module SafeHeuristic

export SafeHeuristicParams, safe_heuristic_action

const PLAY_OR_DISCARD_ACTIONS_PER_SLOT = 2
const DRAW_FROM_DECK_ACTION = 0

Base.@kwdef struct SafeHeuristicParams
    open_target_ratio::Float64 = 0.50
    open_min_card_ratio::Float64 = 0.40
    handshake_target_multiplier::Float64 = 1.15
    handshake_min_card_ratio::Float64 = 0.34
    late_deck_ratio::Float64 = 0.20
    mid_deck_ratio::Float64 = 0.35
    commitment_weight::Float64 = 1.00
    gift_penalty_weight::Float64 = 1.00
    discard_safety_bonus::Float64 = 6.00
    unusable_discard_bonus::Float64 = 20.00
    deck_draw_early_value::Float64 = 2.00
    deck_draw_mid_value::Float64 = 1.00
    deck_draw_late_value::Float64 = -1.00
    deny_opponent_weight::Float64 = 0.40
    winning_deck_bonus::Float64 = 0.75
    losing_deck_penalty::Float64 = 1.25
    losing_visible_draw_bonus::Float64 = 1.50
    speculative_visible_draw_bonus::Float64 = 1.50
    dead_visible_draw_penalty::Float64 = 2.00
    unopened_draw_penalty_three_open::Float64 = 10.00
    unopened_draw_penalty_four_open::Float64 = 20.00
    strong_deny_threshold::Float64 = 10.00
    late_open_block_ratio::Float64 = 0.20
    low_card_sequence_bonus::Float64 = 5.00
    started_expedition_play_bonus::Float64 = 4.00
    started_expedition_followup_bonus::Float64 = 3.00
end

struct DerivedHeuristicConfig
    middle_rank::Int
    max_color_sum::Int
    break_even_sum::Int
    open_target_sum::Float64
    min_open_cards::Int
    min_handshake_numeric_cards::Int
    late_deck_threshold::Int
    mid_deck_threshold::Int
    late_open_block_threshold::Int
    bonus_possible::Bool
    max_expedition_cards::Int
end

play_action(slot0::Int)::Int = PLAY_OR_DISCARD_ACTIONS_PER_SLOT * slot0
discard_action(slot0::Int)::Int = PLAY_OR_DISCARD_ACTIONS_PER_SLOT * slot0 + 1
draw_from_discard_action(color::Int)::Int = 1 + color

card_color(card)::Int = Int(card["color"])
card_rank(card)::Int = Int(card["rank"])
is_handshake(card)::Bool = card_rank(card) == 0
num(config, card)::Int = is_handshake(card) ? 0 : Int(config["min_rank"]) + card_rank(card) - 1
deck_size(config)::Int = Int(config["n_colors"]) * (Int(config["n_ranks"]) + Int(config["n_handshakes"]))
max_rank(config)::Int = Int(config["min_rank"]) + Int(config["n_ranks"]) - 1
card_action_size(config)::Int = 2 * Int(config["hand_size"])
draw_action_size(config)::Int = 1 + Int(config["n_colors"])

function params_for_variant(name::AbstractString)::SafeHeuristicParams
    if name == "loose"
        return SafeHeuristicParams(
            open_target_ratio=0.42,
            open_min_card_ratio=0.30,
            handshake_target_multiplier=1.00,
            handshake_min_card_ratio=0.25,
            late_open_block_ratio=0.12,
        )
    elseif name == "strict"
        return SafeHeuristicParams(
            open_target_ratio=0.62,
            open_min_card_ratio=0.50,
            handshake_target_multiplier=1.35,
            handshake_min_card_ratio=0.45,
            late_open_block_ratio=0.30,
        )
    end
    return SafeHeuristicParams()
end

function derive(config, params::SafeHeuristicParams)::DerivedHeuristicConfig
    n_ranks = Int(config["n_ranks"])
    min_rank = Int(config["min_rank"])
    hand_size = Int(config["hand_size"])
    total_deck_size = deck_size(config)
    n_handshakes = Int(config["n_handshakes"])
    bonus_threshold = Int(config["bonus_threshold"])
    max_color_sum = sum(min_rank + rank - 1 for rank in 1:n_ranks)
    break_even_sum = -Int(config["expedition_penalty"])
    open_target_sum = min(0.8 * Float64(break_even_sum), params.open_target_ratio * Float64(max_color_sum))
    min_open_cards = max(1, min(hand_size, round(Int, hand_size * params.open_min_card_ratio)))
    min_handshake_numeric_cards = max(
        1,
        min(hand_size, round(Int, hand_size * params.handshake_min_card_ratio)),
    )
    late_deck_threshold = max(1, round(Int, total_deck_size * params.late_deck_ratio))
    mid_deck_threshold = max(late_deck_threshold + 1, round(Int, total_deck_size * params.mid_deck_ratio))
    late_open_block_threshold = max(1, round(Int, total_deck_size * params.late_open_block_ratio))
    max_expedition_cards = n_handshakes + n_ranks
    return DerivedHeuristicConfig(
        (n_ranks + 1) ÷ 2,
        max_color_sum,
        break_even_sum,
        open_target_sum,
        min_open_cards,
        min_handshake_numeric_cards,
        late_deck_threshold,
        mid_deck_threshold,
        late_open_block_threshold,
        max_expedition_cards >= bonus_threshold,
        max_expedition_cards,
    )
end

function hand(state, player0::Int)
    return state["hands"][player0 + 1]
end

function expeditions(state, player0::Int)
    return state["expeditions"][player0 + 1]
end

function color_expedition(state, player0::Int, color0::Int)
    return state["expeditions"][player0 + 1][color0 + 1]
end

function discard_pile(state, color0::Int)
    return state["discards"][color0 + 1]
end

function last_numeric_rank(state, player0::Int, color0::Int)::Int
    last = 0
    for card in color_expedition(state, player0, color0)
        rank = card_rank(card)
        if rank > last
            last = rank
        end
    end
    return last
end

has_numeric(state, player0::Int, color0::Int)::Bool = last_numeric_rank(state, player0, color0) > 0

function can_play_card(state, player0::Int, card)::Bool
    rank = card_rank(card)
    color = card_color(card)
    config = state["config"]
    if color < 0 || color >= Int(config["n_colors"]) || rank < 0 || rank > Int(config["n_ranks"])
        return false
    end
    last_rank = last_numeric_rank(state, player0, color)
    if rank == 0
        return last_rank == 0
    end
    return rank > last_rank
end

function score_from_summary(config, len::Int, handshakes::Int, numeric_sum::Int)::Int
    if len == 0
        return 0
    end
    score = (numeric_sum + Int(config["expedition_penalty"])) * (handshakes + 1)
    if len >= Int(config["bonus_threshold"])
        score += Int(config["bonus_amount"])
    end
    return score
end

function total_score(state, player0::Int)::Int
    config = state["config"]
    total = 0
    for expedition in expeditions(state, player0)
        handshakes = 0
        numeric_sum = 0
        for card in expedition
            if is_handshake(card)
                handshakes += 1
            else
                numeric_sum += num(config, card)
            end
        end
        total += score_from_summary(config, length(expedition), handshakes, numeric_sum)
    end
    return total
end

score_diff(state, player0::Int)::Int = total_score(state, player0) - total_score(state, 1 - player0)

function legal_card_mask(state)::Vector{Bool}
    h = hand(state, Int(state["current_player"]))
    mask = fill(false, card_action_size(state["config"]))
    for (idx, card) in enumerate(h)
        slot0 = idx - 1
        if can_play_card(state, Int(state["current_player"]), card)
            mask[play_action(slot0) + 1] = true
        end
        mask[discard_action(slot0) + 1] = true
    end
    return mask
end

function legal_draw_mask(state)::Vector{Bool}
    config = state["config"]
    mask = fill(false, draw_action_size(config))
    pending = get(state, "pending_discarded_color", nothing)
    if length(state["deck"]) > 0
        mask[DRAW_FROM_DECK_ACTION + 1] = true
    end
    for color in 0:(Int(config["n_colors"]) - 1)
        if pending !== nothing && color == Int(pending)
            continue
        end
        if !isempty(discard_pile(state, color))
            mask[draw_from_discard_action(color) + 1] = true
        end
    end
    return mask
end

function first_legal(mask)::Int
    for (idx, value) in enumerate(mask)
        if value
            return idx - 1
        end
    end
    return 0
end

function best_pair(candidates)
    isempty(candidates) && return nothing
    best_value, best_action = candidates[1]
    for (value, action) in candidates[2:end]
        if value > best_value || (value == best_value && action > best_action)
            best_value = value
            best_action = action
        end
    end
    return best_action
end

function best_triple(candidates)
    isempty(candidates) && return nothing
    best = candidates[1]
    for item in candidates[2:end]
        if item[1] > best[1] || (item[1] == best[1] && (item[2] > best[2] || (item[2] == best[2] && item[3] > best[3])))
            best = item
        end
    end
    return best[3]
end

function late_penalty(derived::DerivedHeuristicConfig, deck_left::Int)::Float64
    if deck_left <= derived.late_deck_threshold
        return 15.0
    elseif deck_left <= derived.mid_deck_threshold
        return 8.0
    end
    return 0.0
end

function new_color_open_penalty(opened_colors::Int)::Float64
    opened_colors <= 1 && return 0.0
    opened_colors == 2 && return 6.0
    opened_colors == 3 && return 14.0
    return 28.0
end

function bonus_potential(state, player0::Int, color0::Int, extra_cards::Int, derived::DerivedHeuristicConfig; committed_cards::Int=0, exclude_slot::Union{Nothing,Int}=nothing)::Float64
    !derived.bonus_possible && return 0.0
    config = state["config"]
    expedition_len = length(color_expedition(state, player0, color0)) + committed_cards
    need = Int(config["bonus_threshold"]) - expedition_len
    need <= 0 && return Float64(config["bonus_amount"])
    playable_count = 0
    for (idx, card) in enumerate(hand(state, player0))
        if exclude_slot !== nothing && idx == exclude_slot
            continue
        end
        if card_color(card) == color0 && can_play_card(state, player0, card)
            playable_count += 1
        end
    end
    if playable_count + extra_cards >= need
        return 0.4 * Float64(config["bonus_amount"])
    end
    return 0.0
end

function opening_plan_value(state, params::SafeHeuristicParams, player0::Int, color0::Int, opening_card, derived::DerivedHeuristicConfig, deck_left::Int)::Float64
    config = state["config"]
    numbers = [card for card in hand(state, player0) if card_color(card) == color0 && !is_handshake(card) && card_rank(card) >= card_rank(opening_card)]
    handshakes = [card for card in hand(state, player0) if card_color(card) == color0 && is_handshake(card)]
    opened_colors = count(expedition -> !isempty(expedition), expeditions(state, player0))
    number_sum = sum(num(config, card) for card in numbers; init=0)
    high_cards = [card for card in numbers if card_rank(card) >= derived.middle_rank]
    high_count = length(high_cards)
    opening_value = num(config, opening_card)
    penalty = new_color_open_penalty(opened_colors)
    strong_open = length(numbers) >= derived.min_open_cards && number_sum >= derived.open_target_sum && (!isempty(high_cards) || number_sum >= 0.85 * derived.max_color_sum)
    speculative_open = opened_colors <= 2 && length(numbers) >= 2 && number_sum >= 0.65 * derived.open_target_sum && !isempty(high_cards)
    single_late_open = deck_left <= derived.mid_deck_threshold && length(numbers) >= 1 && opening_value >= 8
    exceptional_open = length(numbers) >= derived.min_open_cards + 1 && number_sum >= max(Float64(derived.break_even_sum), derived.open_target_sum * 1.4) && high_count >= 2 && deck_left > derived.mid_deck_threshold
    if opened_colors == 3
        speculative_open = false
    end
    if opened_colors >= 4
        strong_open = false
        speculative_open = false
        single_late_open = false
    end
    strong_open && return 6.0 + 0.25 * number_sum + 0.8 * length(numbers) + 0.5 * length(handshakes) - penalty
    speculative_open && return 3.0 + 0.18 * number_sum + 0.7 * length(numbers) + 0.4 * length(handshakes) - penalty
    opened_colors == 3 && return 0.0
    exceptional_open && return 10.0 + 0.3 * number_sum + 1.0 * length(numbers) + 0.7 * high_count - penalty
    single_late_open && return 1.5 + 0.2 * opening_value - penalty
    return 0.0
end

function color_commitment(state, params::SafeHeuristicParams, player0::Int, color0::Int, derived::DerivedHeuristicConfig)::Float64
    config = state["config"]
    expedition = color_expedition(state, player0, color0)
    value = isempty(expedition) ? 0.0 : 5.0
    for card in expedition
        value += is_handshake(card) ? 2.0 : 0.25 * num(config, card)
    end
    playable_cards = [card for card in hand(state, player0) if card_color(card) == color0 && can_play_card(state, player0, card)]
    playable_numbers = [card for card in playable_cards if !is_handshake(card)]
    playable_handshakes = [card for card in playable_cards if is_handshake(card)]
    value += 1.2 * length(playable_numbers)
    value += 1.5 * length(playable_handshakes)
    value += 0.15 * sum(num(config, card) for card in playable_numbers; init=0)
    value += 0.05 * bonus_potential(state, player0, color0, 0, derived)
    return value
end

function public_color_commitment_for_opponent(state, opponent0::Int, color0::Int, derived::DerivedHeuristicConfig)::Float64
    config = state["config"]
    expedition = color_expedition(state, opponent0, color0)
    discard = discard_pile(state, color0)
    value = isempty(expedition) ? 0.0 : 5.0
    handshake_count = count(is_handshake, expedition)
    value += 2.0 * handshake_count
    numeric_cards = [card for card in expedition if !is_handshake(card)]
    value += 0.25 * sum(num(config, card) for card in numeric_cards; init=0)
    if !isempty(numeric_cards)
        value += 0.4 * num(config, numeric_cards[end])
    end
    if !isempty(discard)
        top_card = discard[end]
        if can_play_card(state, opponent0, top_card)
            value += is_handshake(top_card) ? 1.5 : 1.0 + 0.1 * num(config, top_card)
        end
    end
    if derived.bonus_possible
        expedition_len = length(expedition)
        if expedition_len + 1 >= Int(config["bonus_threshold"])
            value += 0.2 * Float64(config["bonus_amount"])
        end
    end
    return value
end

function card_value_for_opponent(state, params::SafeHeuristicParams, opponent0::Int, card, derived::DerivedHeuristicConfig)::Float64
    !can_play_card(state, opponent0, card) && return 0.0
    interest = public_color_commitment_for_opponent(state, opponent0, card_color(card), derived)
    is_handshake(card) && return 8.0 + 1.5 * interest
    numeric_value = num(state["config"], card)
    return numeric_value * (0.4 + 0.25 * interest)
end

function card_value_for_me(state, params::SafeHeuristicParams, player0::Int, card, derived::DerivedHeuristicConfig)::Float64
    !can_play_card(state, player0, card) && return 0.0
    commitment = color_commitment(state, params, player0, card_color(card), derived)
    if is_handshake(card)
        return 7.0 + 1.2 * commitment
    end
    numeric_value = num(state["config"], card)
    value = 0.8 * numeric_value + params.commitment_weight * commitment
    if !isempty(color_expedition(state, player0, card_color(card)))
        value += params.started_expedition_play_bonus + params.started_expedition_followup_bonus
    end
    if commitment >= 6.0 && card_rank(card) <= derived.middle_rank
        value += params.low_card_sequence_bonus
    end
    return value
end

function started_expedition_play_value(state, params::SafeHeuristicParams, player0::Int, card, derived::DerivedHeuristicConfig, deck_left::Int)::Float64
    config = state["config"]
    color = card_color(card)
    expedition = color_expedition(state, player0, color)
    numeric_value = num(config, card)
    current_sum = sum(num(config, played) for played in expedition if !is_handshake(played); init=0)
    followups = [followup for followup in hand(state, player0) if followup !== card && card_color(followup) == color && !is_handshake(followup) && card_rank(followup) > card_rank(card)]
    projected_sum = current_sum + numeric_value + sum(num(config, followup) for followup in followups; init=0)
    value = params.started_expedition_play_bonus + params.started_expedition_followup_bonus
    value += Float64(max_rank(config) + 1 - numeric_value)
    if deck_left <= derived.late_deck_threshold
        value += 2.0 * numeric_value
    elseif deck_left <= derived.mid_deck_threshold
        value += 0.8 * numeric_value
    end
    if projected_sum < derived.open_target_sum
        value -= 6.0
    end
    handshakes = count(is_handshake, expedition)
    value += 3.0 * handshakes
    value += bonus_potential(state, player0, color, 0, derived, committed_cards=1)
    return value
end

function visible_number_can_help_open(state, params::SafeHeuristicParams, player0::Int, card, derived::DerivedHeuristicConfig)::Bool
    numbers = [other for other in hand(state, player0) if card_color(other) == card_color(card) && !is_handshake(other) && card_rank(other) >= card_rank(card)]
    push!(numbers, card)
    length(numbers) < derived.min_open_cards && return false
    number_sum = sum(num(state["config"], other) for other in numbers; init=0)
    number_sum < derived.open_target_sum && return false
    return any(card_rank(other) >= derived.middle_rank for other in numbers)
end

function visible_open_support_value(state, params::SafeHeuristicParams, player0::Int, card, derived::DerivedHeuristicConfig)::Float64
    color = card_color(card)
    opened_colors = count(expedition -> !isempty(expedition), expeditions(state, player0))
    same_color_numbers = [other for other in hand(state, player0) if card_color(other) == color && !is_handshake(other)]
    same_color_handshakes = [other for other in hand(state, player0) if card_color(other) == color && is_handshake(other)]
    future_numbers = [other for other in same_color_numbers if other !== card && card_rank(other) >= card_rank(card)]
    value = 0.8 * length(future_numbers) + 1.0 * length(same_color_handshakes)
    if card_rank(card) <= derived.middle_rank
        value += params.speculative_visible_draw_bonus
    end
    if visible_number_can_help_open(state, params, player0, card, derived)
        value += 4.0
    elseif opened_colors <= 2 && (!isempty(future_numbers) || !isempty(same_color_handshakes))
        value += params.speculative_visible_draw_bonus
    end
    if opened_colors <= 2
        value += 0.25 * opening_plan_value(state, params, player0, color, card, derived, length(state["deck"]))
    elseif opened_colors == 3
        value += 0.1 * max(0.0, opening_plan_value(state, params, player0, color, card, derived, length(state["deck"])))
    end
    return value
end

function visible_draw_value(state, params::SafeHeuristicParams, player0::Int, card, derived::DerivedHeuristicConfig)::Float64
    color = card_color(card)
    opponent0 = 1 - player0
    opened_colors = count(expedition -> !isempty(expedition), expeditions(state, player0))
    is_unopened_color = isempty(color_expedition(state, player0, color))
    commitment = color_commitment(state, params, player0, color, derived)
    opponent_value = card_value_for_opponent(state, params, opponent0, card, derived)
    diff = score_diff(state, player0)
    value = params.deny_opponent_weight * opponent_value
    diff <= 0 && (value += params.losing_visible_draw_bonus)
    exceptional_support = false
    if is_unopened_color
        opened_colors >= 4 ? (value -= params.unopened_draw_penalty_four_open) : opened_colors >= 3 && (value -= params.unopened_draw_penalty_three_open)
    end
    if is_handshake(card)
        has_numeric(state, player0, color) && return value - params.dead_visible_draw_penalty
        if isempty(color_expedition(state, player0, color))
            playable_numbers = [other for other in hand(state, player0) if card_color(other) == color && !is_handshake(other) && can_play_card(state, player0, other)]
            number_sum = sum(num(state["config"], other) for other in playable_numbers; init=0)
            required_sum = derived.open_target_sum * params.handshake_target_multiplier
            if length(playable_numbers) < derived.min_handshake_numeric_cards || number_sum < required_sum
                support = visible_open_support_value(state, params, player0, card, derived)
                exceptional_support = support >= 6.0
                if is_unopened_color && opened_colors >= 4 && !exceptional_support && opponent_value < params.strong_deny_threshold && diff > -15
                    return -8.0
                end
                return value + support - 0.5
            end
        end
        return value + 6.0 + commitment
    end
    immediate_playable = can_play_card(state, player0, card)
    if immediate_playable
        value += Float64(num(state["config"], card))
        value += 0.7 * commitment
        if !isempty(color_expedition(state, player0, color))
            value += 5.0
        else
            support = visible_open_support_value(state, params, player0, card, derived)
            exceptional_support = support >= 6.0
            value += support
        end
    else
        value -= params.dead_visible_draw_penalty
        if isempty(color_expedition(state, player0, color))
            support = visible_open_support_value(state, params, player0, card, derived)
            exceptional_support = support >= 6.0
            value += support
        end
    end
    if is_unopened_color && opened_colors >= 4 && !exceptional_support && opponent_value < params.strong_deny_threshold && diff > -15
        return -8.0
    end
    value += bonus_potential(state, player0, color, 1, derived)
    return value
end

function best_handshake_play(state, params::SafeHeuristicParams, player0::Int, legal, derived::DerivedHeuristicConfig, deck_left::Int)
    Int(state["config"]["n_handshakes"]) <= 0 && return nothing
    candidates = Tuple{Float64,Int}[]
    for (idx, card) in enumerate(hand(state, player0))
        slot0 = idx - 1
        action = play_action(slot0)
        (!legal[action + 1] || !is_handshake(card)) && continue
        color = card_color(card)
        expedition = color_expedition(state, player0, color)
        any(!is_handshake(played) for played in expedition) && continue
        playable_numbers = [other for (other_idx, other) in enumerate(hand(state, player0)) if other_idx != idx && card_color(other) == color && !is_handshake(other) && can_play_card(state, player0, other)]
        number_count = length(playable_numbers)
        number_sum = sum(num(state["config"], other) for other in playable_numbers; init=0)
        number_count < derived.min_handshake_numeric_cards && continue
        required_sum = derived.open_target_sum * params.handshake_target_multiplier
        number_sum < required_sum && continue
        deck_left <= derived.late_open_block_threshold && continue
        value = number_sum + 2.0 * number_count
        value += bonus_potential(state, player0, color, 0, derived, committed_cards=1, exclude_slot=idx)
        value -= late_penalty(derived, deck_left)
        push!(candidates, (value, action))
    end
    return best_pair(candidates)
end

function best_number_play(state, params::SafeHeuristicParams, player0::Int, legal, derived::DerivedHeuristicConfig, deck_left::Int)
    candidates = Tuple{Float64,Int}[]
    for (idx, card) in enumerate(hand(state, player0))
        slot0 = idx - 1
        action = play_action(slot0)
        (!legal[action + 1] || is_handshake(card)) && continue
        color = card_color(card)
        if !isempty(color_expedition(state, player0, color))
            push!(candidates, (started_expedition_play_value(state, params, player0, card, derived, deck_left), action))
        elseif deck_left > derived.late_open_block_threshold && opening_plan_value(state, params, player0, color, card, derived, deck_left) > 0.0
            numbers = [c for c in hand(state, player0) if card_color(c) == color && !is_handshake(c) && card_rank(c) >= card_rank(card)]
            handshakes = [c for c in hand(state, player0) if card_color(c) == color && is_handshake(c)]
            number_sum = sum(num(state["config"], c) for c in numbers; init=0)
            value = number_sum + 2.0 * length(numbers) + 1.5 * length(handshakes)
            value += opening_plan_value(state, params, player0, color, card, derived, deck_left)
            value += Float64(state["config"]["expedition_penalty"])
            value += max(0.0, Float64(derived.middle_rank - card_rank(card)))
            value += bonus_potential(state, player0, color, 0, derived, committed_cards=1, exclude_slot=idx)
            value -= late_penalty(derived, deck_left)
            push!(candidates, (value, action))
        end
    end
    return best_pair(candidates)
end

function best_forced_open(state, params::SafeHeuristicParams, player0::Int, legal, derived::DerivedHeuristicConfig, deck_left::Int)
    candidates = Tuple{Float64,Int}[]
    for (idx, card) in enumerate(hand(state, player0))
        slot0 = idx - 1
        action = play_action(slot0)
        (!legal[action + 1] || is_handshake(card)) && continue
        !isempty(color_expedition(state, player0, card_color(card))) && continue
        opening_value = opening_plan_value(state, params, player0, card_color(card), card, derived, deck_left)
        color_numbers = [other for other in hand(state, player0) if card_color(other) == card_color(card) && !is_handshake(other)]
        number_sum = sum(num(state["config"], other) for other in color_numbers; init=0)
        if opening_value <= 0.0 && length(color_numbers) < 2 && number_sum < 0.5 * derived.open_target_sum && deck_left > derived.mid_deck_threshold
            continue
        end
        forced_value = opening_value + 0.2 * number_sum + Float64(max_rank(state["config"]) + 1 - num(state["config"], card))
        push!(candidates, (forced_value, action))
    end
    return best_pair(candidates)
end

function best_discard(state, params::SafeHeuristicParams, player0::Int, legal, derived::DerivedHeuristicConfig)
    candidates = Tuple{Float64,Int}[]
    opponent0 = 1 - player0
    for (idx, card) in enumerate(hand(state, player0))
        slot0 = idx - 1
        action = discard_action(slot0)
        !legal[action + 1] && continue
        my_value = card_value_for_me(state, params, player0, card, derived)
        opponent_value = card_value_for_opponent(state, params, opponent0, card, derived)
        score = -my_value - params.gift_penalty_weight * opponent_value
        !can_play_card(state, player0, card) && (score += params.unusable_discard_bonus)
        !can_play_card(state, opponent0, card) && (score += params.discard_safety_bonus)
        is_handshake(card) && can_play_card(state, player0, card) && (score -= 4.0)
        push!(candidates, (score, action))
    end
    return best_pair(candidates)
end

function deck_draw_value(state, params::SafeHeuristicParams, derived::DerivedHeuristicConfig)::Float64
    deck_left = length(state["deck"])
    diff = score_diff(state, Int(state["current_player"]))
    value = deck_left > derived.mid_deck_threshold ? params.deck_draw_early_value : deck_left > derived.late_deck_threshold ? params.deck_draw_mid_value : params.deck_draw_late_value
    value += diff > 0 ? params.winning_deck_bonus : -params.losing_deck_penalty
    return value
end

function act_card(state, params::SafeHeuristicParams, derived::DerivedHeuristicConfig)::Int
    player0 = Int(state["current_player"])
    legal = legal_card_mask(state)
    deck_left = length(state["deck"])
    action = best_handshake_play(state, params, player0, legal, derived, deck_left)
    action !== nothing && return action
    action = best_number_play(state, params, player0, legal, derived, deck_left)
    action !== nothing && return action
    if all(isempty(expedition) for expedition in expeditions(state, player0))
        action = best_forced_open(state, params, player0, legal, derived, deck_left)
        action !== nothing && return action
    end
    action = best_discard(state, params, player0, legal, derived)
    action !== nothing && return action
    return first_legal(legal)
end

function act_draw(state, params::SafeHeuristicParams, derived::DerivedHeuristicConfig)::Int
    legal = legal_draw_mask(state)
    player0 = Int(state["current_player"])
    candidates = Tuple{Float64,Int,Int}[]
    legal[DRAW_FROM_DECK_ACTION + 1] && push!(candidates, (deck_draw_value(state, params, derived), 1, DRAW_FROM_DECK_ACTION))
    for color in 0:(Int(state["config"]["n_colors"]) - 1)
        action = draw_from_discard_action(color)
        pile = discard_pile(state, color)
        (!legal[action + 1] || isempty(pile)) && continue
        card = pile[end]
        value = visible_draw_value(state, params, player0, card, derived)
        push!(candidates, (value, 0, action))
    end
    action = best_triple(candidates)
    action !== nothing && return action
    return first_legal(legal)
end

function safe_heuristic_action(record_or_state; variant::AbstractString="default")::Int
    state = haskey(record_or_state, "state") ? record_or_state["state"] : record_or_state
    params = params_for_variant(haskey(record_or_state, "variant") ? record_or_state["variant"] : variant)
    derived = derive(state["config"], params)
    return state["phase"] == "card" ? act_card(state, params, derived) : act_draw(state, params, derived)
end

end
