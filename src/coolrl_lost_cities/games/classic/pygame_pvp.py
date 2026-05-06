"""Lost Cities classic pygame GUI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .backends.common import snapshot_summary
from .backends.python import PythonLostCitiesBackend
from .bots import DEFAULT_BOT, LostCitiesBot, available_bot_names, build_bot
from .game import Card, GameState, LostCitiesConfig, classic_config
from .interfaces import LostCitiesBackend, Snapshot
from .resources import theme_path

LOGGER = logging.getLogger("coolrl_lost_cities.games.classic.pygame_pvp")
ModeName = Literal["pvp", "pvc"]


COLOR_PALETTE = [
    (255, 78, 83),
    (83, 141, 244),
    (116, 174, 82),
    (232, 181, 59),
    (164, 99, 202),
    (67, 205, 196),
    (255, 139, 66),
    (220, 92, 170),
]

BG = (0, 0, 0)
CARD_BG = (3, 4, 5)
LINE = (58, 58, 64)
LINE_DARK = (35, 35, 39)
TEXT = (222, 222, 228)
MUTED = (158, 155, 160)
GOLD = (232, 178, 0)

COLOR_NAME_PALETTE = ["Red", "Blue", "Green", "Gold", "Violet", "Cyan", "Orange", "Rose"]


@dataclass(frozen=True)
class ActionTarget:
    rect: Any
    action_id: int
    label: str


def color_rgb(color: int) -> tuple[int, int, int]:
    return COLOR_PALETTE[color % len(COLOR_PALETTE)]


def color_name(color: int) -> str:
    if color < len(COLOR_NAME_PALETTE):
        return COLOR_NAME_PALETTE[color]
    return f"Color {color}"


def card_value_label(card: Card, config: LostCitiesConfig) -> str:
    if card.is_handshake:
        return "H"
    return str(card.numeric_value(config.min_rank))


def card_label(card: Card, config: LostCitiesConfig) -> str:
    return f"{color_name(card.color)} {card_value_label(card, config)}"


def cards_to_json(cards: list[Card]) -> list[dict[str, int]]:
    return [card.to_snapshot() for card in cards]


def snapshot_to_json(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "config": snapshot.config.to_snapshot(),
        "deck": cards_to_json(snapshot.deck),
        "hands": [cards_to_json(hand) for hand in snapshot.hands],
        "expeditions": [
            [cards_to_json(expedition) for expedition in player_expeditions]
            for player_expeditions in snapshot.expeditions
        ],
        "discards": [cards_to_json(discard) for discard in snapshot.discards],
        "current_player": snapshot.current_player,
        "phase": snapshot.phase,
        "pending_discarded_color": snapshot.pending_discarded_color,
        "turn_count": snapshot.turn_count,
        "terminal": snapshot.terminal,
        "legal_mask": list(snapshot.legal_mask),
        "scores": [snapshot.total_score(0), snapshot.total_score(1)],
        "score_diff_player0": snapshot.score_diff(0),
    }


def turn_identity_summary(identity: tuple[int, str, int] | None) -> str:
    if identity is None:
        return "없음"
    player, phase, turn = identity
    phase_name = "카드" if phase == "card" else "뽑기"
    return f"플레이어={player} 단계={phase_name} 턴={turn}"


def undo_until_player_card_phase(
    backend: LostCitiesBackend,
    *,
    player: int,
) -> int:
    undone = 0
    while backend.can_undo():
        changed = backend.undo()
        if not changed:
            break
        undone += 1
        snapshot = backend.snapshot()
        if snapshot.current_player == player and snapshot.phase == "card":
            break
    return undone


def configure_debug_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stderr,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(logging.DEBUG)
    LOGGER.setLevel(logging.DEBUG)


def preferred_font_path() -> Path | None:
    return None


def pygame_display_flags(pygame_module: Any) -> int:
    flags = pygame_module.RESIZABLE
    if os.environ.get("COOLRL_X11_SOFTWARE") != "1":
        flags |= pygame_module.SCALED
    return flags


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Lost Cities in one pygame window.")
    parser.add_argument("--mode", choices=("pvp", "pvc"), default="pvp")
    parser.add_argument("--bot", choices=available_bot_names(), default=DEFAULT_BOT)
    parser.add_argument(
        "--model",
        dest="bot",
        choices=available_bot_names(),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=964)
    parser.add_argument("--screenshot-on-start", action="store_true")
    return parser


class LostCitiesGuiApp:
    def __init__(
        self,
        *,
        mode: ModeName = "pvp",
        bot_name: str = DEFAULT_BOT,
        seed: int | None = None,
        width: int = 1536,
        height: int = 964,
        screenshot_on_start: bool = False,
    ):
        import pygame
        import pygame_gui

        configure_debug_logging()
        pygame.init()
        pygame.display.set_caption("COOLRL LOST CITIES")
        self.pygame = pygame
        self.pygame_gui = pygame_gui
        self.window_size = (width, height)
        self.screen = pygame.display.set_mode(
            self.window_size,
            pygame_display_flags(pygame),
            vsync=0,
        )
        self.font_path = preferred_font_path()
        self.font_cache: dict[tuple[int, bool], Any] = {}
        self.manager = pygame_gui.UIManager(self.window_size)
        self._configure_ui_theme()
        self.clock = pygame.time.Clock()

        self.seed = seed
        self.mode: ModeName = mode
        self.bot_name = bot_name
        self.computer_player = 1
        self.next_computer_action_at_ms = 0
        self.config = classic_config()
        self.computer_bot, self.computer_bot_label = self._build_computer_bot()
        self.backend: LostCitiesBackend = PythonLostCitiesBackend(self.config, self.seed)
        self.ui_elements: list[Any] = []
        self.hand_card_rects: dict[int, Any] = {}
        self.board_targets: list[ActionTarget] = []
        self.mode_dropdown: Any = None
        self.bot_dropdown: Any = None
        self.new_game_button: Any = None
        self.undo_button: Any = None
        self.export_button: Any = None
        self.selected_card_slot: int | None = None
        self.error_text: str | None = None
        self.export_text: str | None = None
        self.match_trace: list[dict[str, Any]] = []
        self.last_turn_identity: tuple[int, str, int] | None = None
        self.turn_flash_until_ms = 0
        self.screenshot_on_start = screenshot_on_start
        self.pending_startup_screenshot = screenshot_on_start
        self._reset_match_trace()
        self.rebuild_ui()
        LOGGER.debug(
            "GUI 앱 초기화: 모드=%s 봇=%s 시드=%s 크기=%sx%s 색상수=%s 손패=%s 덱=%s",
            self.mode,
            self.computer_bot_label,
            self.seed,
            width,
            height,
            self.config.n_colors,
            self.config.hand_size,
            self.config.deck_size,
        )

    def _bot_seed(self) -> int | None:
        if self.seed is None:
            return None
        return self.seed + 10_001

    def _computer_bot_display_name(self) -> str:
        return self.computer_bot_label

    def _build_computer_bot(self) -> tuple[LostCitiesBot, str]:
        return build_bot(self.bot_name, seed=self._bot_seed()), self.bot_name

    def _configure_ui_theme(self) -> None:
        theme = json.loads(theme_path().read_text())
        if self.font_path is not None:
            font_path = str(self.font_path)
            self.manager.add_font_paths("b612", font_path)
            self.manager.preload_fonts(
                [{"name": "b612", "point_size": size} for size in (14, 16, 18, 20)]
            )
            LOGGER.debug("B612 폰트 사용: %s", self.font_path)
        else:
            theme.get("defaults", {}).pop("font", None)
            LOGGER.debug("pygame_gui 기본 대체 폰트 사용")
        self.manager.get_theme().update_theming(theme)

    def run(self) -> None:
        pygame = self.pygame
        running = True
        while running:
            time_delta = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    LOGGER.debug("pygame 종료 이벤트")
                    running = False
                    continue
                self.manager.process_events(event)
                self.handle_ui_event(event)

            self.maybe_apply_computer_action()
            self.manager.update(time_delta)
            self.draw()
            self.manager.draw_ui(self.screen)
            if self.pending_startup_screenshot:
                self.pending_startup_screenshot = False
                self.save_screenshot()
            pygame.display.flip()
        pygame.quit()

    def handle_ui_event(self, event: Any) -> None:
        pygame_gui = self.pygame_gui
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.new_game_button:
                LOGGER.debug("새 게임 버튼 클릭")
                self.reset_game()
                return
            if event.ui_element == self.undo_button:
                LOGGER.debug("되돌리기 버튼 클릭")
                self.undo()
                return
            if event.ui_element == self.export_button:
                LOGGER.debug("내보내기 버튼 클릭")
                self.export_match_trace()
                return
        elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            if event.ui_element == self.mode_dropdown:
                LOGGER.debug("모드 변경: %s -> %s", self.mode, event.text)
                self.mode = event.text
                self.reset_game()
            elif event.ui_element == self.bot_dropdown:
                LOGGER.debug("봇 변경: %s -> %s", self.bot_name, event.text)
                self.bot_name = event.text
                self.reset_game()
        elif event.type == self.pygame.KEYDOWN:
            if event.key == self.pygame.K_z and (event.mod & self.pygame.KMOD_CTRL):
                LOGGER.debug("컨트롤+Z 입력")
                self.undo()
            elif event.key == self.pygame.K_F12:
                LOGGER.debug("F12 입력: 전체 스크린샷 저장")
                self.save_screenshot()
            elif event.key == self.pygame.K_F11:
                LOGGER.debug("F11 입력: 헤더 스크린샷 저장")
                self.save_header_screenshot()
        elif event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
            LOGGER.debug("마우스 클릭: 위치=%s", event.pos)
            self.handle_board_click(event.pos)

    def is_computer_turn(self, snapshot: Snapshot | None = None) -> bool:
        snapshot = snapshot or self.backend.snapshot()
        return (
            self.mode == "pvc"
            and not snapshot.terminal
            and snapshot.current_player == self.computer_player
        )

    def maybe_apply_computer_action(self) -> None:
        snapshot = self.backend.snapshot()
        if not self.is_computer_turn(snapshot):
            self.next_computer_action_at_ms = 0
            return
        now = self.pygame.time.get_ticks()
        if self.next_computer_action_at_ms == 0:
            self.next_computer_action_at_ms = now + 360
            return
        if now < self.next_computer_action_at_ms:
            return
        try:
            bot_input = self._computer_bot_input(snapshot)
            action_id = self.computer_bot.act(bot_input)
            if isinstance(bot_input, GameState):
                action_id = bot_input.to_unified_action(action_id)
            LOGGER.debug(
                "컴퓨터 액션 선택: 봇=%s 입력=%s 액션=%s 상태={%s}",
                self._computer_bot_display_name(),
                type(bot_input).__name__,
                action_id,
                snapshot_summary(snapshot),
            )
            self.apply_action(action_id, rebuild=False)
        except Exception as exc:
            self.error_text = str(exc)
            LOGGER.exception("컴퓨터 액션 실패: 봇=%s", self._computer_bot_display_name())
        self.next_computer_action_at_ms = self.pygame.time.get_ticks() + 360
        self.rebuild_ui()

    def _computer_bot_input(self, snapshot: Snapshot) -> Snapshot | GameState:
        if self.mode == "pvc":
            return GameState.from_snapshot(snapshot_to_json(snapshot))
        return snapshot

    def handle_board_click(self, pos: tuple[int, int]) -> None:
        snapshot = self.backend.snapshot()
        if self.is_computer_turn(snapshot):
            LOGGER.debug(
                "보드 클릭 무시: 컴퓨터 턴 위치=%s 상태={%s}", pos, snapshot_summary(snapshot)
            )
            return
        if snapshot.terminal:
            LOGGER.debug(
                "보드 클릭 무시: 종료 상태 위치=%s 상태={%s}", pos, snapshot_summary(snapshot)
            )
            return
        for target in self.board_targets:
            if target.rect.collidepoint(pos):
                LOGGER.debug(
                    "타겟 클릭: 라벨=%s 액션=%s 위치=%s 상태={%s}",
                    target.label,
                    target.action_id,
                    pos,
                    snapshot_summary(snapshot),
                )
                self.apply_action(target.action_id)
                return
        if snapshot.phase != "card":
            LOGGER.debug(
                "보드 클릭 무시: 카드 단계 아님 위치=%s 상태={%s}", pos, snapshot_summary(snapshot)
            )
            return
        for slot, rect in self.hand_card_rects.items():
            if rect.collidepoint(pos):
                self.selected_card_slot = slot
                card = snapshot.hands[snapshot.current_player][slot]
                LOGGER.debug(
                    "카드 선택: 슬롯=%s 카드=%s 상태={%s}",
                    slot,
                    card_label(card, snapshot.config),
                    snapshot_summary(snapshot),
                )
                return
        LOGGER.debug("보드 클릭: 대상 없음 위치=%s 상태={%s}", pos, snapshot_summary(snapshot))

    def reset_game(self) -> None:
        self.selected_card_slot = None
        self.hand_card_rects = {}
        self.board_targets = []
        self.last_turn_identity = None
        self.turn_flash_until_ms = 0
        self.next_computer_action_at_ms = 0
        self.export_text = None
        try:
            self.computer_bot, self.computer_bot_label = self._build_computer_bot()
            self.backend = PythonLostCitiesBackend(self.config, self.seed)
            self._reset_match_trace()
            self.error_text = None
            LOGGER.debug(
                "게임 초기화 완료: 시드=%s 상태={%s}",
                self.seed,
                snapshot_summary(self.backend.snapshot()),
            )
        except Exception as exc:
            self.error_text = str(exc)
            LOGGER.exception("게임 초기화 실패: 시드=%s", self.seed)
        self.rebuild_ui()

    def apply_action(self, action_id: int, *, rebuild: bool = True) -> None:
        try:
            before = self.backend.snapshot()
            LOGGER.debug("액션 적용 요청: 액션=%s 상태={%s}", action_id, snapshot_summary(before))
            self.backend.apply(action_id)
            after = self.backend.snapshot()
            self._append_match_trace_step(action_id=action_id, before=before, after=after)
            self.selected_card_slot = None
            self.hand_card_rects = {}
            self.board_targets = []
            self.error_text = None
            self.export_text = None
            LOGGER.debug(
                "액션 적용 완료: 액션=%s 상태={%s}",
                action_id,
                snapshot_summary(after),
            )
        except Exception as exc:
            self.error_text = str(exc)
            LOGGER.exception("액션 적용 실패: 액션=%s", action_id)
        if rebuild:
            self.rebuild_ui()

    def undo(self) -> None:
        try:
            before = self.backend.snapshot()
            if self.mode == "pvc":
                undo_count = undo_until_player_card_phase(
                    self.backend,
                    player=1 - self.computer_player,
                )
            else:
                undo_count = 1 if self.backend.undo() else 0
            changed = undo_count > 0
            self.selected_card_slot = None
            self.hand_card_rects = {}
            self.board_targets = []
            self.next_computer_action_at_ms = 0
            self.error_text = None if changed else "되돌릴 수 없음"
            if changed:
                for _ in range(undo_count):
                    if len(self.match_trace) <= 1:
                        break
                    self.match_trace.pop()
                self.export_text = None
            LOGGER.debug(
                "되돌리기 요청: 변경됨=%s 횟수=%s 이전={%s} 이후={%s}",
                changed,
                undo_count,
                snapshot_summary(before),
                snapshot_summary(self.backend.snapshot()),
            )
        except Exception as exc:
            self.error_text = str(exc)
            LOGGER.exception("되돌리기 실패")
        self.rebuild_ui()

    def _reset_match_trace(self) -> None:
        snapshot = self.backend.snapshot()
        self.match_trace = [
            self._trace_record(
                step_index=0,
                action_id=None,
                actor=None,
                phase_before=None,
                snapshot=snapshot,
            )
        ]

    def _append_match_trace_step(
        self,
        *,
        action_id: int,
        before: Snapshot,
        after: Snapshot,
    ) -> None:
        self.match_trace.append(
            self._trace_record(
                step_index=len(self.match_trace),
                action_id=action_id,
                actor=before.current_player,
                phase_before=before.phase,
                snapshot=after,
            )
        )

    def _trace_record(
        self,
        *,
        step_index: int,
        action_id: int | None,
        actor: int | None,
        phase_before: str | None,
        snapshot: Snapshot,
    ) -> dict[str, Any]:
        return {
            "type": "step",
            "step_index": step_index,
            "action_id": action_id,
            "actor": actor,
            "phase_before": phase_before,
            "state": snapshot_to_json(snapshot),
        }

    def export_match_trace(self) -> None:
        snapshot = self.backend.snapshot()
        if not snapshot.terminal:
            self.error_text = "대국 종료 후 내보낼 수 있음"
            self.rebuild_ui()
            return

        export_dir = Path.cwd() / "exports"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        seed_label = "none" if self.seed is None else str(self.seed)
        path = export_dir / (
            f"lost-cities-classic-{self.mode}-python-seed-{seed_label}-{timestamp}.jsonl"
        )
        metadata = {
            "type": "metadata",
            "format": "coolrl_lost_cities.games.classic.match_trace.v1",
            "created_at": datetime.now().astimezone().isoformat(),
            "variant": "classic",
            "mode": self.mode,
            "bot": self._computer_bot_display_name() if self.mode == "pvc" else None,
            "backend": "python",
            "seed": self.seed,
            "config": self.config.to_snapshot(),
            "step_count": len(self.match_trace),
            "score": [snapshot.total_score(0), snapshot.total_score(1)],
            "score_diff_player0": snapshot.score_diff(0),
        }
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                for record in self.match_trace:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.error_text = None
            self.export_text = f"Exported {path}"
            LOGGER.debug("대국 내보내기 완료: %s", path)
        except Exception as exc:
            self.error_text = str(exc)
            self.export_text = None
            LOGGER.exception("대국 내보내기 실패: %s", path)
        self.rebuild_ui()

    def _hand_layout(
        self, snapshot: Snapshot, player: int, y: int
    ) -> tuple[int, int, int, int, int]:
        width, _ = self.window_size
        status_x = width - 404
        card_x = 340
        available = max(360, status_x - card_x - 40)
        count = max(1, snapshot.config.hand_size)
        gap = 14
        card_w = min(126, max(54, (available - gap * (count - 1)) // count))
        card_h = max(70, int(card_w * 1.22))
        if count > 1:
            gap = max(8, min(18, (available - card_w * count) // (count - 1)))
        card_y = y + (14 if player == 1 else 28)
        return card_x, card_y, card_w, card_h, gap

    def rebuild_ui(self) -> None:
        pygame = self.pygame
        pygame_gui = self.pygame_gui
        width, _ = self.window_size
        for element in self.ui_elements:
            element.kill()
        self.ui_elements = []

        self.mode_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=["pvp", "pvc"],
            starting_option=self.mode,
            relative_rect=pygame.Rect(92, 17, 108, 46),
            manager=self.manager,
        )
        self.bot_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=available_bot_names(),
            starting_option=self.bot_name,
            relative_rect=pygame.Rect(288, 17, 138, 46),
            manager=self.manager,
        )
        if self.mode != "pvc":
            self.bot_dropdown.disable()
        game_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(448, 17, 190, 46),
            text="CLASSIC / PYTHON",
            manager=self.manager,
        )
        self.new_game_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(660, 17, 148, 46),
            text="NEW GAME",
            manager=self.manager,
        )
        self.undo_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(width - 248, 17, 110, 46),
            text="UNDO",
            manager=self.manager,
        )
        if not self.backend.can_undo():
            self.undo_button.disable()
        self.export_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(width - 128, 17, 110, 46),
            text="EXPORT",
            manager=self.manager,
        )
        if not self.backend.snapshot().terminal:
            self.export_button.disable()
        self.ui_elements.extend(
            [
                self.mode_dropdown,
                self.bot_dropdown,
                game_label,
                self.new_game_button,
                self.undo_button,
                self.export_button,
            ]
        )

    def draw(self) -> None:
        snapshot = self.backend.snapshot()
        self._sync_turn_identity(snapshot)
        self.hand_card_rects = {}
        self.board_targets = []
        self.screen.fill(BG)
        self._draw_header(snapshot)
        self._draw_player_area(snapshot, player=1, y=110)
        self._draw_center(snapshot)
        self._draw_player_area(snapshot, player=0, y=self.window_size[1] - 257)
        if self.error_text:
            self._draw_text(self.error_text.upper(), (740, 50), color_rgb(0), 18, bold=True)
        elif self.export_text:
            self._draw_text(self.export_text, (740, 50), GOLD, 16, bold=True)

    def _draw_header(self, snapshot: Snapshot) -> None:
        pygame = self.pygame
        width, _ = self.window_size
        compact = width < 1700
        header_left = 1098
        header_right = width - 266
        show_meta = header_right - header_left >= 280
        pygame.draw.line(self.screen, LINE, (0, 0), (width, 0), 1)
        pygame.draw.line(self.screen, LINE, (0, 90), (width, 90), 1)
        pygame.draw.line(self.screen, LINE, (1080, 0), (1080, 90), 1)
        self._draw_text("MODE", (31, 31), MUTED, 18)
        self._draw_text("BOT", (220, 31), MUTED, 18)
        self._draw_text("GAME", (448, 31), MUTED, 18)
        if snapshot.terminal:
            diff = snapshot.score_diff(0)
            if diff > 0:
                status = "P0 WINS" if compact else "PLAYER 0 WINS"
            elif diff < 0:
                status = "P1 WINS" if compact else "PLAYER 1 WINS"
            else:
                status = "DRAW"
            detail = "Start a new game." if compact else "START A NEW GAME TO PLAY AGAIN."
        elif snapshot.phase == "card":
            card = self._selected_card(snapshot)
            if self.is_computer_turn(snapshot):
                status = (
                    "CPU: CHOOSE CARD"
                    if compact
                    else f"COMPUTER ({self._computer_bot_display_name().upper()}): CHOOSE A CARD"
                )
                detail = "Waiting for bot action."
            elif card is None:
                status = (
                    f"P{snapshot.current_player}: CHOOSE CARD"
                    if compact
                    else f"PLAYER {snapshot.current_player}: CHOOSE A CARD"
                )
                detail = "Click active hand." if compact else "Click a card in the active hand."
            else:
                status = (
                    f"P{snapshot.current_player}: CHOOSE TARGET"
                    if compact
                    else f"PLAYER {snapshot.current_player}: CHOOSE CARD DESTINATION"
                )
                detail = (
                    f"{card_label(card, snapshot.config)} selected."
                    if compact
                    else f"{card_label(card, snapshot.config)} selected. Click expedition or discard."
                )
        else:
            if self.is_computer_turn(snapshot):
                status = (
                    "CPU: DRAW"
                    if compact
                    else f"COMPUTER ({self._computer_bot_display_name().upper()}): DRAW A CARD"
                )
                detail = "Waiting for bot action."
            else:
                status = (
                    f"P{snapshot.current_player}: DRAW"
                    if compact
                    else f"PLAYER {snapshot.current_player}: DRAW A CARD"
                )
                detail = (
                    "Click deck/discard."
                    if compact
                    else "Click the deck or a highlighted discard pile."
                )
        title_size = 24 if compact else 30
        detail_size = 14 if compact else 18
        header_clip = pygame.Rect(header_left, 0, max(0, header_right - header_left), 90)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(header_clip)
        self._draw_text(status, (header_left, 19), TEXT, title_size, bold=True)
        self._draw_text(detail, (header_left, 54), MUTED, detail_size)
        if show_meta:
            bot_meta = ""
            if self.mode == "pvc":
                bot_meta = f"   BOT: {self._computer_bot_display_name()}"
            meta = f"{self.mode.upper()}   TURN: {snapshot.turn_count}{bot_meta}"
            self._draw_text_right(meta, (header_right, 54), MUTED, detail_size)
        self.screen.set_clip(previous_clip)

    def _screenshot_dir(self) -> Path:
        return Path("/tmp")

    def _screenshot_path(self, suffix: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return self._screenshot_dir() / f"lost-cities-{suffix}-{timestamp}.png"

    def save_screenshot(self) -> Path:
        path = self._screenshot_path("full")
        self._screenshot_dir().mkdir(parents=True, exist_ok=True)
        self.pygame.image.save(self.screen, str(path))
        self.export_text = f"Saved {path}"
        LOGGER.debug("전체 스크린샷 저장: %s", path)
        return path

    def save_header_screenshot(self) -> Path:
        path = self._screenshot_path("header")
        self._screenshot_dir().mkdir(parents=True, exist_ok=True)
        header_surface = self.screen.subsurface(
            self.pygame.Rect(0, 0, self.window_size[0], 90)
        ).copy()
        self.pygame.image.save(header_surface, str(path))
        self.export_text = f"Saved {path}"
        LOGGER.debug("헤더 스크린샷 저장: %s", path)
        return path

    def _draw_player_area(self, snapshot: Snapshot, *, player: int, y: int) -> None:
        pygame = self.pygame
        width, _ = self.window_size
        active = not snapshot.terminal and snapshot.current_player == player
        border = GOLD if active else LINE
        panel = pygame.Rect(26, y, width - 52, 182 if player == 1 else 210)
        border_width = 4 if active and self._turn_flash_active(player) else 2 if active else 1
        self._draw_panel_rect(panel, border, width=border_width)
        self._draw_text(f"PLAYER {player}", (48, y + 23), TEXT, 28)
        self._draw_text(f"SCORE {snapshot.total_score(player)}", (204, y + 29), TEXT, 18)
        self._draw_text("HAND", (80, y + 76), MUTED, 18)
        pygame.draw.line(self.screen, LINE, (44, y + 77), (67, y + 77), 1)
        pygame.draw.line(self.screen, LINE, (49, y + 72), (49, y + 87), 1)

        card_x, card_y, card_w, card_h, card_gap = self._hand_layout(snapshot, player, y)

        for slot, card in enumerate(snapshot.hands[player]):
            selectable = active and snapshot.phase == "card"
            selected = selectable and slot == self.selected_card_slot
            rect = self._draw_card(
                card,
                (card_x + slot * (card_w + card_gap), card_y),
                snapshot.config,
                large=True,
                size=(card_w, card_h),
                selected=selected,
                selectable=selectable,
            )
            if selectable:
                self.hand_card_rects[slot] = rect

        if active:
            status_box = pygame.Rect(width - 404, y + (45 if player == 0 else 42), 330, 92)
            pygame.draw.rect(self.screen, BG, status_box)
            pygame.draw.rect(self.screen, LINE, status_box, width=1)
            active_label = "COMPUTER" if self.is_computer_turn(snapshot) else "ACTIVE"
            self._draw_text(active_label, (status_box.x + 22, status_box.y + 22), GOLD, 20)
            if self.is_computer_turn(snapshot):
                prompt = f"Bot: {self._computer_bot_display_name()}"
            elif snapshot.phase == "card":
                prompt = (
                    "Yellow border: expedition or discard"
                    if self.selected_card_slot is not None
                    else "Select card, then click a target"
                )
            else:
                prompt = "Click deck or discard"
            self._draw_text(prompt, (status_box.x + 22, status_box.y + 56), MUTED, 16)

    def _sync_turn_identity(self, snapshot: Snapshot) -> None:
        identity = (snapshot.current_player, snapshot.phase, snapshot.turn_count)
        if identity == self.last_turn_identity:
            return
        previous = self.last_turn_identity
        self.last_turn_identity = identity
        if snapshot.phase == "card" and not snapshot.terminal:
            self.turn_flash_until_ms = self.pygame.time.get_ticks() + 1200
            LOGGER.debug(
                "턴 전환 감지: 이전=%s 현재플레이어=%s 단계=%s 턴=%s",
                turn_identity_summary(previous),
                snapshot.current_player,
                "카드" if snapshot.phase == "card" else "뽑기",
                snapshot.turn_count,
            )

    def _turn_flash_active(self, player: int) -> bool:
        snapshot = self.backend.snapshot()
        return (
            snapshot.current_player == player
            and snapshot.phase == "card"
            and self.pygame.time.get_ticks() < self.turn_flash_until_ms
        )

    def _draw_center(self, snapshot: Snapshot) -> None:
        pygame = self.pygame
        width, height = self.window_size
        bottom_panel_y = height - 257
        board_y = 313
        board_h = max(220, bottom_panel_y - board_y - 20)
        board_x = 263 if width >= 1450 else 196
        board_rect = pygame.Rect(board_x, board_y, width - board_x - 30, board_h)
        pygame.draw.rect(self.screen, BG, board_rect)
        pygame.draw.rect(self.screen, LINE, board_rect, width=1)

        deck_h = min(264, board_h - 44)
        deck_rect = pygame.Rect(44, board_y + 44, 173, deck_h)
        deck_action = snapshot.card_action_size
        deck_target = snapshot.phase == "draw" and self._is_legal(snapshot, deck_action)
        self._draw_deck(snapshot, deck_rect, target=deck_target)
        if deck_target:
            self._register_target(deck_rect, deck_action, "덱에서 뽑기")

        selected_card = self._selected_card(snapshot)
        lane_count = snapshot.config.n_colors
        lane_gap = 14
        lane_x = board_rect.x + 26
        lane_y = board_rect.y + 63
        lane_w = (board_rect.width - 52 - lane_gap * (lane_count - 1)) // lane_count
        zone_h = max(42, (board_rect.height - 86 - 24) // 3)

        for color in range(lane_count):
            x = lane_x + color * (lane_w + lane_gap)
            lane_color = color_rgb(color)
            if color > 0:
                pygame.draw.line(
                    self.screen, LINE, (x - 8, board_rect.y), (x - 8, board_rect.bottom), 1
                )
            self._draw_text(
                color_name(color).upper(),
                (x + 12, board_rect.y + 24),
                lane_color,
                23,
                bold=True,
            )
            pygame.draw.line(
                self.screen,
                lane_color,
                (x + 12, board_rect.y + 51),
                (x + lane_w - 12, board_rect.y + 51),
                2,
            )

            p1_rect = pygame.Rect(x + 26, lane_y, lane_w - 52, zone_h)
            discard_rect = pygame.Rect(x + 26, lane_y + zone_h + 12, lane_w - 52, zone_h)
            p0_rect = pygame.Rect(x + 26, lane_y + 2 * (zone_h + 12), lane_w - 52, zone_h)

            play_target_player = None
            play_action = None
            discard_action = None
            if snapshot.phase == "card" and selected_card and selected_card.color == color:
                play_target_player = snapshot.current_player
                play_action = 2 * (self.selected_card_slot or 0)
                discard_action = play_action + 1

            for player, rect in ((1, p1_rect), (0, p0_rect)):
                action = play_action if play_target_player == player else None
                is_target = action is not None and self._is_legal(snapshot, action)
                self._draw_zone(
                    rect,
                    f"P{player} expedition",
                    f"Score {snapshot.expedition_score(player, color)}",
                    snapshot.expeditions[player][color],
                    snapshot.config,
                    target=is_target,
                    target_label="Play here" if is_target else None,
                )
                if is_target and action is not None:
                    self._register_target(rect, action, "탐험대에 놓기")

            draw_discard_action = snapshot.card_action_size + 1 + color
            discard_target = False
            discard_label = None
            target_action = None
            if (
                snapshot.phase == "card"
                and selected_card
                and selected_card.color == color
                and discard_action is not None
                and self._is_legal(snapshot, discard_action)
            ):
                discard_target = True
                discard_label = "Discard here"
                target_action = discard_action
            elif snapshot.phase == "draw" and self._is_legal(snapshot, draw_discard_action):
                discard_target = True
                discard_label = "Draw"
                target_action = draw_discard_action

            self._draw_zone(
                discard_rect,
                "Discard",
                "Stack",
                snapshot.discards[color],
                snapshot.config,
                target=discard_target,
                target_label=discard_label,
            )
            if discard_target and target_action is not None:
                target_label = (
                    "버린 더미에 버리기"
                    if discard_label == "Discard here"
                    else "버린 더미에서 뽑기"
                )
                self._register_target(discard_rect, target_action, target_label)

    def _draw_deck(self, snapshot: Snapshot, rect: Any, *, target: bool = False) -> None:
        pygame = self.pygame
        border = GOLD if target else LINE
        pygame.draw.rect(self.screen, BG, rect)
        pygame.draw.rect(self.screen, border, rect, width=2 if target else 1)
        compact = rect.height < 230
        self._draw_text_center(
            "DECK", pygame.Rect(rect.x, rect.y + 18, rect.width, 36), TEXT, 20 if compact else 22
        )
        card_w, card_h = (42, 58) if compact else (52, 72)
        card_back = pygame.Rect(
            rect.centerx - card_w // 2 - 4, rect.y + (68 if compact else 72), card_w, card_h
        )
        pygame.draw.rect(self.screen, BG, card_back)
        pygame.draw.rect(self.screen, TEXT, card_back, width=2)
        pygame.draw.rect(self.screen, LINE, card_back.move(8, 8), width=2)
        count_rect = (
            pygame.Rect(rect.x, card_back.bottom + 8, rect.width, 40)
            if compact
            else pygame.Rect(rect.x, rect.bottom - 92, rect.width, 64)
        )
        self._draw_text_center(
            str(len(snapshot.deck)), count_rect, TEXT, 36 if compact else 42, bold=True
        )
        if target:
            self._draw_text_center(
                "DRAW",
                pygame.Rect(rect.x, rect.bottom - 34, rect.width, 24),
                GOLD,
                15 if compact else 17,
            )

    def _draw_zone(
        self,
        rect: Any,
        title: str,
        subtitle: str,
        cards: list[Card],
        config: LostCitiesConfig,
        *,
        target: bool = False,
        target_label: str | None = None,
    ) -> None:
        pygame = self.pygame
        border = GOLD if target else LINE
        pygame.draw.rect(self.screen, BG, rect)
        pygame.draw.rect(self.screen, border, width=2 if target else 1, rect=rect)
        if rect.height < 58:
            if target and target_label:
                self._draw_target_badge(rect, target_label)
            if not cards:
                self._draw_text(
                    self._zone_title(title, rect.width), (rect.x + 18, rect.y + 12), TEXT, 14
                )
                self._draw_text("EMPTY", (rect.right - 56, rect.y + 13), MUTED, 12)
            else:
                self._draw_mini_card_row(
                    cards,
                    pygame.Rect(rect.x + 8, rect.y + 9, rect.width - 16, 24),
                    config,
                    preferred_size=24,
                    min_size=14,
                )
            return
        title_text = self._zone_title(title, rect.width)
        self._draw_text(
            title_text, (rect.x + 20, rect.y + 16), TEXT, 17 if rect.width < 220 else 18
        )
        if target and target_label:
            self._draw_target_badge(rect, target_label)
        if not cards:
            self._draw_text(subtitle, (rect.x + 20, rect.y + 51), MUTED, 15)
            self._draw_text("EMPTY", (rect.right - 72, rect.y + 51), MUTED, 15)
            return

        # Once cards exist, reserve the bottom band exclusively for cards.
        # This prevents multi-card expeditions from covering score/top-card text.
        self._draw_text_right(
            subtitle, (rect.right - 18, rect.y + 20), MUTED, 12 if rect.width < 220 else 13
        )
        row_size = min(30, max(18, rect.height - 70))
        row_rect = pygame.Rect(rect.x + 12, rect.bottom - row_size - 10, rect.width - 24, row_size)
        self._draw_mini_card_row(cards, row_rect, config, preferred_size=row_size)

    def _zone_title(self, title: str, width: int) -> str:
        text = title.upper()
        if width >= 220:
            return text
        return text.replace(" EXPEDITION", " EXP")

    def _mini_card_layout(
        self,
        width: int,
        count: int,
        *,
        preferred_size: int = 30,
        min_size: int = 16,
    ) -> tuple[int, int]:
        gap = 5
        usable = max(32, width)
        if count <= 1:
            return min(preferred_size, usable), gap

        size = min(preferred_size, (usable - gap * (count - 1)) // count)
        if size >= min_size:
            return size, gap

        # Keep cards readable before shrinking too hard: negative gap means overlap.
        size = min(preferred_size, max(min_size, usable // min(count, 5)))
        gap = (usable - size * count) // (count - 1)
        return size, min(5, gap)

    def _draw_mini_card_row(
        self,
        cards: list[Card],
        row_rect: Any,
        config: LostCitiesConfig,
        *,
        preferred_size: int = 30,
        min_size: int = 16,
    ) -> None:
        if not cards:
            return
        size, gap = self._mini_card_layout(
            row_rect.width,
            len(cards),
            preferred_size=preferred_size,
            min_size=min_size,
        )
        step = size + gap
        total_width = size + max(0, len(cards) - 1) * step
        start_x = row_rect.x + max(0, (row_rect.width - total_width) // 2)
        y = row_rect.y + max(0, (row_rect.height - size) // 2)
        for index, card in enumerate(cards):
            self._draw_mini_card(card, (start_x + index * step, y), config, size=size)

    def _draw_target_badge(self, rect: Any, label: str) -> None:
        if rect.width < 220:
            return
        pygame = self.pygame
        text = label.upper()
        font = self._font(12)
        surface = font.render(text, True, GOLD)
        badge = surface.get_rect()
        badge.width += 12
        badge.height += 6
        badge.topright = (rect.right - 8, rect.y - 11)
        pygame.draw.rect(self.screen, BG, badge)
        pygame.draw.rect(self.screen, GOLD, badge, width=1)
        self.screen.blit(surface, (badge.x + 6, badge.y + 3))

    def _draw_card(
        self,
        card: Card,
        pos: tuple[int, int],
        config: LostCitiesConfig,
        *,
        large: bool = False,
        size: tuple[int, int] | None = None,
        selected: bool = False,
        selectable: bool = False,
    ) -> Any:
        pygame = self.pygame
        x, y = pos
        width, height = size or ((126, 154) if large else (30, 30))
        color = color_rgb(card.color)
        rect = pygame.Rect(x, y, width, height)
        if selected:
            pygame.draw.rect(
                self.screen,
                GOLD,
                rect.inflate(10, 10),
                width=2,
            )
        pygame.draw.rect(self.screen, CARD_BG, rect)
        pygame.draw.rect(self.screen, color, rect, width=2)
        if large:
            label_size = max(10, min(19, width // 6))
            value_size = max(26, min(62, int(width * 0.48)))
            self._draw_text(color_name(card.color).upper(), (x + 10, y + 10), color, label_size)
            self._draw_text_center(
                card_value_label(card, config),
                pygame.Rect(x, y + max(24, height // 4), width, height - max(32, height // 3)),
                color,
                value_size,
                bold=True,
            )
        else:
            self._draw_text_center(
                card_value_label(card, config),
                rect,
                color,
                16,
                bold=True,
            )
        return rect

    def _draw_mini_card(
        self,
        card: Card,
        pos: tuple[int, int],
        config: LostCitiesConfig,
        *,
        size: int = 30,
    ) -> None:
        pygame = self.pygame
        x, y = pos
        rect = pygame.Rect(x, y, size, size)
        color = color_rgb(card.color)
        pygame.draw.rect(self.screen, CARD_BG, rect)
        pygame.draw.rect(self.screen, color, rect, width=1)
        self._draw_text_center(
            card_value_label(card, config), rect, color, max(11, size // 2), bold=True
        )

    def _selected_card(self, snapshot: Snapshot) -> Card | None:
        if snapshot.phase != "card" or self.selected_card_slot is None:
            return None
        hand = snapshot.hands[snapshot.current_player]
        if self.selected_card_slot >= len(hand):
            self.selected_card_slot = None
            return None
        return hand[self.selected_card_slot]

    def _is_legal(self, snapshot: Snapshot, action_id: int) -> bool:
        return 0 <= action_id < len(snapshot.legal_mask) and snapshot.legal_mask[action_id]

    def _register_target(self, rect: Any, action_id: int, label: str) -> None:
        self.board_targets.append(ActionTarget(rect=rect, action_id=action_id, label=label))

    def _draw_panel(self, rect: tuple[int, int, int, int], border: tuple[int, int, int]) -> None:
        pygame = self.pygame
        panel = pygame.Rect(*rect)
        self._draw_panel_rect(panel, border)

    def _draw_panel_rect(
        self,
        panel: Any,
        border: tuple[int, int, int],
        *,
        width: int | None = None,
    ) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, BG, panel)
        border_width = width if width is not None else 2 if border == GOLD else 1
        pygame.draw.rect(self.screen, border, panel, width=border_width)

    def _draw_text(
        self,
        text: str,
        pos: tuple[int, int],
        color: tuple[int, int, int],
        size: int,
        *,
        bold: bool = False,
    ) -> None:
        font = self._font(size, bold=bold)
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)

    def _draw_text_center(
        self,
        text: str,
        rect: Any,
        color: tuple[int, int, int],
        size: int,
        *,
        bold: bool = False,
    ) -> None:
        font = self._font(size, bold=bold)
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(center=rect.center))

    def _draw_text_right(
        self,
        text: str,
        top_right: tuple[int, int],
        color: tuple[int, int, int],
        size: int,
        *,
        bold: bool = False,
    ) -> None:
        font = self._font(size, bold=bold)
        surface = font.render(text, True, color)
        self.screen.blit(surface, surface.get_rect(topright=top_right))

    def _font(self, size: int, *, bold: bool = False) -> Any:
        key = (size, bold)
        if key not in self.font_cache:
            if self.font_path is not None:
                font = self.pygame.font.Font(str(self.font_path), size)
                font.set_bold(bold)
            else:
                font = self.pygame.font.SysFont("dejavusansmono", size, bold=bold)
            self.font_cache[key] = font
        return self.font_cache[key]


PvpApp = LostCitiesGuiApp


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    app = LostCitiesGuiApp(
        mode=args.mode,
        bot_name=args.bot,
        seed=args.seed,
        width=args.width,
        height=args.height,
        screenshot_on_start=args.screenshot_on_start,
    )
    app.run()


if __name__ == "__main__":
    main()
