from __future__ import annotations

import torch

from coolrl_lost_cities.games.classic.deep_cfr.config import NetworkConfig
from coolrl_lost_cities.games.classic.deep_cfr.networks import (
    ColorAttention,
    ColorSharedNetwork,
    DeepCFRMLP,
)


class TestDeepCFRMLP:
    def test_basic_mlp_forward(self) -> None:
        mlp = DeepCFRMLP(input_dim=64, output_dim=32, hidden_size=128, num_layers=2)
        x = torch.randn(16, 64)
        output = mlp(x)
        assert output.shape == (16, 32)

    def test_mlp_from_config(self) -> None:
        config = NetworkConfig(kind="mlp", hidden_size=64, num_layers=2)
        mlp = DeepCFRMLP.from_config(input_dim=100, output_dim=50, config=config)
        assert isinstance(mlp, DeepCFRMLP)
        x = torch.randn(8, 100)
        output = mlp(x)
        assert output.shape == (8, 50)

    def test_mlp_zero_layers(self) -> None:
        mlp = DeepCFRMLP(input_dim=64, output_dim=32, hidden_size=128, num_layers=0)
        x = torch.randn(16, 64)
        output = mlp(x)
        assert output.shape == (16, 32)

    def test_mlp_gelu_activation(self) -> None:
        mlp = DeepCFRMLP(
            input_dim=64, output_dim=32, hidden_size=128, num_layers=2, activation="gelu"
        )
        x = torch.randn(16, 64)
        output = mlp(x)
        assert output.shape == (16, 32)

    def test_mlp_gradients(self) -> None:
        mlp = DeepCFRMLP(input_dim=64, output_dim=32, hidden_size=128, num_layers=2)
        x = torch.randn(16, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestColorSharedNetwork:
    def test_color_shared_basic(self) -> None:
        network = ColorSharedNetwork(input_dim=100, output_dim=50, hidden_size=64, num_layers=2)
        x = torch.randn(16, 100)
        output = network(x)
        assert output.shape == (16, 50)

    def test_color_shared_from_config(self) -> None:
        config = NetworkConfig(kind="color_shared", hidden_size=64, num_layers=2)
        network = DeepCFRMLP.from_config(input_dim=150, output_dim=75, config=config)
        assert isinstance(network, ColorSharedNetwork)
        x = torch.randn(8, 150)
        output = network(x)
        assert output.shape == (8, 75)

    def test_color_shared_splits_input_correctly(self) -> None:
        n_colors = 5
        color_block_size = 20
        input_dim = n_colors * color_block_size
        network = ColorSharedNetwork(input_dim=input_dim, output_dim=32, hidden_size=64)
        assert network.n_colors == n_colors
        assert network.color_block_size == color_block_size

    def test_color_shared_with_remainder(self) -> None:
        input_dim = 105
        network = ColorSharedNetwork(input_dim=input_dim, output_dim=50, hidden_size=64)
        x = torch.randn(8, input_dim)
        output = network(x)
        assert output.shape == (8, 50)

    def test_color_shared_different_batch_sizes(self) -> None:
        network = ColorSharedNetwork(input_dim=100, output_dim=50, hidden_size=64)
        for batch_size in [1, 4, 16, 32, 64]:
            x = torch.randn(batch_size, 100)
            output = network(x)
            assert output.shape == (batch_size, 50)

    def test_color_shared_gradients(self) -> None:
        network = ColorSharedNetwork(input_dim=100, output_dim=50, hidden_size=64, num_layers=2)
        x = torch.randn(16, 100, requires_grad=True)
        output = network(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        for param in network.parameters():
            assert param.grad is not None

    def test_color_shared_deterministic_with_seed(self) -> None:
        torch.manual_seed(42)
        network1 = ColorSharedNetwork(input_dim=100, output_dim=50, hidden_size=64)
        torch.manual_seed(42)
        network2 = ColorSharedNetwork(input_dim=100, output_dim=50, hidden_size=64)

        x = torch.randn(8, 100)
        torch.manual_seed(42)
        output1 = network1(x)
        torch.manual_seed(42)
        output2 = network2(x)
        torch.testing.assert_close(output1, output2)

    def test_color_shared_without_attention(self) -> None:
        network = ColorSharedNetwork(
            input_dim=100,
            output_dim=50,
            hidden_size=64,
            color_attention_layers=0,
        )
        assert network.color_attention is None
        x = torch.randn(8, 100)
        output = network(x)
        assert output.shape == (8, 50)


class TestColorAttention:
    def test_color_attention_forward(self) -> None:
        attention = ColorAttention(dim=64, num_layers=1, num_heads=4)
        x = torch.randn(8, 5, 64)
        output = attention(x)
        assert output.shape == (8, 5, 64)

    def test_color_attention_multiple_layers(self) -> None:
        for num_layers in [1, 2, 3]:
            attention = ColorAttention(dim=64, num_layers=num_layers, num_heads=4)
            x = torch.randn(8, 5, 64)
            output = attention(x)
            assert output.shape == (8, 5, 64)

    def test_color_attention_different_heads(self) -> None:
        for num_heads in [1, 2, 4, 8]:
            attention = ColorAttention(dim=64, num_layers=1, num_heads=num_heads)
            x = torch.randn(8, 5, 64)
            output = attention(x)
            assert output.shape == (8, 5, 64)

    def test_color_attention_gradients(self) -> None:
        attention = ColorAttention(dim=64, num_layers=1, num_heads=4)
        x = torch.randn(8, 5, 64, requires_grad=True)
        output = attention(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        for param in attention.parameters():
            assert param.grad is not None

    def test_color_attention_gelu(self) -> None:
        attention = ColorAttention(dim=64, num_layers=1, num_heads=4, activation="gelu")
        x = torch.randn(8, 5, 64)
        output = attention(x)
        assert output.shape == (8, 5, 64)


class TestColorSharedNetworkWithAttention:
    def test_color_shared_with_attention(self) -> None:
        network = ColorSharedNetwork(
            input_dim=100,
            output_dim=50,
            hidden_size=64,
            num_layers=2,
            color_attention_layers=1,
            color_attention_heads=4,
        )
        assert network.color_attention is not None
        x = torch.randn(8, 100)
        output = network(x)
        assert output.shape == (8, 50)

    def test_color_shared_with_multi_layer_attention(self) -> None:
        network = ColorSharedNetwork(
            input_dim=100,
            output_dim=50,
            hidden_size=64,
            num_layers=2,
            color_attention_layers=3,
            color_attention_heads=4,
        )
        x = torch.randn(8, 100)
        output = network(x)
        assert output.shape == (8, 50)

    def test_color_shared_with_attention_from_config(self) -> None:
        config = NetworkConfig(
            kind="color_shared",
            hidden_size=64,
            num_layers=2,
            color_attention_layers=2,
            color_attention_heads=4,
        )
        network = DeepCFRMLP.from_config(input_dim=100, output_dim=50, config=config)
        assert isinstance(network, ColorSharedNetwork)
        assert network.color_attention is not None
        x = torch.randn(8, 100)
        output = network(x)
        assert output.shape == (8, 50)

    def test_color_shared_with_attention_gradients(self) -> None:
        network = ColorSharedNetwork(
            input_dim=100,
            output_dim=50,
            hidden_size=64,
            num_layers=2,
            color_attention_layers=1,
            color_attention_heads=4,
        )
        x = torch.randn(8, 100, requires_grad=True)
        output = network(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        for param in network.parameters():
            assert param.grad is not None


class TestNetworkBackwardCompatibility:
    def test_default_config_is_mlp(self) -> None:
        config = NetworkConfig()
        assert config.kind == "mlp"

    def test_from_config_respects_kind(self) -> None:
        mlp_config = NetworkConfig(kind="mlp")
        color_shared_config = NetworkConfig(kind="color_shared")

        mlp = DeepCFRMLP.from_config(input_dim=100, output_dim=50, config=mlp_config)
        color_shared = DeepCFRMLP.from_config(
            input_dim=100, output_dim=50, config=color_shared_config
        )

        assert isinstance(mlp, DeepCFRMLP)
        assert not isinstance(mlp, ColorSharedNetwork)
        assert isinstance(color_shared, ColorSharedNetwork)

    def test_mlp_and_color_shared_same_output_shape(self) -> None:
        input_dim = 100
        output_dim = 50
        x = torch.randn(8, input_dim)

        mlp_config = NetworkConfig(kind="mlp", hidden_size=64, num_layers=2)
        color_shared_config = NetworkConfig(kind="color_shared", hidden_size=64, num_layers=2)

        mlp = DeepCFRMLP.from_config(input_dim, output_dim, mlp_config)
        color_shared = DeepCFRMLP.from_config(input_dim, output_dim, color_shared_config)

        mlp_output = mlp(x)
        color_shared_output = color_shared(x)

        assert mlp_output.shape == color_shared_output.shape == (8, output_dim)


class TestNetworkIntegration:
    def test_mlp_with_real_input_size(self) -> None:
        from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
        from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

        game_config = LostCitiesConfig()
        state = GameState.new_game(game_config)
        dim = input_dim(state)
        action_size = 2 * game_config.hand_size + 1 + game_config.n_colors

        config = NetworkConfig(kind="mlp", hidden_size=64, num_layers=2)
        network = DeepCFRMLP.from_config(dim, action_size, config)

        x = torch.randn(8, dim)
        output = network(x)
        assert output.shape == (8, action_size)

    def test_color_shared_with_real_input_size(self) -> None:
        from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
        from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

        game_config = LostCitiesConfig()
        state = GameState.new_game(game_config)
        dim = input_dim(state)
        action_size = 2 * game_config.hand_size + 1 + game_config.n_colors

        config = NetworkConfig(kind="color_shared", hidden_size=64, num_layers=2)
        network = DeepCFRMLP.from_config(dim, action_size, config)

        x = torch.randn(8, dim)
        output = network(x)
        assert output.shape == (8, action_size)

    def test_color_shared_with_attention_real_input_size(self) -> None:
        from coolrl_lost_cities.games.classic.deep_cfr.encoding import input_dim
        from coolrl_lost_cities.games.classic.game import GameState, LostCitiesConfig

        game_config = LostCitiesConfig()
        state = GameState.new_game(game_config)
        dim = input_dim(state)
        action_size = 2 * game_config.hand_size + 1 + game_config.n_colors

        config = NetworkConfig(
            kind="color_shared", hidden_size=64, num_layers=2, color_attention_layers=1
        )
        network = DeepCFRMLP.from_config(dim, action_size, config)

        x = torch.randn(8, dim)
        output = network(x)
        assert output.shape == (8, action_size)


class TestComputeLostCitiesColorLayout:
    def test_layout_for_full_encoding_input_dim(self) -> None:
        from coolrl_lost_cities.games.classic.deep_cfr.networks import (
            compute_lost_cities_color_layout,
        )

        layout = compute_lost_cities_color_layout(297)
        assert layout is not None
        assert layout.n_colors == 5
        assert layout.color_block_size == 39
        assert layout.common_size == 297 - 5 * 39

        all_color_idx: set[int] = set()
        for indices in layout.per_color_indices:
            assert len(indices) == 39
            all_color_idx.update(indices)
        assert len(all_color_idx) == 5 * 39
        assert set(layout.common_indices).isdisjoint(all_color_idx)
        assert all_color_idx | set(layout.common_indices) == set(range(297))

    def test_layout_returns_none_for_unknown_input_dim(self) -> None:
        from coolrl_lost_cities.games.classic.deep_cfr.networks import (
            compute_lost_cities_color_layout,
        )

        assert compute_lost_cities_color_layout(100) is None
        assert compute_lost_cities_color_layout(150) is None
        assert compute_lost_cities_color_layout(296) is None

    def test_layout_recognises_all_four_flag_combinations(self) -> None:
        from coolrl_lost_cities.games.classic.deep_cfr.networks import (
            compute_lost_cities_color_layout,
        )

        for dim in (171, 219, 249, 297):
            layout = compute_lost_cities_color_layout(dim)
            assert layout is not None, f"layout missing for input_dim={dim}"
            assert layout.n_colors == 5

    def test_color_shared_uses_proper_layout_for_real_encoding(self) -> None:
        import warnings

        from coolrl_lost_cities.games.classic.deep_cfr.networks import ColorSharedNetwork

        network = ColorSharedNetwork(input_dim=297, output_dim=22, hidden_size=64)
        assert network.use_chunked_fallback is False
        assert network.color_block_size == 39

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x = torch.randn(4, 297)
            out = network(x)
        assert out.shape == (4, 22)
        assert not any("chunked" in str(w.message).lower() for w in caught)

    def test_color_shared_warns_on_non_lost_cities_input_dim(self) -> None:
        import warnings

        from coolrl_lost_cities.games.classic.deep_cfr.networks import ColorSharedNetwork

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            network = ColorSharedNetwork(input_dim=100, output_dim=20, hidden_size=64)
        assert network.use_chunked_fallback is True
        assert any("chunked" in str(w.message).lower() for w in caught)
