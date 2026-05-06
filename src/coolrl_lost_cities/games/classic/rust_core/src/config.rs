use crate::error::EngineError;
use crate::proto;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub n_colors: usize,
    pub n_ranks: u32,
    pub min_rank: u32,
    pub n_handshakes: u32,
    pub hand_size: usize,
    pub expedition_penalty: i32,
    pub bonus_threshold: usize,
    pub bonus_amount: i32,
    pub seed: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_colors: 5,
            n_ranks: 9,
            min_rank: 2,
            n_handshakes: 3,
            hand_size: 8,
            expedition_penalty: -20,
            bonus_threshold: 8,
            bonus_amount: 20,
            seed: None,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.n_colors == 0 {
            return Err(EngineError::invalid_argument("n_colors must be positive"));
        }
        if self.n_ranks == 0 {
            return Err(EngineError::invalid_argument("n_ranks must be positive"));
        }
        if self.min_rank == 0 {
            return Err(EngineError::invalid_argument("min_rank must be positive"));
        }
        if self.hand_size == 0 {
            return Err(EngineError::invalid_argument("hand_size must be positive"));
        }
        if self.bonus_threshold == 0 {
            return Err(EngineError::invalid_argument(
                "bonus_threshold must be positive",
            ));
        }
        if self.deck_size() < 2 * self.hand_size {
            return Err(EngineError::invalid_argument(
                "deck must contain at least both initial hands",
            ));
        }
        Ok(())
    }

    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }

    pub fn deck_size(&self) -> usize {
        self.n_colors * (self.n_ranks as usize + self.n_handshakes as usize)
    }

    pub fn card_action_size(&self) -> usize {
        self.hand_size * 2
    }

    pub fn draw_action_size(&self) -> usize {
        1 + self.n_colors
    }

    pub fn action_space_size(&self) -> usize {
        self.card_action_size() + self.draw_action_size()
    }
}

impl TryFrom<&proto::GameConfig> for Config {
    type Error = EngineError;

    fn try_from(value: &proto::GameConfig) -> Result<Self, Self::Error> {
        let n_colors = usize::try_from(value.n_colors)
            .map_err(|_| EngineError::invalid_argument("n_colors is out of range"))?;
        let hand_size = usize::try_from(value.hand_size)
            .map_err(|_| EngineError::invalid_argument("hand_size is out of range"))?;
        let bonus_threshold = usize::try_from(value.bonus_threshold)
            .map_err(|_| EngineError::invalid_argument("bonus_threshold is out of range"))?;

        let config = Self {
            n_colors,
            n_ranks: value.n_ranks,
            min_rank: value.min_rank,
            n_handshakes: value.n_handshakes,
            hand_size,
            expedition_penalty: value.expedition_penalty,
            bonus_threshold,
            bonus_amount: value.bonus_amount,
            seed: value.seed,
        };
        config.validate()?;
        Ok(config)
    }
}

impl TryFrom<proto::GameConfig> for Config {
    type Error = EngineError;

    fn try_from(value: proto::GameConfig) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl From<&Config> for proto::GameConfig {
    fn from(value: &Config) -> Self {
        Self {
            n_colors: value.n_colors as u32,
            n_ranks: value.n_ranks,
            min_rank: value.min_rank,
            n_handshakes: value.n_handshakes,
            hand_size: value.hand_size as u32,
            expedition_penalty: value.expedition_penalty,
            bonus_threshold: value.bonus_threshold as u32,
            bonus_amount: value.bonus_amount,
            seed: value.seed,
        }
    }
}
