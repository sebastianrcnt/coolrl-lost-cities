pub mod proto {
    tonic::include_proto!("lost_cities.v1");
}

mod config;
mod engine;
mod error;
mod service;
mod state;

pub use config::Config;
pub use engine::LostCitiesEngine;
pub use error::{EngineError, EngineErrorKind};
pub use service::LostCitiesGrpcService;
pub use state::{score_expedition, Card, GameState, Phase};
