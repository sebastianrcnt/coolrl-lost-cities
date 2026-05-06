use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use crate::error::{EngineError, EngineErrorKind};
use crate::proto;
use crate::LostCitiesEngine;

#[derive(Clone, Default)]
pub struct LostCitiesGrpcService {
    engine: Arc<Mutex<LostCitiesEngine>>,
}

impl LostCitiesGrpcService {
    pub fn new(engine: LostCitiesEngine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
        }
    }
}

#[tonic::async_trait]
impl proto::lost_cities_server::LostCities for LostCitiesGrpcService {
    async fn new_game(
        &self,
        request: Request<proto::NewGameRequest>,
    ) -> Result<Response<proto::GameObservation>, Status> {
        let mut engine = self.engine.lock().map_err(|_| poisoned_engine_status())?;
        let observation = engine
            .new_game(request.into_inner())
            .map_err(map_engine_error)?;
        Ok(Response::new(observation))
    }

    async fn get_observation(
        &self,
        request: Request<proto::SessionRef>,
    ) -> Result<Response<proto::GameObservation>, Status> {
        let engine = self.engine.lock().map_err(|_| poisoned_engine_status())?;
        let observation = engine
            .get_observation(request.into_inner())
            .map_err(map_engine_error)?;
        Ok(Response::new(observation))
    }

    async fn apply_action(
        &self,
        request: Request<proto::ApplyActionRequest>,
    ) -> Result<Response<proto::StepResult>, Status> {
        let mut engine = self.engine.lock().map_err(|_| poisoned_engine_status())?;
        let result = engine
            .apply_action(request.into_inner())
            .map_err(map_engine_error)?;
        Ok(Response::new(result))
    }

    async fn end_session(
        &self,
        request: Request<proto::SessionRef>,
    ) -> Result<Response<()>, Status> {
        let mut engine = self.engine.lock().map_err(|_| poisoned_engine_status())?;
        engine
            .end_session(request.into_inner())
            .map_err(map_engine_error)?;
        Ok(Response::new(()))
    }
}

fn poisoned_engine_status() -> Status {
    Status::internal("lost cities engine mutex poisoned")
}

fn map_engine_error(error: EngineError) -> Status {
    match error.kind() {
        EngineErrorKind::AlreadyExists => Status::already_exists(error.message().to_string()),
        EngineErrorKind::NotFound => Status::not_found(error.message().to_string()),
        EngineErrorKind::FailedPrecondition => {
            Status::failed_precondition(error.message().to_string())
        }
        EngineErrorKind::InvalidArgument => Status::invalid_argument(error.message().to_string()),
    }
}
