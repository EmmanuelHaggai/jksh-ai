use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
    
    #[error("Authentication error: {0}")]
    Auth(String),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Internal server error: {0}")]
    Internal(String),
    
    #[error("External service error: {0}")]
    ExternalService(String),
    
    #[error("Computation error: {0}")]
    Computation(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::Database(ref e) => {
                tracing::error!("Database error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "Database error occurred")
            }
            AppError::Redis(ref e) => {
                tracing::error!("Redis error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "Cache error occurred")
            }
            AppError::Auth(ref message) => (StatusCode::UNAUTHORIZED, message.as_str()),
            AppError::Validation(ref message) => (StatusCode::BAD_REQUEST, message.as_str()),
            AppError::NotFound(ref message) => (StatusCode::NOT_FOUND, message.as_str()),
            AppError::Internal(ref message) => {
                tracing::error!("Internal error: {}", message);
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
            }
            AppError::ExternalService(ref message) => {
                tracing::error!("External service error: {}", message);
                (StatusCode::BAD_GATEWAY, "External service error")
            }
            AppError::Computation(ref message) => {
                tracing::error!("Computation error: {}", message);
                (StatusCode::INTERNAL_SERVER_ERROR, "Computation failed")
            }
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16()
        }));

        (status, body).into_response()
    }
}

pub type Result<T> = std::result::Result<T, AppError>;