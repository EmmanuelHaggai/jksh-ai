mod api;
mod auth;
mod cache;
mod config;
mod database;
mod error;
mod models;
mod services;
mod financial;
mod whatsapp;
mod ai;

use axum::{
    routing::{get, post},
    Router,
    http::Method,
};
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use std::sync::Arc;

use crate::{
    api::handlers,
    config::AppConfig,
    database::Database,
    cache::Cache,
    services::AppServices,
};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub database: Arc<Database>,
    pub cache: Arc<Cache>,
    pub services: Arc<AppServices>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "jksh_backend=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Arc::new(AppConfig::new()?);
    let database = Arc::new(Database::new(&config).await?);
    let cache = Arc::new(Cache::new(&config).await?);
    let services = Arc::new(AppServices::new(database.clone(), cache.clone()).await?);

    let app_state = AppState {
        config: config.clone(),
        database,
        cache,
        services,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(handlers::health_check))
        .route("/api/health", get(handlers::health_check))
        // WhatsApp Cloud API endpoints
        .route("/webhook", get(handlers::webhook_verify))
        .route("/webhook", post(handlers::webhook_receive))
        // Financial AI endpoints
        .route("/api/financial/market-data", get(handlers::get_market_data))
        .route("/api/financial/analyze-stock", post(handlers::analyze_stock))
        .route("/api/financial/portfolio-analysis", post(handlers::portfolio_analysis))
        .route("/api/financial/risk-assessment", post(handlers::risk_assessment))
        .route("/api/financial/ai-recommendation", post(handlers::ai_recommendation))
        .route("/api/financial/technical-analysis", post(handlers::technical_analysis))
        .route("/api/financial/sentiment-analysis", post(handlers::sentiment_analysis))
        .route("/api/financial/price-prediction", post(handlers::price_prediction))
        .route("/api/financial/market-news", get(handlers::market_news))
        // Authentication
        .route("/api/auth/register", post(handlers::register))
        .route("/api/auth/login", post(handlers::login))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(cors),
        )
        .with_state(app_state);

    let port = config.server.port;
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    
    tracing::info!("🚀 Server starting on port {}", port);
    
    axum::serve(listener, app).await?;

    Ok(())
}
