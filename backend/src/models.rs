use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use rust_decimal::Decimal;
use bigdecimal::BigDecimal;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub password_hash: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub is_active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateUserRequest {
    pub email: String,
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginResponse {
    pub token: String,
    pub user: UserProfile,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserProfile {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub created_at: DateTime<Utc>,
    pub is_active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplexComputationRequest {
    pub operation: String,
    pub data: Vec<f64>,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplexComputationResponse {
    pub result: serde_json::Value,
    pub computation_time_ms: u128,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchProcessRequest {
    pub items: Vec<serde_json::Value>,
    pub operation_type: String,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchProcessResponse {
    pub processed_count: usize,
    pub results: Vec<serde_json::Value>,
    pub errors: Vec<String>,
    pub processing_time_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MLPredictRequest {
    pub model_type: String,
    pub features: Vec<f64>,
    pub parameters: Option<HashMap<String, f64>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MLPredictResponse {
    pub prediction: f64,
    pub confidence: f64,
    pub model_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MLTrainRequest {
    pub model_type: String,
    pub training_data: Vec<TrainingExample>,
    pub hyperparameters: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingExample {
    pub features: Vec<f64>,
    pub target: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MLTrainResponse {
    pub model_id: String,
    pub accuracy: f64,
    pub training_time_ms: u128,
    pub model_parameters: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BlockchainBlock {
    pub index: u64,
    pub timestamp: DateTime<Utc>,
    pub data: String,
    pub previous_hash: String,
    pub hash: String,
    pub nonce: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BlockchainValidateRequest {
    pub blocks: Vec<BlockchainBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BlockchainValidateResponse {
    pub is_valid: bool,
    pub invalid_blocks: Vec<u64>,
    pub validation_time_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CryptoRequest {
    pub operation: String,
    pub data: String,
    pub algorithm: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CryptoResponse {
    pub result: String,
    pub algorithm_used: String,
    pub processing_time_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub coordinates: Option<(f64, f64)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub weight: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ShortestPathRequest {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub start: String,
    pub end: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ShortestPathResponse {
    pub path: Vec<String>,
    pub total_distance: f64,
    pub computation_time_ms: u128,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TravelingSalesmanRequest {
    pub cities: Vec<GraphNode>,
    pub distance_matrix: Vec<Vec<f64>>,
    pub optimization_method: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TravelingSalesmanResponse {
    pub best_route: Vec<String>,
    pub total_distance: f64,
    pub computation_time_ms: u128,
    pub iterations: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalyticsData {
    pub metric_name: String,
    pub value: f64,
    pub timestamp: DateTime<Utc>,
    pub dimensions: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalyticsResponse {
    pub data: Vec<AnalyticsData>,
    pub aggregations: HashMap<String, f64>,
    pub time_range: (DateTime<Utc>, DateTime<Utc>),
}

// WhatsApp Cloud API Models
#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppWebhookRequest {
    pub object: String,
    pub entry: Vec<WhatsAppEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppEntry {
    pub id: String,
    pub changes: Vec<WhatsAppChange>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppChange {
    pub value: WhatsAppValue,
    pub field: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppValue {
    pub messaging_product: String,
    pub metadata: WhatsAppMetadata,
    pub messages: Option<Vec<WhatsAppMessage>>,
    pub statuses: Option<Vec<WhatsAppStatus>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppMetadata {
    pub display_phone_number: String,
    pub phone_number_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppMessage {
    pub from: String,
    pub id: String,
    pub timestamp: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub text: Option<WhatsAppText>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppText {
    pub body: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppStatus {
    pub id: String,
    pub status: String,
    pub timestamp: String,
    pub recipient_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppOutgoingMessage {
    pub messaging_product: String,
    pub to: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub text: Option<WhatsAppOutgoingText>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WhatsAppOutgoingText {
    pub body: String,
}

// Financial AI Models
#[derive(Debug, Serialize, Deserialize)]
pub struct StockData {
    pub symbol: String,
    pub price: Decimal,
    pub change: Decimal,
    pub change_percent: Decimal,
    pub volume: u64,
    pub market_cap: Option<Decimal>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketDataRequest {
    pub symbols: Vec<String>,
    pub timeframe: Option<String>, // 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketDataResponse {
    pub data: Vec<StockData>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StockAnalysisRequest {
    pub symbol: String,
    pub analysis_type: String, // technical, fundamental, sentiment
    pub timeframe: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StockAnalysisResponse {
    pub symbol: String,
    pub analysis_type: String,
    pub recommendation: String, // BUY, SELL, HOLD
    pub confidence: f64,
    pub price_target: Option<Decimal>,
    pub key_metrics: HashMap<String, serde_json::Value>,
    pub analysis_summary: String,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortfolioAnalysisRequest {
    pub holdings: Vec<PortfolioHolding>,
    pub cash_balance: Decimal,
    pub analysis_period_days: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortfolioHolding {
    pub symbol: String,
    pub shares: Decimal,
    pub average_cost: Decimal,
    pub current_price: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortfolioAnalysisResponse {
    pub total_value: Decimal,
    pub total_return: Decimal,
    pub total_return_percent: Decimal,
    pub daily_change: Decimal,
    pub daily_change_percent: Decimal,
    pub holdings_analysis: Vec<HoldingAnalysis>,
    pub risk_metrics: RiskMetrics,
    pub recommendations: Vec<String>,
    pub diversification_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HoldingAnalysis {
    pub symbol: String,
    pub current_value: Decimal,
    pub unrealized_gain_loss: Decimal,
    pub unrealized_gain_loss_percent: Decimal,
    pub weight_in_portfolio: Decimal,
    pub recommendation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub beta: Option<f64>,
    pub sharpe_ratio: Option<f64>,
    pub volatility: f64,
    pub max_drawdown: f64,
    pub var_95: f64, // Value at Risk 95%
    pub risk_score: f64, // 1-10 scale
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskAssessmentRequest {
    pub investment_amount: Decimal,
    pub risk_tolerance: String, // conservative, moderate, aggressive
    pub investment_horizon: u32, // years
    pub current_age: Option<u32>,
    pub financial_goals: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskAssessmentResponse {
    pub risk_profile: String,
    pub recommended_allocation: HashMap<String, Decimal>, // asset_class -> percentage
    pub expected_return: f64,
    pub expected_volatility: f64,
    pub investment_recommendations: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIRecommendationRequest {
    pub user_id: String,
    pub query: String,
    pub context: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIRecommendationResponse {
    pub response: String,
    pub recommendations: Vec<String>,
    pub relevant_stocks: Vec<String>,
    pub confidence: f64,
    pub response_type: String, // advice, analysis, news, explanation
    pub follow_up_questions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TechnicalAnalysisRequest {
    pub symbol: String,
    pub indicators: Vec<String>, // rsi, macd, sma, ema, bollinger_bands
    pub period: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TechnicalAnalysisResponse {
    pub symbol: String,
    pub indicators: HashMap<String, TechnicalIndicator>,
    pub signals: Vec<TechnicalSignal>,
    pub overall_signal: String, // BULLISH, BEARISH, NEUTRAL
    pub strength: f64, // 0-1
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TechnicalIndicator {
    pub name: String,
    pub value: f64,
    pub signal: String, // BUY, SELL, NEUTRAL
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TechnicalSignal {
    pub indicator: String,
    pub signal: String,
    pub strength: f64,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SentimentAnalysisRequest {
    pub symbol: String,
    pub sources: Vec<String>, // news, social, analyst_reports
    pub timeframe: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SentimentAnalysisResponse {
    pub symbol: String,
    pub overall_sentiment: f64, // -1 to 1
    pub sentiment_label: String, // VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE
    pub confidence: f64,
    pub source_breakdown: HashMap<String, f64>,
    pub key_topics: Vec<String>,
    pub sentiment_trend: Vec<SentimentDataPoint>,
    pub news_summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SentimentDataPoint {
    pub timestamp: DateTime<Utc>,
    pub sentiment: f64,
    pub volume: u32, // number of mentions/articles
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PricePredictionRequest {
    pub symbol: String,
    pub prediction_horizon: u32, // days
    pub model_type: String, // lstm, arima, prophet, ensemble
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PricePredictionResponse {
    pub symbol: String,
    pub current_price: Decimal,
    pub predicted_prices: Vec<PricePrediction>,
    pub confidence_intervals: Vec<ConfidenceInterval>,
    pub model_accuracy: f64,
    pub key_factors: Vec<String>,
    pub prediction_summary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PricePrediction {
    pub date: DateTime<Utc>,
    pub predicted_price: Decimal,
    pub confidence: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub date: DateTime<Utc>,
    pub lower_bound: Decimal,
    pub upper_bound: Decimal,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketNewsRequest {
    pub categories: Vec<String>, // stocks, crypto, economy, earnings
    pub limit: Option<u32>,
    pub symbols: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MarketNewsResponse {
    pub articles: Vec<NewsArticle>,
    pub total_count: u32,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NewsArticle {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub url: String,
    pub source: String,
    pub published_at: DateTime<Utc>,
    pub sentiment: Option<f64>,
    pub related_symbols: Vec<String>,
    pub category: String,
}

// User session management for WhatsApp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    pub user_id: String,
    pub phone_number: String,
    pub session_id: String,
    pub current_conversation: String,
    pub context: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub is_active: bool,
}