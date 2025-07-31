use crate::{
    error::Result,
    models::*,
    AppState,
    whatsapp::MessageProcessor,
};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use serde_json::json;
use std::collections::HashMap;
use tracing::{info, warn, error};

pub async fn health_check(State(state): State<AppState>) -> Result<impl IntoResponse> {
    let db_healthy = state.database.health_check().await.unwrap_or(false);
    let cache_healthy = state.cache.health_check().await.unwrap_or(false);

    let status = if db_healthy && cache_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let response = json!({
        "status": "ok",
        "timestamp": chrono::Utc::now(),
        "database": db_healthy,
        "cache": cache_healthy,
        "version": "1.0.0"
    });

    Ok((status, Json(response)))
}

pub async fn register(
    State(state): State<AppState>,
    Json(payload): Json<CreateUserRequest>,
) -> Result<impl IntoResponse> {
    // Validate input
    if payload.email.is_empty() || payload.username.is_empty() || payload.password.len() < 8 {
        return Ok((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Invalid input data"})),
        ));
    }

    // Check if user already exists
    if let Ok(Some(_)) = state.database.find_user_by_email(&payload.email).await {
        return Ok((
            StatusCode::CONFLICT,
            Json(json!({"error": "User already exists"})),
        ));
    }

    // Hash password
    let password_hash = state.services.auth.hash_password(&payload.password)?;

    // Create user
    let user = state
        .database
        .create_user(&payload.email, &payload.username, &password_hash)
        .await?;

    // Generate token
    let token = state
        .services
        .auth
        .generate_token(user.id, &user.email)?;

    let user_profile = UserProfile {
        id: user.id,
        email: user.email,
        username: user.username,
        created_at: user.created_at,
        is_active: user.is_active,
    };

    let response = LoginResponse {
        token,
        user: user_profile,
    };

    Ok((StatusCode::CREATED, Json(response)))
}

pub async fn login(
    State(state): State<AppState>,
    Json(payload): Json<LoginRequest>,
) -> Result<impl IntoResponse> {
    let user = state
        .database
        .find_user_by_email(&payload.email)
        .await?
        .ok_or_else(|| crate::error::AppError::Auth("Invalid credentials".to_string()))?;

    let is_valid = state
        .services
        .auth
        .verify_password(&payload.password, &user.password_hash)?;

    if !is_valid {
        return Ok((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "Invalid credentials"})),
        ));
    }

    let token = state
        .services
        .auth
        .generate_token(user.id, &user.email)?;

    let user_profile = UserProfile {
        id: user.id,
        email: user.email,
        username: user.username,
        created_at: user.created_at,
        is_active: user.is_active,
    };

    let response = LoginResponse {
        token,
        user: user_profile,
    };

    Ok((StatusCode::OK, Json(response)))
}

// WhatsApp Cloud API Handlers
pub async fn webhook_verify(
    Query(params): Query<HashMap<String, String>>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse> {
    let mode = params.get("hub.mode").unwrap_or(&"".to_string());
    let token = params.get("hub.verify_token").unwrap_or(&"".to_string());
    let challenge = params.get("hub.challenge").unwrap_or(&"".to_string());

    match state.services.whatsapp.verify_webhook(mode, token, challenge) {
        Ok(challenge_response) => {
            info!("Webhook verified successfully");
            Ok((StatusCode::OK, challenge_response))
        }
        Err(_) => {
            warn!("Webhook verification failed");
            Ok((StatusCode::FORBIDDEN, "Verification failed".to_string()))
        }
    }
}

pub async fn webhook_receive(
    State(state): State<AppState>,
    Json(payload): Json<WhatsAppWebhookRequest>,
) -> Result<impl IntoResponse> {
    info!("Received WhatsApp webhook: {:?}", payload);

    for entry in &payload.entry {
        for change in &entry.changes {
            if let Some(messages) = &change.value.messages {
                for message in messages {
                    if let Some(text) = &message.text {
                        // Process the incoming message
                        let response = process_whatsapp_message(&state, &message.from, &text.body).await?;
                        
                        // Send response back to user
                        if let Err(e) = state.services.whatsapp.send_message(&message.from, &response).await {
                            error!("Failed to send WhatsApp response: {}", e);
                        }
                    }
                }
            }
        }
    }

    Ok((StatusCode::OK, Json(json!({"status": "ok"}))))
}

async fn process_whatsapp_message(state: &AppState, phone_number: &str, message: &str) -> Result<String> {
    info!("Processing message from {}: {}", phone_number, message);

    // Extract command and process query
    let (command, symbol) = MessageProcessor::extract_command_and_symbol(message);
    
    match command.as_str() {
        "help" => {
            state.services.whatsapp.send_help_menu(phone_number).await?;
            Ok("Help menu sent!".to_string())
        }
        "price" => {
            if let Some(sym) = symbol {
                let market_data = state.services.financial.get_market_data(MarketDataRequest {
                    symbols: vec![sym.clone()],
                    timeframe: Some("1d".to_string()),
                }).await?;
                
                if let Some(stock) = market_data.data.first() {
                    Ok(format!(
                        "💰 {} Price Update:\n\n\
                        Current: ${:.2}\n\
                        Change: {:+.2}% (${:+.2})\n\
                        Volume: {:,}\n\n\
                        Last updated: {}\n\n\
                        Type 'analyze {}' for detailed analysis!",
                        sym,
                        stock.price.to_f64().unwrap_or(0.0),
                        stock.change_percent.to_f64().unwrap_or(0.0),
                        stock.change.to_f64().unwrap_or(0.0),
                        stock.volume,
                        stock.timestamp.format("%H:%M UTC"),
                        sym
                    ))
                } else {
                    Ok("Sorry, I couldn't find price data for that symbol. Please check the symbol and try again.".to_string())
                }
            } else {
                Ok("Please specify a stock symbol. Example: 'price AAPL'".to_string())
            }
        }
        "analyze" => {
            if let Some(sym) = symbol {
                let analysis = state.services.financial.analyze_stock(StockAnalysisRequest {
                    symbol: sym.clone(),
                    analysis_type: "technical".to_string(),
                    timeframe: Some("1mo".to_string()),
                }).await?;

                Ok(format!(
                    "📊 Analysis for {}:\n\n\
                    Recommendation: {} ({}% confidence)\n\n\
                    Summary: {}\n\n\
                    💡 This is technical analysis only. Consider fundamental factors and your risk tolerance before making investment decisions.\n\n\
                    Ask 'should I buy {}?' for investment advice!",
                    sym,
                    analysis.recommendation,
                    (analysis.confidence * 100.0) as u32,
                    analysis.analysis_summary,
                    sym
                ))
            } else {
                Ok("Please specify a stock symbol to analyze. Example: 'analyze TSLA'".to_string())
            }
        }
        "buy_advice" | "sell_advice" | "general_query" => {
            let ai_request = AIRecommendationRequest {
                user_id: phone_number.to_string(),
                query: message.to_string(),
                context: Some(HashMap::new()),
            };

            let ai_response = state.services.ai.process_query(ai_request).await?;
            state.services.whatsapp.send_ai_response(phone_number, &ai_response).await?;
            Ok("AI response sent!".to_string())
        }
        _ => {
            // Default to AI processing for natural language queries
            if MessageProcessor::is_financial_query(message) {
                let ai_request = AIRecommendationRequest {
                    user_id: phone_number.to_string(),
                    query: message.to_string(),
                    context: Some(HashMap::new()),
                };

                let ai_response = state.services.ai.process_query(ai_request).await?;
                state.services.whatsapp.send_ai_response(phone_number, &ai_response).await?;
                Ok("AI response sent!".to_string())
            } else {
                Ok("🤖 I'm your Financial AI Assistant! I can help you with:\n\n\
                   📊 Stock prices and analysis\n\
                   💼 Investment advice\n\
                   📈 Market insights\n\
                   🎓 Financial education\n\n\
                   Try asking:\n\
                   • 'price AAPL'\n\
                   • 'analyze Tesla'\n\
                   • 'should I buy Microsoft?'\n\
                   • 'help' for more options\n\n\
                   What would you like to know about?".to_string())
            }
        }
    }
}

// Financial AI API Handlers
pub async fn get_market_data(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse> {
    let symbols: Vec<String> = params
        .get("symbols")
        .map(|s| s.split(',').map(|symbol| symbol.trim().to_uppercase()).collect())
        .unwrap_or_else(|| vec!["SPY".to_string(), "QQQ".to_string(), "DIA".to_string()]);

    let timeframe = params.get("timeframe").cloned();

    let request = MarketDataRequest { symbols, timeframe };
    let response = state.services.financial.get_market_data(request).await?;

    Ok(Json(response))
}

pub async fn analyze_stock(
    State(state): State<AppState>,
    Json(payload): Json<StockAnalysisRequest>,
) -> Result<impl IntoResponse> {
    let response = state.services.financial.analyze_stock(payload).await?;
    Ok(Json(response))
}

pub async fn portfolio_analysis(
    State(state): State<AppState>,
    Json(payload): Json<PortfolioAnalysisRequest>,
) -> Result<impl IntoResponse> {
    let response = state.services.financial.analyze_portfolio(payload).await?;
    Ok(Json(response))
}

pub async fn risk_assessment(
    State(state): State<AppState>,
    Json(payload): Json<RiskAssessmentRequest>,
) -> Result<impl IntoResponse> {
    let response = state.services.financial.assess_risk(payload).await?;
    Ok(Json(response))
}

pub async fn ai_recommendation(
    State(state): State<AppState>,
    Json(payload): Json<AIRecommendationRequest>,
) -> Result<impl IntoResponse> {
    let response = state.services.ai.process_query(payload).await?;
    Ok(Json(response))
}

pub async fn technical_analysis(
    State(state): State<AppState>,
    Json(payload): Json<TechnicalAnalysisRequest>,
) -> Result<impl IntoResponse> {
    // Use the stock analysis with technical type
    let analysis_request = StockAnalysisRequest {
        symbol: payload.symbol,
        analysis_type: "technical".to_string(),
        timeframe: None,
    };

    let analysis = state.services.financial.analyze_stock(analysis_request).await?;
    
    // Convert to technical analysis response format
    let mut indicators = HashMap::new();
    for (key, value) in &analysis.key_metrics {
        if let Some(val) = value.as_f64() {
            indicators.insert(key.clone(), TechnicalIndicator {
                name: key.clone(),
                value: val,
                signal: if key == "rsi" {
                    if val > 70.0 { "SELL" } else if val < 30.0 { "BUY" } else { "NEUTRAL" }
                } else { "NEUTRAL" }.to_string(),
                description: format!("{} indicator", key),
            });
        }
    }

    let response = TechnicalAnalysisResponse {
        symbol: analysis.symbol,
        indicators,
        signals: vec![],
        overall_signal: analysis.recommendation.clone(),
        strength: analysis.confidence,
    };

    Ok(Json(response))
}

pub async fn sentiment_analysis(
    State(state): State<AppState>,
    Json(payload): Json<SentimentAnalysisRequest>,
) -> Result<impl IntoResponse> {
    let analysis = state.services.financial.analyze_stock(StockAnalysisRequest {
        symbol: payload.symbol.clone(),
        analysis_type: "sentiment".to_string(),
        timeframe: payload.timeframe,
    }).await?;

    let sentiment_score = analysis.key_metrics
        .get("overall_sentiment")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let sentiment_label = match sentiment_score {
        s if s > 0.5 => "VERY_POSITIVE",
        s if s > 0.2 => "POSITIVE", 
        s if s > -0.2 => "NEUTRAL",
        s if s > -0.5 => "NEGATIVE",
        _ => "VERY_NEGATIVE",
    }.to_string();

    let response = SentimentAnalysisResponse {
        symbol: payload.symbol,
        overall_sentiment: sentiment_score,
        sentiment_label,
        confidence: analysis.confidence,
        source_breakdown: HashMap::new(),
        key_topics: vec!["Market sentiment".to_string(), "News analysis".to_string()],
        sentiment_trend: vec![],
        news_summary: analysis.analysis_summary,
    };

    Ok(Json(response))
}

pub async fn price_prediction(
    State(state): State<AppState>,
    Json(payload): Json<PricePredictionRequest>,
) -> Result<impl IntoResponse> {
    let market_data = state.services.financial.get_market_data(MarketDataRequest {
        symbols: vec![payload.symbol.clone()],
        timeframe: Some("1d".to_string()),
    }).await?;

    let current_price = market_data.data.first()
        .map(|d| d.price)
        .unwrap_or_default();

    // Generate mock predictions
    let mut predictions = Vec::new();
    let mut intervals = Vec::new();
    let base_price = current_price.to_f64().unwrap_or(100.0);
    
    for i in 1..=payload.prediction_horizon {
        let predicted_price = base_price * (1.0 + (rand::random::<f64>() - 0.5) * 0.1);
        let date = chrono::Utc::now() + chrono::Duration::days(i as i64);
        
        predictions.push(PricePrediction {
            date,
            predicted_price: rust_decimal::Decimal::from_f64_retain(predicted_price).unwrap_or_default(),
            confidence: 0.7,
        });

        intervals.push(ConfidenceInterval {
            date,
            lower_bound: rust_decimal::Decimal::from_f64_retain(predicted_price * 0.95).unwrap_or_default(),
            upper_bound: rust_decimal::Decimal::from_f64_retain(predicted_price * 1.05).unwrap_or_default(),
        });
    }

    let response = PricePredictionResponse {
        symbol: payload.symbol,
        current_price,
        predicted_prices: predictions,
        confidence_intervals: intervals,
        model_accuracy: 0.75,
        key_factors: vec![
            "Technical indicators".to_string(),
            "Market sentiment".to_string(), 
            "Economic conditions".to_string(),
        ],
        prediction_summary: "Price prediction based on technical analysis and market trends".to_string(),
    };

    Ok(Json(response))
}

pub async fn market_news(
    State(_state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse> {
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(10);

    // Generate mock news articles
    let articles = vec![
        NewsArticle {
            id: "1".to_string(),
            title: "Tech Stocks Surge on AI Optimism".to_string(),
            summary: "Major technology companies see gains as investors remain bullish on AI developments".to_string(),
            url: "https://example.com/news/1".to_string(),
            source: "Financial Times".to_string(),
            published_at: chrono::Utc::now() - chrono::Duration::hours(2),
            sentiment: Some(0.8),
            related_symbols: vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()],
            category: "stocks".to_string(),
        },
        NewsArticle {
            id: "2".to_string(),
            title: "Federal Reserve Holds Interest Rates Steady".to_string(),
            summary: "Fed maintains current rates amid mixed economic signals".to_string(),
            url: "https://example.com/news/2".to_string(),
            source: "Reuters".to_string(),
            published_at: chrono::Utc::now() - chrono::Duration::hours(4),
            sentiment: Some(0.1),
            related_symbols: vec!["SPY".to_string(), "TLT".to_string()],
            category: "economy".to_string(),
        },
    ];

    let response = MarketNewsResponse {
        articles: articles.into_iter().take(limit as usize).collect(),
        total_count: limit,
        last_updated: chrono::Utc::now(),
    };

    Ok(Json(response))
}