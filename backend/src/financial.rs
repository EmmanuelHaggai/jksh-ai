use crate::{
    error::{AppError, Result},
    models::*,
    cache::Cache,
};
use reqwest::Client;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc, Duration};
use serde_json::json;
use ta::{indicators::*, Next};

pub struct FinancialService {
    client: Client,
    cache: Arc<Cache>,
    api_keys: ApiKeys,
}

#[derive(Debug, Clone)]
struct ApiKeys {
    alpha_vantage: String,
    finnhub: String,
    polygon: String,
}

impl FinancialService {
    pub fn new(cache: Arc<Cache>) -> Self {
        Self {
            client: Client::new(),
            cache,
            api_keys: ApiKeys {
                alpha_vantage: std::env::var("ALPHA_VANTAGE_API_KEY")
                    .unwrap_or_else(|_| "demo".to_string()),
                finnhub: std::env::var("FINNHUB_API_KEY")
                    .unwrap_or_else(|_| "demo".to_string()),
                polygon: std::env::var("POLYGON_API_KEY")
                    .unwrap_or_else(|_| "demo".to_string()),
            },
        }
    }

    // Market Data Functions
    pub async fn get_market_data(&self, request: MarketDataRequest) -> Result<MarketDataResponse> {
        let cache_key = format!("market_data_{:?}_{}", request.symbols, request.timeframe.as_deref().unwrap_or("1d"));
        
        // Try cache first
        if let Ok(Some(cached_data)) = self.cache.get::<MarketDataResponse>(&cache_key).await {
            return Ok(cached_data);
        }

        let mut stock_data = Vec::new();
        
        for symbol in &request.symbols {
            match self.fetch_stock_data(symbol).await {
                Ok(data) => stock_data.push(data),
                Err(e) => {
                    tracing::warn!("Failed to fetch data for {}: {}", symbol, e);
                    // Create mock data for demo
                    stock_data.push(self.create_mock_stock_data(symbol));
                }
            }
        }

        let response = MarketDataResponse {
            data: stock_data,
            last_updated: Utc::now(),
        };

        // Cache for 1 minute
        let _ = self.cache.set(&cache_key, &response, Some(std::time::Duration::from_secs(60))).await;

        Ok(response)
    }

    async fn fetch_stock_data(&self, symbol: &str) -> Result<StockData> {
        // Try multiple data sources for redundancy
        
        // First try Alpha Vantage
        if let Ok(data) = self.fetch_from_alpha_vantage(symbol).await {
            return Ok(data);
        }

        // Fallback to mock data
        Ok(self.create_mock_stock_data(symbol))
    }

    async fn fetch_from_alpha_vantage(&self, symbol: &str) -> Result<StockData> {
        let url = format!(
            "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={}&apikey={}",
            symbol, self.api_keys.alpha_vantage
        );

        let response = self.client.get(&url).send().await
            .map_err(|e| AppError::ExternalService(format!("Alpha Vantage API error: {}", e)))?;

        let data: serde_json::Value = response.json().await
            .map_err(|e| AppError::ExternalService(format!("JSON parse error: {}", e)))?;

        if let Some(quote) = data.get("Global Quote") {
            let price = quote.get("05. price")
                .and_then(|p| p.as_str())
                .and_then(|p| p.parse::<f64>().ok())
                .unwrap_or(0.0);

            let change = quote.get("09. change")
                .and_then(|c| c.as_str())
                .and_then(|c| c.parse::<f64>().ok())
                .unwrap_or(0.0);

            let change_percent = quote.get("10. change percent")
                .and_then(|cp| cp.as_str())
                .and_then(|cp| cp.trim_end_matches('%').parse::<f64>().ok())
                .unwrap_or(0.0);

            let volume = quote.get("06. volume")
                .and_then(|v| v.as_str())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);

            return Ok(StockData {
                symbol: symbol.to_string(),
                price: Decimal::from_f64_retain(price).unwrap_or_default(),
                change: Decimal::from_f64_retain(change).unwrap_or_default(),
                change_percent: Decimal::from_f64_retain(change_percent).unwrap_or_default(),
                volume,
                market_cap: None,
                timestamp: Utc::now(),
            });
        }

        Err(AppError::ExternalService("Invalid response from Alpha Vantage".to_string()))
    }

    fn create_mock_stock_data(&self, symbol: &str) -> StockData {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let base_price = match symbol {
            "AAPL" => 180.0,
            "GOOGL" => 140.0,
            "MSFT" => 380.0,
            "TSLA" => 240.0,
            "NVDA" => 480.0,
            "AMZN" => 140.0,
            "META" => 320.0,
            "SPY" => 440.0,
            _ => 100.0,
        };

        let price_variation = rng.gen_range(-0.05..0.05);
        let price = base_price * (1.0 + price_variation);
        let change = base_price * price_variation;
        let change_percent = price_variation * 100.0;

        StockData {
            symbol: symbol.to_string(),
            price: Decimal::from_f64_retain(price).unwrap_or_default(),
            change: Decimal::from_f64_retain(change).unwrap_or_default(),
            change_percent: Decimal::from_f64_retain(change_percent).unwrap_or_default(),
            volume: rng.gen_range(1_000_000..50_000_000),
            market_cap: Some(Decimal::from_f64_retain(price * 1_000_000_000.0).unwrap_or_default()),
            timestamp: Utc::now(),
        }
    }

    // Stock Analysis Functions
    pub async fn analyze_stock(&self, request: StockAnalysisRequest) -> Result<StockAnalysisResponse> {
        let cache_key = format!("analysis_{}_{}", request.symbol, request.analysis_type);
        
        if let Ok(Some(cached)) = self.cache.get::<StockAnalysisResponse>(&cache_key).await {
            return Ok(cached);
        }

        let analysis = match request.analysis_type.as_str() {
            "technical" => self.perform_technical_analysis(&request.symbol).await?,
            "fundamental" => self.perform_fundamental_analysis(&request.symbol).await?,
            "sentiment" => self.perform_sentiment_analysis(&request.symbol).await?,
            _ => return Err(AppError::Validation("Invalid analysis type".to_string())),
        };

        // Cache for 15 minutes
        let _ = self.cache.set(&cache_key, &analysis, Some(std::time::Duration::from_secs(900))).await;

        Ok(analysis)
    }

    async fn perform_technical_analysis(&self, symbol: &str) -> Result<StockAnalysisResponse> {
        // Get historical data and calculate technical indicators
        let historical_data = self.get_historical_prices(symbol, 50).await?;
        
        // Calculate RSI
        let rsi = self.calculate_rsi(&historical_data, 14);
        
        // Calculate MACD
        let (macd_line, signal_line) = self.calculate_macd(&historical_data);
        
        // Calculate Moving Averages
        let sma_20 = self.calculate_sma(&historical_data, 20);
        let sma_50 = self.calculate_sma(&historical_data, 50);
        
        let current_price = historical_data.last().unwrap_or(&100.0);
        
        // Generate recommendation based on indicators
        let (recommendation, confidence) = self.generate_technical_recommendation(
            rsi, macd_line, signal_line, *current_price, sma_20, sma_50
        );

        let mut key_metrics = HashMap::new();
        key_metrics.insert("rsi".to_string(), json!(rsi));
        key_metrics.insert("macd".to_string(), json!(macd_line));
        key_metrics.insert("signal".to_string(), json!(signal_line));
        key_metrics.insert("sma_20".to_string(), json!(sma_20));
        key_metrics.insert("sma_50".to_string(), json!(sma_50));

        let analysis_summary = format!(
            "Technical analysis shows {} momentum. RSI: {:.2}, MACD: {:.2}. \
            Price is {} the 20-day SMA (${:.2}) and {} the 50-day SMA (${:.2}). \
            Current technical indicators suggest a {} position.",
            if rsi > 70.0 { "overbought" } else if rsi < 30.0 { "oversold" } else { "neutral" },
            rsi,
            macd_line,
            if *current_price > sma_20 { "above" } else { "below" },
            sma_20,
            if *current_price > sma_50 { "above" } else { "below" },
            sma_50,
            recommendation.to_lowercase()
        );

        Ok(StockAnalysisResponse {
            symbol: symbol.to_string(),
            analysis_type: "technical".to_string(),
            recommendation,
            confidence,
            price_target: None,
            key_metrics,
            analysis_summary,
            generated_at: Utc::now(),
        })
    }

    async fn perform_fundamental_analysis(&self, symbol: &str) -> Result<StockAnalysisResponse> {
        // Mock fundamental analysis - in real implementation, would fetch from financial APIs
        let mut key_metrics = HashMap::new();
        
        // Generate mock fundamental metrics
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let pe_ratio = rng.gen_range(15.0..35.0);
        let debt_to_equity = rng.gen_range(0.2..1.5);
        let roe = rng.gen_range(10.0..25.0);
        let revenue_growth = rng.gen_range(-5.0..15.0);
        let profit_margin = rng.gen_range(5.0..25.0);

        key_metrics.insert("pe_ratio".to_string(), json!(pe_ratio));
        key_metrics.insert("debt_to_equity".to_string(), json!(debt_to_equity));
        key_metrics.insert("roe".to_string(), json!(roe));
        key_metrics.insert("revenue_growth".to_string(), json!(revenue_growth));
        key_metrics.insert("profit_margin".to_string(), json!(profit_margin));

        let (recommendation, confidence) = self.generate_fundamental_recommendation(
            pe_ratio, debt_to_equity, roe, revenue_growth, profit_margin
        );

        let analysis_summary = format!(
            "Fundamental analysis indicates {} value. P/E ratio of {:.1} suggests the stock is {}. \
            ROE of {:.1}% shows {} profitability. Revenue growth of {:.1}% indicates {} business momentum. \
            Overall financial health appears {}.",
            if pe_ratio < 20.0 { "good" } else { "premium" },
            pe_ratio,
            if pe_ratio < 20.0 { "reasonably valued" } else { "expensive" },
            roe,
            if roe > 15.0 { "strong" } else { "moderate" },
            revenue_growth,
            if revenue_growth > 5.0 { "positive" } else { "challenging" },
            if roe > 15.0 && debt_to_equity < 1.0 { "solid" } else { "mixed" }
        );

        Ok(StockAnalysisResponse {
            symbol: symbol.to_string(),
            analysis_type: "fundamental".to_string(),
            recommendation,
            confidence,
            price_target: Some(Decimal::from_f64_retain(pe_ratio * 5.0).unwrap_or_default()),
            key_metrics,
            analysis_summary,
            generated_at: Utc::now(),
        })
    }

    async fn perform_sentiment_analysis(&self, symbol: &str) -> Result<StockAnalysisResponse> {
        // Mock sentiment analysis - in real implementation, would analyze news and social media
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let sentiment_score = rng.gen_range(-1.0..1.0);
        let news_sentiment = rng.gen_range(-1.0..1.0);
        let social_sentiment = rng.gen_range(-1.0..1.0);
        let analyst_sentiment = rng.gen_range(-1.0..1.0);

        let mut key_metrics = HashMap::new();
        key_metrics.insert("overall_sentiment".to_string(), json!(sentiment_score));
        key_metrics.insert("news_sentiment".to_string(), json!(news_sentiment));
        key_metrics.insert("social_sentiment".to_string(), json!(social_sentiment));
        key_metrics.insert("analyst_sentiment".to_string(), json!(analyst_sentiment));

        let (recommendation, confidence) = self.generate_sentiment_recommendation(sentiment_score);

        let sentiment_label = match sentiment_score {
            s if s > 0.5 => "Very Positive",
            s if s > 0.2 => "Positive",
            s if s > -0.2 => "Neutral",
            s if s > -0.5 => "Negative",
            _ => "Very Negative",
        };

        let analysis_summary = format!(
            "Sentiment analysis reveals {} market sentiment ({:.2}). \
            News coverage is {}, social media buzz is {}, and analyst opinions are {}. \
            Overall market mood suggests {} investor confidence.",
            sentiment_label.to_lowercase(),
            sentiment_score,
            if news_sentiment > 0.0 { "positive" } else { "negative" },
            if social_sentiment > 0.0 { "bullish" } else { "bearish" },
            if analyst_sentiment > 0.0 { "optimistic" } else { "cautious" },
            if sentiment_score > 0.2 { "high" } else if sentiment_score < -0.2 { "low" } else { "moderate" }
        );

        Ok(StockAnalysisResponse {
            symbol: symbol.to_string(),
            analysis_type: "sentiment".to_string(),
            recommendation,
            confidence,
            price_target: None,
            key_metrics,
            analysis_summary,
            generated_at: Utc::now(),
        })
    }

    // Portfolio Analysis
    pub async fn analyze_portfolio(&self, request: PortfolioAnalysisRequest) -> Result<PortfolioAnalysisResponse> {
        let mut total_value = request.cash_balance;
        let mut total_cost = Decimal::ZERO;
        let mut holdings_analysis = Vec::new();
        let mut daily_change = Decimal::ZERO;

        for holding in &request.holdings {
            let current_price = if let Some(price) = holding.current_price {
                price
            } else {
                // Fetch current price
                let market_data = self.get_market_data(MarketDataRequest {
                    symbols: vec![holding.symbol.clone()],
                    timeframe: Some("1d".to_string()),
                }).await?;
                market_data.data.first().map(|d| d.price).unwrap_or_default()
            };

            let current_value = current_price * holding.shares;
            let cost_basis = holding.average_cost * holding.shares;
            let unrealized_gain_loss = current_value - cost_basis;
            let unrealized_gain_loss_percent = if cost_basis > Decimal::ZERO {
                (unrealized_gain_loss / cost_basis) * Decimal::from(100)
            } else {
                Decimal::ZERO
            };

            total_value += current_value;
            total_cost += cost_basis;

            // Calculate daily change (mock for demo)
            let daily_change_amount = current_value * Decimal::from_f64_retain(rand::random::<f64>() * 0.04 - 0.02).unwrap_or_default();
            daily_change += daily_change_amount;

            let recommendation = if unrealized_gain_loss_percent > Decimal::from(20) {
                "Consider taking profits".to_string()
            } else if unrealized_gain_loss_percent < Decimal::from(-15) {
                "Monitor closely, consider averaging down".to_string()
            } else {
                "Hold".to_string()
            };

            holdings_analysis.push(HoldingAnalysis {
                symbol: holding.symbol.clone(),
                current_value,
                unrealized_gain_loss,
                unrealized_gain_loss_percent,
                weight_in_portfolio: if total_value > Decimal::ZERO { (current_value / total_value) * Decimal::from(100) } else { Decimal::ZERO },
                recommendation,
            });
        }

        let total_return = total_value - total_cost - request.cash_balance;
        let total_return_percent = if total_cost > Decimal::ZERO {
            (total_return / total_cost) * Decimal::from(100)
        } else {
            Decimal::ZERO
        };

        let daily_change_percent = if total_value > Decimal::ZERO {
            (daily_change / total_value) * Decimal::from(100)
        } else {
            Decimal::ZERO
        };

        // Calculate risk metrics
        let risk_metrics = self.calculate_portfolio_risk(&request.holdings).await;

        // Generate recommendations
        let recommendations = self.generate_portfolio_recommendations(&holdings_analysis, &risk_metrics);

        // Calculate diversification score
        let diversification_score = self.calculate_diversification_score(&holdings_analysis);

        Ok(PortfolioAnalysisResponse {
            total_value,
            total_return,
            total_return_percent,
            daily_change,
            daily_change_percent,
            holdings_analysis,
            risk_metrics,
            recommendations,
            diversification_score,
        })
    }

    // Risk Assessment
    pub async fn assess_risk(&self, request: RiskAssessmentRequest) -> Result<RiskAssessmentResponse> {
        let risk_score = self.calculate_risk_score(&request);
        let risk_profile = self.determine_risk_profile(risk_score, &request.risk_tolerance);
        let recommended_allocation = self.generate_asset_allocation(&risk_profile, request.current_age);
        
        let (expected_return, expected_volatility) = self.calculate_expected_metrics(&recommended_allocation);
        
        let investment_recommendations = self.generate_investment_recommendations(&request, &risk_profile);
        let warnings = self.generate_risk_warnings(&request, risk_score);

        Ok(RiskAssessmentResponse {
            risk_profile,
            recommended_allocation,
            expected_return,
            expected_volatility,
            investment_recommendations,
            warnings,
        })
    }

    // Helper methods for technical analysis
    async fn get_historical_prices(&self, symbol: &str, days: usize) -> Result<Vec<f64>> {
        // Mock historical data - in real implementation, fetch from API
        let base_price = match symbol {
            "AAPL" => 180.0,
            "GOOGL" => 140.0,
            "MSFT" => 380.0,
            _ => 100.0,
        };

        let mut prices = Vec::new();
        let mut current_price = base_price;
        
        for _ in 0..days {
            let change = (rand::random::<f64>() - 0.5) * 0.04; // ±2% daily change
            current_price *= 1.0 + change;
            prices.push(current_price);
        }

        Ok(prices)
    }

    fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral RSI
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..=period {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_macd(&self, prices: &[f64]) -> (f64, f64) {
        if prices.len() < 26 {
            return (0.0, 0.0);
        }

        let ema_12 = self.calculate_ema(prices, 12);
        let ema_26 = self.calculate_ema(prices, 26);
        let macd_line = ema_12 - ema_26;
        let signal_line = macd_line * 0.9; // Simplified signal line

        (macd_line, signal_line)
    }

    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];

        for &price in prices.iter().skip(1) {
            ema = (price * multiplier) + (ema * (1.0 - multiplier));
        }

        ema
    }

    fn calculate_sma(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return prices.iter().sum::<f64>() / prices.len() as f64;
        }

        prices.iter().rev().take(period).sum::<f64>() / period as f64
    }

    fn generate_technical_recommendation(&self, rsi: f64, macd: f64, signal: f64, price: f64, sma_20: f64, sma_50: f64) -> (String, f64) {
        let mut buy_signals = 0;
        let mut sell_signals = 0;

        // RSI signals
        if rsi < 30.0 {
            buy_signals += 1;
        } else if rsi > 70.0 {
            sell_signals += 1;
        }

        // MACD signals
        if macd > signal {
            buy_signals += 1;
        } else {
            sell_signals += 1;
        }

        // Moving average signals
        if price > sma_20 && sma_20 > sma_50 {
            buy_signals += 1;
        } else if price < sma_20 && sma_20 < sma_50 {
            sell_signals += 1;
        }

        let total_signals = buy_signals + sell_signals;
        let confidence = if total_signals > 0 {
            (buy_signals.max(sell_signals) as f64 / total_signals as f64) * 0.8 + 0.2
        } else {
            0.5
        };

        let recommendation = if buy_signals > sell_signals {
            "BUY".to_string()
        } else if sell_signals > buy_signals {
            "SELL".to_string()
        } else {
            "HOLD".to_string()
        };

        (recommendation, confidence)
    }

    fn generate_fundamental_recommendation(&self, pe: f64, debt_eq: f64, roe: f64, growth: f64, margin: f64) -> (String, f64) {
        let mut score = 0.0;

        // P/E ratio scoring
        if pe < 15.0 {
            score += 1.0;
        } else if pe < 25.0 {
            score += 0.5;
        }

        // Debt-to-equity scoring
        if debt_eq < 0.5 {
            score += 1.0;
        } else if debt_eq < 1.0 {
            score += 0.5;
        }

        // ROE scoring
        if roe > 20.0 {
            score += 1.0;
        } else if roe > 15.0 {
            score += 0.5;
        }

        // Growth scoring
        if growth > 10.0 {
            score += 1.0;
        } else if growth > 5.0 {
            score += 0.5;
        }

        // Margin scoring
        if margin > 20.0 {
            score += 1.0;
        } else if margin > 15.0 {
            score += 0.5;
        }

        let confidence = (score / 5.0).min(1.0);
        let recommendation = if score >= 3.5 {
            "BUY"
        } else if score >= 2.0 {
            "HOLD"
        } else {
            "SELL"
        }.to_string();

        (recommendation, confidence)
    }

    fn generate_sentiment_recommendation(&self, sentiment: f64) -> (String, f64) {
        let recommendation = if sentiment > 0.3 {
            "BUY"
        } else if sentiment < -0.3 {
            "SELL"
        } else {
            "HOLD"
        }.to_string();

        let confidence = sentiment.abs().min(1.0) * 0.7 + 0.3;

        (recommendation, confidence)
    }

    async fn calculate_portfolio_risk(&self, _holdings: &[PortfolioHolding]) -> RiskMetrics {
        // Mock risk calculations - in real implementation, would use historical data
        use rand::Rng;
        let mut rng = rand::thread_rng();

        RiskMetrics {
            beta: Some(rng.gen_range(0.8..1.5)),
            sharpe_ratio: Some(rng.gen_range(0.5..2.0)),
            volatility: rng.gen_range(15.0..35.0),
            max_drawdown: rng.gen_range(5.0..25.0),
            var_95: rng.gen_range(2.0..8.0),
            risk_score: rng.gen_range(3.0..8.0),
        }
    }

    fn generate_portfolio_recommendations(&self, holdings: &[HoldingAnalysis], risk: &RiskMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check concentration risk
        if holdings.iter().any(|h| h.weight_in_portfolio > Decimal::from(25)) {
            recommendations.push("Consider reducing position size in overweight holdings to improve diversification".to_string());
        }

        // Check risk level
        if risk.risk_score > 7.0 {
            recommendations.push("Portfolio risk is high - consider rebalancing to more conservative assets".to_string());
        }

        // Check performance
        let avg_performance: f64 = holdings.iter()
            .map(|h| h.unrealized_gain_loss_percent.to_f64().unwrap_or(0.0))
            .sum::<f64>() / holdings.len() as f64;

        if avg_performance < -10.0 {
            recommendations.push("Several positions are underperforming - review investment thesis and consider rebalancing".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Portfolio looks well-balanced - continue monitoring and consider periodic rebalancing".to_string());
        }

        recommendations
    }

    fn calculate_diversification_score(&self, holdings: &[HoldingAnalysis]) -> f64 {
        if holdings.is_empty() {
            return 0.0;
        }

        // Calculate Herfindahl-Hirschman Index for concentration
        let hhi: f64 = holdings.iter()
            .map(|h| {
                let weight = h.weight_in_portfolio.to_f64().unwrap_or(0.0) / 100.0;
                weight * weight
            })
            .sum();

        // Convert to diversification score (1 - HHI, normalized)
        ((1.0 - hhi) * 10.0).min(10.0).max(0.0)
    }

    fn calculate_risk_score(&self, request: &RiskAssessmentRequest) -> f64 {
        let mut score = 5.0; // Base score

        // Age factor
        if let Some(age) = request.current_age {
            if age < 30 {
                score += 1.0;
            } else if age > 50 {
                score -= 1.0;
            }
        }

        // Risk tolerance
        match request.risk_tolerance.as_str() {
            "conservative" => score -= 2.0,
            "moderate" => score += 0.0,
            "aggressive" => score += 2.0,
            _ => {}
        }

        // Investment horizon
        if request.investment_horizon < 5 {
            score -= 1.0;
        } else if request.investment_horizon > 15 {
            score += 1.0;
        }

        score.max(1.0).min(10.0)
    }

    fn determine_risk_profile(&self, risk_score: f64, tolerance: &str) -> String {
        match (risk_score, tolerance) {
            (s, "conservative") if s < 4.0 => "Very Conservative",
            (s, "conservative") => "Conservative",
            (s, "moderate") if s < 4.0 => "Conservative",
            (s, "moderate") if s > 7.0 => "Moderate Aggressive",
            (_, "moderate") => "Moderate",
            (s, "aggressive") if s > 7.0 => "Very Aggressive",
            (_, "aggressive") => "Aggressive",
            _ => "Moderate",
        }.to_string()
    }

    fn generate_asset_allocation(&self, risk_profile: &str, age: Option<u32>) -> HashMap<String, Decimal> {
        let mut allocation = HashMap::new();
        
        let stock_percentage = match risk_profile {
            "Very Conservative" => 20.0,
            "Conservative" => 40.0,
            "Moderate" => 60.0,
            "Moderate Aggressive" => 75.0,
            "Aggressive" => 85.0,
            "Very Aggressive" => 90.0,
            _ => 60.0,
        };

        // Age-based adjustment (rule of thumb: 100 - age = stock %)
        let adjusted_stock_percentage = if let Some(age) = age {
            let age_based = (100 - age as f64).max(20.0).min(90.0);
            (stock_percentage + age_based) / 2.0
        } else {
            stock_percentage
        };

        allocation.insert("stocks".to_string(), Decimal::from_f64_retain(adjusted_stock_percentage).unwrap_or_default());
        allocation.insert("bonds".to_string(), Decimal::from_f64_retain(100.0 - adjusted_stock_percentage).unwrap_or_default());

        allocation
    }

    fn calculate_expected_metrics(&self, allocation: &HashMap<String, Decimal>) -> (f64, f64) {
        let stock_weight = allocation.get("stocks").unwrap_or(&Decimal::ZERO).to_f64().unwrap_or(0.0) / 100.0;
        let bond_weight = allocation.get("bonds").unwrap_or(&Decimal::ZERO).to_f64().unwrap_or(0.0) / 100.0;

        // Historical averages
        let stock_return = 10.0;
        let bond_return = 4.0;
        let stock_volatility = 16.0;
        let bond_volatility = 4.0;

        let expected_return = (stock_weight * stock_return) + (bond_weight * bond_return);
        let expected_volatility = ((stock_weight * stock_volatility).powi(2) + (bond_weight * bond_volatility).powi(2)).sqrt();

        (expected_return, expected_volatility)
    }

    fn generate_investment_recommendations(&self, request: &RiskAssessmentRequest, risk_profile: &str) -> Vec<String> {
        let mut recommendations = Vec::new();

        match risk_profile {
            "Very Conservative" | "Conservative" => {
                recommendations.push("Focus on high-grade bonds and dividend-paying stocks".to_string());
                recommendations.push("Consider Treasury bonds and CDs for capital preservation".to_string());
                recommendations.push("Limit individual stock holdings to blue-chip companies".to_string());
            },
            "Moderate" => {
                recommendations.push("Balance between growth and value stocks".to_string());
                recommendations.push("Include international diversification".to_string());
                recommendations.push("Consider index funds for broad market exposure".to_string());
            },
            "Aggressive" | "Very Aggressive" => {
                recommendations.push("Focus on growth stocks and emerging markets".to_string());
                recommendations.push("Consider small-cap and technology stocks".to_string());
                recommendations.push("May include alternative investments like REITs".to_string());
            },
            _ => {
                recommendations.push("Maintain diversified portfolio across asset classes".to_string());
            }
        }

        if request.investment_horizon > 20 {
            recommendations.push("Long investment horizon allows for higher risk tolerance".to_string());
        }

        recommendations
    }

    fn generate_risk_warnings(&self, request: &RiskAssessmentRequest, risk_score: f64) -> Vec<String> {
        let mut warnings = Vec::new();

        if risk_score > 8.0 && request.risk_tolerance == "conservative" {
            warnings.push("Risk profile and tolerance mismatch - consider more conservative approach".to_string());
        }

        if request.investment_horizon < 5 && risk_score > 6.0 {
            warnings.push("Short investment horizon with high risk may lead to losses".to_string());
        }

        if let Some(age) = request.current_age {
            if age > 55 && risk_score > 7.0 {
                warnings.push("Consider reducing risk as you approach retirement age".to_string());
            }
        }

        warnings
    }
}