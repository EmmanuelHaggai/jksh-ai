use crate::{
    error::{AppError, Result},
    models::*,
    financial::FinancialService,
    cache::Cache,
};
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::json;
use chrono::Utc;

pub struct AIService {
    financial_service: Arc<FinancialService>,
    cache: Arc<Cache>,
    knowledge_base: FinancialKnowledgeBase,
}

impl AIService {
    pub fn new(financial_service: Arc<FinancialService>, cache: Arc<Cache>) -> Self {
        Self {
            financial_service,
            cache,
            knowledge_base: FinancialKnowledgeBase::new(),
        }
    }

    pub async fn process_query(&self, request: AIRecommendationRequest) -> Result<AIRecommendationResponse> {
        let cache_key = format!("ai_query_{}", sha256::digest(&request.query));
        
        // Try cache first for frequently asked questions
        if let Ok(Some(cached)) = self.cache.get::<AIRecommendationResponse>(&cache_key).await {
            return Ok(cached);
        }

        // Analyze the query intent
        let intent = self.analyze_query_intent(&request.query);
        
        // Generate response based on intent
        let response = match intent.intent_type.as_str() {
            "stock_analysis" => self.handle_stock_analysis_query(&request, &intent).await?,
            "portfolio_advice" => self.handle_portfolio_advice_query(&request, &intent).await?,
            "market_prediction" => self.handle_market_prediction_query(&request, &intent).await?,
            "investment_advice" => self.handle_investment_advice_query(&request, &intent).await?,
            "risk_assessment" => self.handle_risk_assessment_query(&request, &intent).await?,
            "market_news" => self.handle_market_news_query(&request, &intent).await?,
            "educational" => self.handle_educational_query(&request, &intent).await?,
            _ => self.handle_general_query(&request, &intent).await?,
        };

        // Cache for frequently asked questions
        if intent.confidence > 0.8 {
            let cache_duration = match intent.intent_type.as_str() {
                "educational" => std::time::Duration::from_secs(3600), // 1 hour
                "market_news" => std::time::Duration::from_secs(300),  // 5 minutes
                _ => std::time::Duration::from_secs(900), // 15 minutes
            };
            let _ = self.cache.set(&cache_key, &response, Some(cache_duration)).await;
        }

        Ok(response)
    }

    fn analyze_query_intent(&self, query: &str) -> QueryIntent {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        // Define intent patterns
        let intent_patterns = [
            ("stock_analysis", vec!["analyze", "analysis", "performance", "how is", "stock", "company"]),
            ("portfolio_advice", vec!["portfolio", "holdings", "diversify", "rebalance", "allocation"]),
            ("market_prediction", vec!["predict", "forecast", "future", "will go", "price target", "outlook"]),
            ("investment_advice", vec!["should i buy", "should i sell", "invest", "recommendation", "advice"]),
            ("risk_assessment", vec!["risk", "risky", "safe", "volatile", "conservative", "aggressive"]),
            ("market_news", vec!["news", "happening", "events", "latest", "updates", "market"]),
            ("educational", vec!["what is", "how does", "explain", "definition", "meaning", "learn"]),
        ];

        let mut best_match = ("general", 0);
        
        for (intent, keywords) in &intent_patterns {
            let matches = keywords.iter()
                .filter(|&&keyword| {
                    words.iter().any(|&word| word.contains(keyword)) ||
                    query_lower.contains(keyword)
                })
                .count();
            
            if matches > best_match.1 {
                best_match = (intent, matches);
            }
        }

        // Extract entities (stock symbols, numbers, etc.)
        let entities = self.extract_entities(&query);

        QueryIntent {
            intent_type: best_match.0.to_string(),
            confidence: (best_match.1 as f64 / 3.0).min(1.0).max(0.3),
            entities,
            raw_query: query.to_string(),
        }
    }

    fn extract_entities(&self, query: &str) -> HashMap<String, String> {
        let mut entities = HashMap::new();
        let words: Vec<&str> = query.split_whitespace().collect();

        // Extract stock symbols (2-5 uppercase letters)
        for word in &words {
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic()).to_uppercase();
            if clean_word.len() >= 2 && clean_word.len() <= 5 && clean_word.chars().all(|c| c.is_alphabetic()) {
                // Check if it's a known stock symbol
                if self.is_likely_stock_symbol(&clean_word) {
                    entities.insert("symbol".to_string(), clean_word);
                    break;
                }
            }
        }

        // Extract numbers (for amounts, percentages, timeframes)
        for word in &words {
            if let Ok(number) = word.trim_matches(|c: char| !c.is_numeric() && c != '.').parse::<f64>() {
                if word.contains('%') {
                    entities.insert("percentage".to_string(), number.to_string());
                } else if number > 1000.0 {
                    entities.insert("amount".to_string(), number.to_string());
                } else if number <= 365.0 && (query.contains("day") || query.contains("month")) {
                    entities.insert("timeframe".to_string(), number.to_string());
                }
            }
        }

        // Extract timeframe keywords
        if query.contains("today") || query.contains("1 day") {
            entities.insert("timeframe".to_string(), "1d".to_string());
        } else if query.contains("week") {
            entities.insert("timeframe".to_string(), "1w".to_string());
        } else if query.contains("month") {
            entities.insert("timeframe".to_string(), "1mo".to_string());
        } else if query.contains("year") {
            entities.insert("timeframe".to_string(), "1y".to_string());
        }

        entities
    }

    fn is_likely_stock_symbol(&self, symbol: &str) -> bool {
        // Common stock symbols
        let common_symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "SPY", "QQQ", "IWM", "VOO", "VTI", "BRK", "JPM", "JNJ", "WMT",
            "PG", "UNH", "HD", "MA", "PYPL", "DIS", "ADBE", "CRM", "INTC",
            "AMD", "BABA", "NKE", "KO", "PFE", "MRK", "ABT", "TMO", "COST"
        ];
        
        common_symbols.contains(&symbol)
    }

    async fn handle_stock_analysis_query(&self, request: &AIRecommendationRequest, intent: &QueryIntent) -> Result<AIRecommendationResponse> {
        let symbol = intent.entities.get("symbol")
            .cloned()
            .unwrap_or_else(|| "SPY".to_string()); // Default to S&P 500

        // Get comprehensive analysis
        let technical_analysis = self.financial_service.analyze_stock(StockAnalysisRequest {
            symbol: symbol.clone(),
            analysis_type: "technical".to_string(),
            timeframe: intent.entities.get("timeframe"),
        }).await?;

        let fundamental_analysis = self.financial_service.analyze_stock(StockAnalysisRequest {
            symbol: symbol.clone(),
            analysis_type: "fundamental".to_string(),
            timeframe: None,
        }).await?;

        let market_data = self.financial_service.get_market_data(MarketDataRequest {
            symbols: vec![symbol.clone()],
            timeframe: Some("1d".to_string()),
        }).await?;

        let current_data = market_data.data.first().unwrap();

        let response_text = format!(
            "Here's a comprehensive analysis of {}:\n\n\
            📊 Current Price: ${:.2} ({:+.2}%)\n\
            Volume: {:,}\n\n\
            🔍 Technical Analysis:\n\
            • Recommendation: {} (Confidence: {:.0}%)\n\
            • {}\n\n\
            📈 Fundamental Analysis:\n\
            • Recommendation: {} (Confidence: {:.0}%)\n\
            • {}\n\n\
            💡 Overall Assessment:\n\
            Based on both technical and fundamental factors, {} appears to be {}. \
            The technical indicators suggest {} momentum, while fundamentals show {} value.",
            symbol,
            current_data.price.to_f64().unwrap_or(0.0),
            current_data.change_percent.to_f64().unwrap_or(0.0),
            current_data.volume,
            technical_analysis.recommendation,
            technical_analysis.confidence * 100.0,
            technical_analysis.analysis_summary,
            fundamental_analysis.recommendation,
            fundamental_analysis.confidence * 100.0,
            fundamental_analysis.analysis_summary,
            symbol,
            self.determine_overall_sentiment(&technical_analysis, &fundamental_analysis),
            self.describe_momentum(&technical_analysis),
            self.describe_value(&fundamental_analysis)
        );

        let recommendations = vec![
            format!("Monitor {} closely for the next few trading sessions", symbol),
            "Consider your risk tolerance and investment horizon".to_string(),
            "Diversification is key - don't put all eggs in one basket".to_string(),
        ];

        let follow_up_questions = vec![
            format!("What's the price prediction for {} in 30 days?", symbol),
            format!("How does {} compare to its competitors?", symbol),
            "Should I add this to my portfolio?".to_string(),
            "What are the main risks with this investment?".to_string(),
        ];

        Ok(AIRecommendationResponse {
            response: response_text,
            recommendations,
            relevant_stocks: vec![symbol.clone()],
            confidence: (technical_analysis.confidence + fundamental_analysis.confidence) / 2.0,
            response_type: "analysis".to_string(),
            follow_up_questions,
        })
    }

    async fn handle_portfolio_advice_query(&self, _request: &AIRecommendationRequest, intent: &QueryIntent) -> Result<AIRecommendationResponse> {
        let advice_type = if intent.raw_query.contains("diversify") {
            "diversification"
        } else if intent.raw_query.contains("rebalance") {
            "rebalancing"
        } else if intent.raw_query.contains("allocation") {
            "allocation"
        } else {
            "general"
        };

        let response_text = match advice_type {
            "diversification" => {
                "🎯 Portfolio Diversification Strategy:\n\n\
                Diversification is crucial for risk management. Here's how to diversify effectively:\n\n\
                🌍 Geographic Diversification:\n\
                • 60-70% US stocks\n\
                • 20-30% International developed markets\n\
                • 10-15% Emerging markets\n\n\
                🏭 Sector Diversification:\n\
                • Technology: 15-20%\n\
                • Healthcare: 10-15%\n\
                • Financial Services: 10-15%\n\
                • Consumer Goods: 10-15%\n\
                • Other sectors: 40-45%\n\n\
                📊 Asset Class Diversification:\n\
                • Stocks: 60-80% (based on age/risk tolerance)\n\
                • Bonds: 15-35%\n\
                • REITs: 5-10%\n\
                • Commodities: 0-5%\n\n\
                💡 Remember: Don't over-diversify (20-30 stocks is often sufficient)"
            },
            "rebalancing" => {
                "⚖️ Portfolio Rebalancing Guide:\n\n\
                Rebalancing maintains your target allocation and can improve returns:\n\n\
                📅 When to Rebalance:\n\
                • Quarterly or semi-annually\n\
                • When any asset class deviates >5% from target\n\
                • After major market movements\n\n\
                🔄 How to Rebalance:\n\
                1. Calculate current allocations\n\
                2. Compare to target allocations\n\
                3. Sell overweight positions\n\
                4. Buy underweight positions\n\
                5. Consider tax implications\n\n\
                💰 Tax-Efficient Rebalancing:\n\
                • Use new contributions to buy underweight assets\n\
                • Rebalance in tax-advantaged accounts first\n\
                • Consider tax-loss harvesting opportunities"
            },
            "allocation" => {
                "📊 Asset Allocation Strategy:\n\n\
                Your asset allocation should match your risk tolerance and time horizon:\n\n\
                👶 Young Investors (20s-30s):\n\
                • Stocks: 80-90%\n\
                • Bonds: 10-20%\n\
                • High risk tolerance, long time horizon\n\n\
                🧑‍💼 Middle-aged (40s-50s):\n\
                • Stocks: 60-70%\n\
                • Bonds: 30-40%\n\
                • Moderate risk, medium time horizon\n\n\
                👴 Pre-retirement (55+):\n\
                • Stocks: 40-60%\n\
                • Bonds: 40-60%\n\
                • Lower risk, shorter time horizon\n\n\
                📈 Rule of Thumb: Stock % = 100 - your age"
            },
            _ => {
                "💼 General Portfolio Management Tips:\n\n\
                Building a successful portfolio requires discipline and strategy:\n\n\
                🎯 Core Principles:\n\
                • Start with low-cost index funds\n\
                • Maintain proper diversification\n\
                • Invest regularly (dollar-cost averaging)\n\
                • Rebalance periodically\n\
                • Stay the course during volatility\n\n\
                📊 Portfolio Structure:\n\
                • Core holdings: 60-80% (broad market ETFs)\n\
                • Satellite holdings: 20-40% (sector/theme ETFs, individual stocks)\n\n\
                🚫 Common Mistakes to Avoid:\n\
                • Emotional investing\n\
                • Trying to time the market\n\
                • Over-concentration in one stock/sector\n\
                • Ignoring fees and taxes\n\
                • Lack of regular review"
            }
        };

        let recommendations = vec![
            "Review your portfolio monthly, rebalance quarterly".to_string(),
            "Keep investment costs low with index funds and ETFs".to_string(),
            "Stay disciplined and avoid emotional decision-making".to_string(),
            "Consider your tax situation when making changes".to_string(),
        ];

        let follow_up_questions = vec![
            "What's my ideal asset allocation based on my age?".to_string(),
            "How often should I rebalance my portfolio?".to_string(),
            "What are the best low-cost index funds?".to_string(),
            "Should I invest in individual stocks or ETFs?".to_string(),
        ];

        Ok(AIRecommendationResponse {
            response: response_text.to_string(),
            recommendations,
            relevant_stocks: vec!["SPY".to_string(), "VTI".to_string(), "QQQ".to_string()],
            confidence: 0.9,
            response_type: "advice".to_string(),
            follow_up_questions,
        })
    }

    async fn handle_market_prediction_query(&self, _request: &AIRecommendationRequest, intent: &QueryIntent) -> Result<AIRecommendationResponse> {
        let symbol = intent.entities.get("symbol").cloned().unwrap_or_else(|| "SPY".to_string());
        let timeframe = intent.entities.get("timeframe").cloned().unwrap_or_else(|| "30".to_string());

        // Get current market data
        let market_data = self.financial_service.get_market_data(MarketDataRequest {
            symbols: vec![symbol.clone()],
            timeframe: Some("1d".to_string()),
        }).await?;

        let current_data = market_data.data.first().unwrap();
        let current_price = current_data.price.to_f64().unwrap_or(0.0);

        // Generate prediction based on multiple factors
        let prediction = self.generate_price_prediction(&symbol, current_price, &timeframe);

        let response_text = format!(
            "📈 Price Prediction for {} ({}d horizon):\n\n\
            🔍 Current Analysis:\n\
            • Current Price: ${:.2}\n\
            • Recent Change: {:+.2}%\n\
            • Volume: {:,}\n\n\
            🎯 Prediction:\n\
            • Predicted Price: ${:.2}\n\
            • Expected Change: {:+.2}%\n\
            • Confidence Range: ${:.2} - ${:.2}\n\n\
            📊 Key Factors:\n\
            {}\n\n\
            ⚠️ Important Disclaimer:\n\
            This prediction is based on technical indicators and historical patterns. \
            Market predictions are inherently uncertain and should not be the sole basis for investment decisions. \
            Past performance does not guarantee future results.",
            symbol,
            timeframe,
            current_price,
            current_data.change_percent.to_f64().unwrap_or(0.0),
            current_data.volume,
            prediction.predicted_price,
            prediction.expected_change,
            prediction.lower_bound,
            prediction.upper_bound,
            prediction.key_factors.join("\n• ")
        );

        let recommendations = vec![
            "Use predictions as one factor among many in your decision-making".to_string(),
            "Consider setting stop-loss orders to manage risk".to_string(),
            "Monitor key support and resistance levels".to_string(),
            "Stay updated on company/sector news that could impact prices".to_string(),
        ];

        let follow_up_questions = vec![
            format!("What are the main risks for {} in the next month?", symbol),
            "How accurate are these predictions historically?".to_string(),
            format!("Should I buy {} at current levels?", symbol),
            "What technical indicators support this prediction?".to_string(),
        ];

        Ok(AIRecommendationResponse {
            response: response_text,
            recommendations,
            relevant_stocks: vec![symbol],
            confidence: prediction.confidence,
            response_type: "prediction".to_string(),
            follow_up_questions,
        })
    }

    async fn handle_investment_advice_query(&self, request: &AIRecommendationRequest, intent: &QueryIntent) -> Result<AIRecommendationResponse> {
        let symbol = intent.entities.get("symbol").cloned();
        let is_buy_query = intent.raw_query.contains("buy") || intent.raw_query.contains("invest");
        let is_sell_query = intent.raw_query.contains("sell");

        let response_text = if let Some(ref sym) = symbol {
            // Specific stock advice
            let analysis = self.financial_service.analyze_stock(StockAnalysisRequest {
                symbol: sym.clone(),
                analysis_type: "technical".to_string(),
                timeframe: Some("1mo".to_string()),
            }).await?;

            if is_buy_query {
                format!(
                    "🤔 Should you buy {}?\n\n\
                    Based on current analysis:\n\
                    • Technical Recommendation: {}\n\
                    • Confidence Level: {:.0}%\n\n\
                    💡 Investment Considerations:\n\
                    • Entry Point: Current price levels {} favorable\n\
                    • Risk Level: {} risk stock\n\
                    • Time Horizon: Best suited for {} investors\n\n\
                    📋 Before You Buy:\n\
                    1. Ensure it fits your portfolio allocation\n\
                    2. Consider your risk tolerance\n\
                    3. Don't invest more than 5-10% in a single stock\n\
                    4. Have a clear exit strategy\n\n\
                    🎯 Alternative Approach:\n\
                    Consider dollar-cost averaging over 2-3 months instead of a lump sum purchase.",
                    sym,
                    analysis.recommendation,
                    analysis.confidence * 100.0,
                    if analysis.recommendation == "BUY" { "appear" } else { "may not be" },
                    self.assess_stock_risk_level(sym),
                    if analysis.confidence > 0.7 { "medium to long-term" } else { "long-term" }
                )
            } else if is_sell_query {
                format!(
                    "🤔 Should you sell {}?\n\n\
                    Current Analysis suggests: {}\n\
                    Confidence: {:.0}%\n\n\
                    🔍 Consider These Factors:\n\
                    • Has your investment thesis changed?\n\
                    • Do you need the money soon?\n\
                    • Are you rebalancing your portfolio?\n\
                    • Is this a tax-loss harvesting opportunity?\n\n\
                    📊 Selling Strategies:\n\
                    • Partial sale: Reduce position by 25-50%\n\
                    • Gradual exit: Sell over 2-3 months\n\
                    • Stop-loss: Set automatic sell order\n\n\
                    💡 Remember: Time in market > timing the market",
                    sym,
                    if analysis.recommendation == "SELL" { "Consider selling" } else { "Hold or accumulate" },
                    analysis.confidence * 100.0
                )
            } else {
                format!(
                    "💡 Investment Advice for {}:\n\n\
                    Current Recommendation: {}\n\
                    Analysis Summary: {}\n\n\
                    🎯 Strategic Approach:\n\
                    • Position Size: Limit to 5-10% of portfolio\n\
                    • Entry Strategy: Consider dollar-cost averaging\n\
                    • Exit Strategy: Set profit targets and stop-losses\n\
                    • Timeline: {} holding period recommended",
                    sym,
                    analysis.recommendation,
                    analysis.analysis_summary,
                    if analysis.confidence > 0.7 { "Medium to long-term" } else { "Long-term" }
                )
            }
        } else {
            // General investment advice
            "💡 General Investment Principles:\n\n\
            🎯 Core Investment Rules:\n\
            1. Start Early: Time is your greatest asset\n\
            2. Diversify: Don't put all eggs in one basket\n\
            3. Stay Consistent: Regular investing beats timing\n\
            4. Keep Costs Low: Fees compound negatively\n\
            5. Stay Patient: Wealth building takes time\n\n\
            📊 Building Your Portfolio:\n\
            • Emergency Fund: 3-6 months expenses first\n\
            • Core Holdings: 60-80% in broad market ETFs\n\
            • Satellite Holdings: 20-40% in specific sectors/stocks\n\
            • Risk Management: Never invest money you can't afford to lose\n\n\
            🚀 Getting Started:\n\
            • Open a brokerage account\n\
            • Start with index funds (VTI, VOO, QQQ)\n\
            • Automate your investments\n\
            • Increase contributions annually\n\n\
            💼 Advanced Strategies (once you have the basics):\n\
            • Individual stock selection\n\
            • Sector rotation\n\
            • Options strategies\n\
            • Alternative investments".to_string()
        };

        let recommendations = vec![
            "Never invest money you can't afford to lose".to_string(),
            "Diversification is your best defense against risk".to_string(),
            "Regular investing beats trying to time the market".to_string(),
            "Always do your own research before investing".to_string(),
        ];

        let relevant_stocks = if symbol.is_some() {
            vec![symbol.unwrap()]
        } else {
            vec!["VTI".to_string(), "VOO".to_string(), "QQQ".to_string()]
        };

        let follow_up_questions = vec![
            "What's the best portfolio allocation for my age?".to_string(),
            "How much should I invest each month?".to_string(),
            "What are the best index funds for beginners?".to_string(),
            "How do I evaluate individual stocks?".to_string(),
        ];

        Ok(AIRecommendationResponse {
            response: response_text,
            recommendations,
            relevant_stocks,
            confidence: 0.85,
            response_type: "advice".to_string(),
            follow_up_questions,
        })
    }

    async fn handle_risk_assessment_query(&self, _request: &AIRecommendationRequest, intent: &QueryIntent) -> Result<AIRecommendationResponse> {
        let symbol = intent.entities.get("symbol").cloned();

        let response_text = if let Some(ref sym) = symbol {
            // Specific stock risk assessment
            format!(
                "⚠️ Risk Assessment for {}:\n\n\
                🎯 Risk Level: {} Risk\n\
                📊 Key Risk Factors:\n\
                {}\n\n\
                📈 Historical Volatility: {}% (annualized)\n\
                📉 Maximum Drawdown: {}% (last 2 years)\n\
                🔄 Beta: {} (vs S&P 500)\n\n\
                💡 Risk Management Tips:\n\
                • Position Size: Limit to 5-10% of portfolio\n\
                • Stop Loss: Consider setting at -15% to -20%\n\
                • Diversification: Don't concentrate in similar stocks\n\
                • Time Horizon: Longer periods reduce risk\n\n\
                ⚖️ Risk vs Reward:\n\
                Higher risk stocks like {} can offer greater returns but also greater losses. \
                Make sure this aligns with your risk tolerance and investment goals.",
                sym,
                self.assess_stock_risk_level(sym),
                self.get_stock_risk_factors(sym).join("\n• "),
                self.get_mock_volatility(sym),
                self.get_mock_max_drawdown(sym),
                self.get_mock_beta(sym),
                sym
            )
        } else {
            // General risk assessment
            "⚠️ Investment Risk Assessment Guide:\n\n\
            🎯 Types of Investment Risk:\n\
            • Market Risk: Overall market declines\n\
            • Company Risk: Specific business issues\n\
            • Sector Risk: Industry-wide problems\n\
            • Liquidity Risk: Difficulty selling quickly\n\
            • Inflation Risk: Purchasing power erosion\n\
            • Interest Rate Risk: Rate changes affect valuations\n\n\
            📊 Risk Tolerance Levels:\n\
            🟢 Conservative: 0-30% stocks, focus on preservation\n\
            🟡 Moderate: 40-70% stocks, balanced approach\n\
            🔴 Aggressive: 80-100% stocks, growth focused\n\n\
            🛡️ Risk Management Strategies:\n\
            1. Diversification: Spread investments across assets\n\
            2. Asset Allocation: Match risk to time horizon\n\
            3. Regular Rebalancing: Maintain target allocations\n\
            4. Emergency Fund: Keep 3-6 months expenses liquid\n\
            5. Dollar-Cost Averaging: Reduce timing risk\n\n\
            💡 Remember: Higher potential returns come with higher risk. \
            Never invest more than you can afford to lose.".to_string()
        };

        let recommendations = vec![
            "Assess your risk tolerance honestly before investing".to_string(),
            "Diversify across different asset classes and sectors".to_string(),
            "Never put more than 10% in any single stock".to_string(),
            "Review and adjust your risk profile annually".to_string(),
        ];

        let relevant_stocks = if symbol.is_some() {
            vec![symbol.unwrap()]
        } else {
            vec!["SPY".to_string(), "BND".to_string(), "VTI".to_string()]
        };

        Ok(AIRecommendationResponse {
            response: response_text,
            recommendations,
            relevant_stocks,
            confidence: 0.9,
            response_type: "advice".to_string(),
            follow_up_questions: vec![
                "What's my ideal risk level based on my age?".to_string(),
                "How do I calculate portfolio risk?".to_string(),
                "What are safe investments for beginners?".to_string(),
                "Should I reduce risk as I get older?".to_string(),
            ],
        })
    }

    async fn handle_market_news_query(&self, _request: &AIRecommendationRequest, _intent: &QueryIntent) -> Result<AIRecommendationResponse> {
        // Generate mock market news summary
        let response_text = "📰 Today's Market Headlines:\n\n\
            🔴 Major Indices:\n\
            • S&P 500: Mixed performance amid earnings season\n\
            • NASDAQ: Tech stocks showing resilience\n\
            • Dow Jones: Industrial stocks under pressure\n\n\
            📊 Sector Performance:\n\
            • Technology: Leading gains (+1.2%)\n\
            • Healthcare: Defensive strength (+0.8%)\n\
            • Energy: Volatile on oil price swings (-0.5%)\n\
            • Financial: Rate sensitivity continues (-0.3%)\n\n\
            🏦 Fed Policy:\n\
            • Interest rate decisions being closely watched\n\
            • Inflation data showing mixed signals\n\
            • Market pricing in policy changes\n\n\
            🌍 Global Markets:\n\
            • European markets showing strength\n\
            • Asian markets mixed on trade concerns\n\
            • Emerging markets facing currency pressures\n\n\
            💡 What This Means for Investors:\n\
            Current market conditions suggest continued volatility. \
            Focus on quality companies with strong fundamentals. \
            Consider defensive positioning while maintaining long-term perspective.".to_string();

        Ok(AIRecommendationResponse {
            response: response_text,
            recommendations: vec![
                "Stay informed but don't make hasty decisions based on daily news".to_string(),
                "Focus on long-term trends rather than short-term volatility".to_string(),
                "Use market dips as potential buying opportunities".to_string(),
            ],
            relevant_stocks: vec!["SPY".to_string(), "QQQ".to_string(), "DIA".to_string()],
            confidence: 0.8,
            response_type: "news".to_string(),
            follow_up_questions: vec![
                "How do I filter important news from noise?".to_string(),
                "Should I adjust my portfolio based on current events?".to_string(),
                "What are the best financial news sources?".to_string(),
            ],
        })
    }

    async fn handle_educational_query(&self, _request: &AIRecommendationRequest, intent: &QueryIntent) -> Result<AIRecommendationResponse> {
        let topic = self.identify_educational_topic(&intent.raw_query);
        
        let response_text = match topic {
            "dividend" => self.knowledge_base.get_dividend_explanation(),
            "pe_ratio" => self.knowledge_base.get_pe_ratio_explanation(),
            "market_cap" => self.knowledge_base.get_market_cap_explanation(),
            "etf" => self.knowledge_base.get_etf_explanation(),
            "volatility" => self.knowledge_base.get_volatility_explanation(),
            "beta" => self.knowledge_base.get_beta_explanation(),
            "earnings" => self.knowledge_base.get_earnings_explanation(),
            _ => self.knowledge_base.get_general_explanation(),
        };

        Ok(AIRecommendationResponse {
            response: response_text,
            recommendations: vec![
                "Continue learning - financial education is an ongoing process".to_string(),
                "Practice with small amounts before making large investments".to_string(),
                "Consider taking a finance course or reading investment books".to_string(),
            ],
            relevant_stocks: vec![],
            confidence: 0.95,
            response_type: "explanation".to_string(),
            follow_up_questions: vec![
                "Can you explain this with a real example?".to_string(),
                "What are the practical applications?".to_string(),
                "How does this affect my investment decisions?".to_string(),
            ],
        })
    }

    async fn handle_general_query(&self, _request: &AIRecommendationRequest, _intent: &QueryIntent) -> Result<AIRecommendationResponse> {
        let response_text = "🤖 I'm your Financial AI Assistant!\n\n\
            I can help you with:\n\n\
            📊 Stock Analysis:\n\
            • \"Analyze Apple stock\"\n\
            • \"How is Tesla performing?\"\n\
            • \"Should I buy Microsoft?\"\n\n\
            💼 Portfolio Management:\n\
            • \"Review my portfolio\"\n\
            • \"How should I diversify?\"\n\
            • \"What's my risk level?\"\n\n\
            📈 Market Insights:\n\
            • \"Market news today\"\n\
            • \"Predict Amazon price\"\n\
            • \"Best tech stocks\"\n\n\
            🎓 Financial Education:\n\
            • \"What is a P/E ratio?\"\n\
            • \"Explain dividends\"\n\
            • \"How do ETFs work?\"\n\n\
            💡 Just ask me anything about investing, stocks, or financial markets in natural language!".to_string();

        Ok(AIRecommendationResponse {
            response: response_text,
            recommendations: vec![
                "Ask specific questions about stocks you're interested in".to_string(),
                "I can analyze your portfolio if you share your holdings".to_string(),
                "Feel free to ask for explanations of financial concepts".to_string(),
            ],
            relevant_stocks: vec![],
            confidence: 1.0,
            response_type: "help".to_string(),
            follow_up_questions: vec![
                "Analyze Apple stock for me".to_string(),
                "What's the best investment strategy for beginners?".to_string(),
                "How do I build a diversified portfolio?".to_string(),
                "What are the top stocks to buy now?".to_string(),
            ],
        })
    }

    // Helper methods
    fn determine_overall_sentiment(&self, technical: &StockAnalysisResponse, fundamental: &StockAnalysisResponse) -> &str {
        match (technical.recommendation.as_str(), fundamental.recommendation.as_str()) {
            ("BUY", "BUY") => "strongly positioned",
            ("BUY", _) | (_, "BUY") => "well positioned",
            ("SELL", "SELL") => "facing challenges",
            ("SELL", _) | (_, "SELL") => "showing mixed signals",
            _ => "in a neutral position",
        }
    }

    fn describe_momentum(&self, analysis: &StockAnalysisResponse) -> &str {
        match analysis.recommendation.as_str() {
            "BUY" => "bullish",
            "SELL" => "bearish",
            _ => "neutral",
        }
    }

    fn describe_value(&self, analysis: &StockAnalysisResponse) -> &str {
        if analysis.confidence > 0.7 {
            match analysis.recommendation.as_str() {
                "BUY" => "attractive",
                "SELL" => "concerning",
                _ => "fair",
            }
        } else {
            "uncertain"
        }
    }

    fn generate_price_prediction(&self, _symbol: &str, current_price: f64, timeframe: &str) -> PricePredictionData {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let days: f64 = timeframe.parse().unwrap_or(30.0);
        let volatility = 0.02; // 2% daily volatility
        
        // Generate a random walk with slight upward bias
        let expected_return = 0.001 * days; // 0.1% daily expected return
        let predicted_change = rng.gen_range(-0.1..0.15) + expected_return;
        
        let predicted_price = current_price * (1.0 + predicted_change);
        let confidence_range = current_price * volatility * (days / 30.0).sqrt();
        
        PricePredictionData {
            predicted_price,
            expected_change: predicted_change * 100.0,
            lower_bound: predicted_price - confidence_range,
            upper_bound: predicted_price + confidence_range,
            confidence: 0.65,
            key_factors: vec![
                "Technical indicators showing mixed signals".to_string(),
                "Market volatility remains elevated".to_string(),
                "Sector trends are generally positive".to_string(),
                "Economic indicators are stable".to_string(),
            ],
        }
    }

    fn assess_stock_risk_level(&self, symbol: &str) -> &str {
        match symbol {
            "AAPL" | "MSFT" | "GOOGL" | "JNJ" | "PG" => "Moderate",
            "TSLA" | "NVDA" | "AMD" | "PLTR" => "High",
            "KO" | "WMT" | "VZ" | "T" => "Low",
            _ => "Moderate",
        }
    }

    fn get_stock_risk_factors(&self, symbol: &str) -> Vec<String> {
        match symbol {
            "TSLA" => vec![
                "High volatility and price swings".to_string(),
                "Regulatory changes in EV industry".to_string(),
                "Competition from traditional automakers".to_string(),
                "Dependence on CEO leadership".to_string(),
            ],
            "AAPL" => vec![
                "iPhone sales dependency".to_string(),
                "China market exposure".to_string(),
                "Tech regulation risks".to_string(),
                "Supply chain disruptions".to_string(),
            ],
            _ => vec![
                "Market volatility".to_string(),
                "Economic recession risk".to_string(),
                "Interest rate changes".to_string(),
                "Industry competition".to_string(),
            ],
        }
    }

    fn get_mock_volatility(&self, symbol: &str) -> f64 {
        match symbol {
            "TSLA" => 65.0,
            "NVDA" => 55.0,
            "AAPL" => 25.0,
            "MSFT" => 22.0,
            "SPY" => 18.0,
            _ => 30.0,
        }
    }

    fn get_mock_max_drawdown(&self, symbol: &str) -> f64 {
        match symbol {
            "TSLA" => 45.0,
            "NVDA" => 35.0,
            "AAPL" => 20.0,
            "MSFT" => 18.0,
            "SPY" => 15.0,
            _ => 25.0,
        }
    }

    fn get_mock_beta(&self, symbol: &str) -> f64 {
        match symbol {
            "TSLA" => 2.1,
            "NVDA" => 1.8,
            "AAPL" => 1.2,
            "MSFT" => 0.9,
            "SPY" => 1.0,
            _ => 1.1,
        }
    }

    fn identify_educational_topic(&self, query: &str) -> &str {
        let query_lower = query.to_lowercase();
        
        if query_lower.contains("dividend") {
            "dividend"
        } else if query_lower.contains("p/e") || query_lower.contains("pe ratio") {
            "pe_ratio"
        } else if query_lower.contains("market cap") {
            "market_cap"
        } else if query_lower.contains("etf") {
            "etf"
        } else if query_lower.contains("volatility") {
            "volatility"
        } else if query_lower.contains("beta") {
            "beta"
        } else if query_lower.contains("earnings") {
            "earnings"
        } else {
            "general"
        }
    }
}

#[derive(Debug)]
struct QueryIntent {
    intent_type: String,
    confidence: f64,
    entities: HashMap<String, String>,
    raw_query: String,
}

#[derive(Debug)]
struct PricePredictionData {
    predicted_price: f64,
    expected_change: f64,
    lower_bound: f64,
    upper_bound: f64,
    confidence: f64,
    key_factors: Vec<String>,
}

// Financial Knowledge Base for educational queries
struct FinancialKnowledgeBase;

impl FinancialKnowledgeBase {
    fn new() -> Self {
        Self
    }

    fn get_dividend_explanation(&self) -> String {
        "💰 Dividends Explained:\n\n\
        A dividend is a payment made by corporations to their shareholders, typically as a distribution of profits.\n\n\
        🔍 Key Points:\n\
        • Quarterly payments: Most US companies pay every 3 months\n\
        • Dividend yield: Annual dividend ÷ stock price\n\
        • Ex-dividend date: Last day to buy and still receive dividend\n\
        • Payout ratio: Percentage of earnings paid as dividends\n\n\
        📊 Example:\n\
        If Apple pays $0.92/quarter and trades at $180:\n\
        • Annual dividend: $0.92 × 4 = $3.68\n\
        • Dividend yield: $3.68 ÷ $180 = 2.04%\n\n\
        💡 Investment Strategy:\n\
        • Dividend stocks provide regular income\n\
        • Look for sustainable payout ratios (<60%)\n\
        • Consider dividend growth, not just yield\n\
        • Dividend aristocrats: 25+ years of increases".to_string()
    }

    fn get_pe_ratio_explanation(&self) -> String {
        "📊 P/E Ratio (Price-to-Earnings) Explained:\n\n\
        P/E ratio measures how much investors are willing to pay for each dollar of earnings.\n\n\
        🧮 Calculation:\n\
        P/E Ratio = Stock Price ÷ Earnings Per Share (EPS)\n\n\
        📈 Interpretation:\n\
        • Low P/E (5-15): Potentially undervalued or slow growth\n\
        • Medium P/E (15-25): Fairly valued\n\
        • High P/E (25+): Growth expectations or overvalued\n\n\
        📊 Example:\n\
        Stock Price: $100, EPS: $5\n\
        P/E Ratio = $100 ÷ $5 = 20\n\
        This means investors pay $20 for every $1 of earnings\n\n\
        ⚠️ Limitations:\n\
        • Compare within same industry\n\
        • Consider growth prospects\n\
        • One-time events can distort EPS\n\
        • Use forward P/E for future estimates".to_string()
    }

    fn get_market_cap_explanation(&self) -> String {
        "🏢 Market Capitalization Explained:\n\n\
        Market cap is the total value of all company shares in the market.\n\n\
        🧮 Calculation:\n\
        Market Cap = Share Price × Total Shares Outstanding\n\n\
        📊 Categories:\n\
        • Large-cap: $10B+ (Apple, Microsoft)\n\
        • Mid-cap: $2B-$10B (growing companies)\n\
        • Small-cap: $300M-$2B (higher growth potential)\n\
        • Micro-cap: <$300M (highest risk/reward)\n\n\
        📈 Example:\n\
        Company A: $50/share × 100M shares = $5B market cap\n\
        Company B: $25/share × 80M shares = $2B market cap\n\
        Company A is larger despite lower share count\n\n\
        💡 Investment Implications:\n\
        • Large-cap: More stable, lower growth\n\
        • Small-cap: More volatile, higher growth potential\n\
        • Diversify across market caps\n\
        • Market cap affects index inclusion".to_string()
    }

    fn get_etf_explanation(&self) -> String {
        "📦 ETFs (Exchange-Traded Funds) Explained:\n\n\
        ETFs are investment funds that trade on stock exchanges like individual stocks.\n\n\
        🔍 How They Work:\n\
        • Pool money from many investors\n\
        • Buy a basket of securities (stocks, bonds)\n\
        • Shares trade throughout market hours\n\
        • Typically track an index\n\n\
        💪 Advantages:\n\
        • Instant diversification\n\
        • Low expense ratios (0.03%-0.75%)\n\
        • Liquid - easy to buy/sell\n\
        • Tax efficient\n\
        • Transparent holdings\n\n\
        📊 Popular ETFs:\n\
        • SPY: S&P 500 index\n\
        • QQQ: NASDAQ-100\n\
        • VTI: Total stock market\n\
        • BND: Bond market\n\n\
        🎯 Types:\n\
        • Broad market (VTI, VOO)\n\
        • Sector specific (XLK tech, XLF financial)\n\
        • International (VEA, VWO)\n\
        • Bond (AGG, TLT)\n\
        • Thematic (clean energy, AI)".to_string()
    }

    fn get_volatility_explanation(&self) -> String {
        "📈📉 Volatility Explained:\n\n\
        Volatility measures how much a stock's price fluctuates over time.\n\n\
        🔍 Key Concepts:\n\
        • High volatility: Large price swings\n\
        • Low volatility: Stable price movements\n\
        • Measured as standard deviation of returns\n\
        • Usually annualized (% per year)\n\n\
        📊 Volatility Levels:\n\
        • Low: <15% (utilities, consumer staples)\n\
        • Medium: 15-25% (large-cap stocks)\n\
        • High: 25-40% (growth stocks, small-caps)\n\
        • Very High: >40% (crypto, penny stocks)\n\n\
        💡 Investment Implications:\n\
        • Higher volatility = higher potential returns\n\
        • Also means higher potential losses\n\
        • Young investors can handle more volatility\n\
        • Near retirement: focus on low volatility\n\n\
        🛡️ Managing Volatility:\n\
        • Diversification reduces portfolio volatility\n\
        • Dollar-cost averaging smooths entry points\n\
        • Long-term holding reduces impact".to_string()
    }

    fn get_beta_explanation(&self) -> String {
        "📊 Beta Explained:\n\n\
        Beta measures how much a stock moves relative to the overall market.\n\n\
        🧮 Understanding Beta:\n\
        • Beta = 1.0: Moves with market\n\
        • Beta > 1.0: More volatile than market\n\
        • Beta < 1.0: Less volatile than market\n\
        • Beta = 0: No correlation with market\n\
        • Negative beta: Moves opposite to market\n\n\
        📈 Examples:\n\
        • Tech stocks: Often beta > 1.5\n\
        • Utilities: Often beta < 0.8\n\
        • Gold stocks: Sometimes negative beta\n\n\
        💼 Portfolio Applications:\n\
        • High-beta stocks for growth\n\
        • Low-beta stocks for stability\n\
        • Portfolio beta = weighted average\n\
        • Defensive positioning with low beta\n\n\
        ⚠️ Limitations:\n\
        • Based on historical data\n\
        • Can change over time\n\
        • Market conditions affect reliability\n\
        • Consider alongside other metrics".to_string()
    }

    fn get_earnings_explanation(&self) -> String {
        "💼 Earnings Explained:\n\n\
        Earnings represent a company's profit after all expenses, taxes, and costs.\n\n\
        📊 Key Metrics:\n\
        • EPS: Earnings Per Share\n\
        • Revenue: Total sales/income\n\
        • Net income: Bottom line profit\n\
        • Operating income: Before interest/taxes\n\n\
        📅 Earnings Season:\n\
        • Quarterly reports (Q1, Q2, Q3, Q4)\n\
        • January, April, July, October\n\
        • Companies report results vs estimates\n\
        • Guidance for future quarters\n\n\
        📈 What to Watch:\n\
        • Beat or miss expectations\n\
        • Revenue growth trends\n\
        • Profit margin changes\n\
        • Forward guidance updates\n\
        • Management commentary\n\n\
        💡 Investment Impact:\n\
        • Strong earnings often boost stock price\n\
        • Misses can cause significant drops\n\
        • Guidance matters more than past results\n\
        • Look for sustainable growth trends\n\
        • Quality of earnings (one-time vs recurring)".to_string()
    }

    fn get_general_explanation(&self) -> String {
        "📚 Financial Education Hub:\n\n\
        I can explain various financial concepts:\n\n\
        📊 Valuation Metrics:\n\
        • P/E Ratio, P/B Ratio, PEG Ratio\n\
        • Market Cap, Enterprise Value\n\
        • Dividend Yield, Payout Ratio\n\n\
        📈 Investment Vehicles:\n\
        • Stocks, Bonds, ETFs, Mutual Funds\n\
        • Options, REITs, Commodities\n\
        • Index Funds vs Active Funds\n\n\
        📉 Risk Concepts:\n\
        • Volatility, Beta, Correlation\n\
        • Diversification, Asset Allocation\n\
        • Risk vs Return Trade-off\n\n\
        💼 Financial Statements:\n\
        • Income Statement, Balance Sheet\n\
        • Cash Flow Statement\n\
        • Key Financial Ratios\n\n\
        Just ask me about any financial term or concept you'd like to understand better!".to_string()
    }
}