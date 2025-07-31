use crate::{
    error::{AppError, Result},
    models::*,
    config::AppConfig,
};
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use tracing::{info, error, warn};

pub struct WhatsAppService {
    client: Client,
    phone_number_id: String,
    access_token: String,
    verify_token: String,
    api_base_url: String,
}

impl WhatsAppService {
    pub fn new(config: &AppConfig) -> Self {
        Self {
            client: Client::new(),
            phone_number_id: std::env::var("WHATSAPP_PHONE_NUMBER_ID")
                .unwrap_or_else(|_| "your_phone_number_id".to_string()),
            access_token: std::env::var("WHATSAPP_ACCESS_TOKEN")
                .unwrap_or_else(|_| "your_access_token".to_string()),
            verify_token: std::env::var("WHATSAPP_VERIFY_TOKEN")
                .unwrap_or_else(|_| "your_verify_token".to_string()),
            api_base_url: "https://graph.facebook.com/v18.0".to_string(),
        }
    }

    pub fn verify_webhook(&self, mode: &str, token: &str, challenge: &str) -> Result<String> {
        if mode == "subscribe" && token == self.verify_token {
            info!("Webhook verified successfully");
            Ok(challenge.to_string())
        } else {
            warn!("Webhook verification failed");
            Err(AppError::Auth("Invalid verification token".to_string()))
        }
    }

    pub async fn send_message(&self, to: &str, message: &str) -> Result<()> {
        let url = format!("{}/{}/messages", self.api_base_url, self.phone_number_id);
        
        let payload = WhatsAppOutgoingMessage {
            messaging_product: "whatsapp".to_string(),
            to: to.to_string(),
            message_type: "text".to_string(),
            text: Some(WhatsAppOutgoingText {
                body: message.to_string(),
            }),
        };

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| AppError::ExternalService(format!("WhatsApp API error: {}", e)))?;

        if response.status().is_success() {
            info!("Message sent successfully to {}", to);
            Ok(())
        } else {
            let error_text = response.text().await.unwrap_or_default();
            error!("Failed to send message: {}", error_text);
            Err(AppError::ExternalService(format!("Failed to send message: {}", error_text)))
        }
    }

    pub async fn send_financial_summary(&self, to: &str, summary: &FinancialSummary) -> Result<()> {
        let message = format!(
            "📊 *Financial Summary*\n\n\
            💰 Portfolio Value: ${:.2}\n\
            📈 Daily Change: ${:.2} ({:.2}%)\n\
            🎯 Top Performer: {} (+{:.2}%)\n\
            ⚠️ Biggest Loser: {} ({:.2}%)\n\n\
            📋 Recent Activity:\n{}\n\n\
            💡 AI Recommendation: {}\n\n\
            Type 'help' for more options or ask me anything about your finances!",
            summary.total_value,
            summary.daily_change,
            summary.daily_change_percent,
            summary.top_performer.0,
            summary.top_performer.1,
            summary.biggest_loser.0,
            summary.biggest_loser.1,
            summary.recent_activity.join("\n"),
            summary.ai_recommendation
        );

        self.send_message(to, &message).await
    }

    pub async fn send_market_alert(&self, to: &str, alert: &MarketAlert) -> Result<()> {
        let emoji = match alert.alert_type.as_str() {
            "price_target" => "🎯",
            "volatility" => "⚡",
            "news" => "📰",
            "earnings" => "📊",
            _ => "🚨",
        };

        let message = format!(
            "{} *Market Alert*\n\n\
            🏷️ Symbol: *{}*\n\
            📊 Current Price: ${:.2}\n\
            📈 Change: {:.2}%\n\n\
            ℹ️ Alert: {}\n\n\
            🔍 Details: {}\n\n\
            ⏰ {}\n\n\
            Reply 'analyze {}' for detailed analysis!",
            emoji,
            alert.symbol,
            alert.current_price,
            alert.change_percent,
            alert.title,
            alert.description,
            alert.timestamp.format("%H:%M UTC"),
            alert.symbol
        );

        self.send_message(to, &message).await
    }

    pub async fn send_ai_response(&self, to: &str, response: &AIRecommendationResponse) -> Result<()> {
        let mut message = format!("🤖 *AI Financial Assistant*\n\n{}\n\n", response.response);

        if !response.recommendations.is_empty() {
            message.push_str("💡 *Recommendations:*\n");
            for (i, rec) in response.recommendations.iter().enumerate() {
                message.push_str(&format!("{}. {}\n", i + 1, rec));
            }
            message.push('\n');
        }

        if !response.relevant_stocks.is_empty() {
            message.push_str("📈 *Relevant Stocks:* ");
            message.push_str(&response.relevant_stocks.join(", "));
            message.push_str("\n\n");
        }

        if !response.follow_up_questions.is_empty() {
            message.push_str("❓ *You can also ask:*\n");
            for question in &response.follow_up_questions {
                message.push_str(&format!("• {}\n", question));
            }
        }

        message.push_str(&format!("\n🔍 Confidence: {:.0}%", response.confidence * 100.0));

        self.send_message(to, &message).await
    }

    pub async fn send_help_menu(&self, to: &str) -> Result<()> {
        let message = r#"🤖 *Financial AI Assistant - Help Menu*

📊 *Portfolio Commands:*
• "portfolio" - View your portfolio summary
• "analyze AAPL" - Analyze a specific stock
• "risk assessment" - Check your risk profile

📈 *Market Data:*
• "price TSLA" - Get current stock price
• "market news" - Latest financial news
• "trending stocks" - Today's movers

🎯 *AI Features:*
• "predict AAPL 30" - Price prediction (30 days)
• "should I buy NVDA?" - Investment advice
• "technical analysis SPY" - Technical indicators

⚙️ *Settings:*
• "set alerts AAPL" - Price alerts
• "risk tolerance moderate" - Update profile
• "watchlist add MSFT" - Manage watchlist

💡 *Examples:*
• "How is Tesla performing?"
• "What's the best tech stock to buy?"
• "Analyze my portfolio risk"
• "Should I sell my Apple shares?"

Just type your question naturally - I understand conversational language! 🗣️"#;

        self.send_message(to, &message).await
    }
}

// Helper structs for formatted responses
#[derive(Debug)]
pub struct FinancialSummary {
    pub total_value: f64,
    pub daily_change: f64,
    pub daily_change_percent: f64,
    pub top_performer: (String, f64),
    pub biggest_loser: (String, f64),
    pub recent_activity: Vec<String>,
    pub ai_recommendation: String,
}

#[derive(Debug)]
pub struct MarketAlert {
    pub symbol: String,
    pub current_price: f64,
    pub change_percent: f64,
    pub alert_type: String,
    pub title: String,
    pub description: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// Message processing utilities
pub struct MessageProcessor;

impl MessageProcessor {
    pub fn extract_command_and_symbol(message: &str) -> (String, Option<String>) {
        let words: Vec<&str> = message.to_lowercase().split_whitespace().collect();
        
        if words.is_empty() {
            return ("unknown".to_string(), None);
        }

        let command = match words[0] {
            "analyze" | "analysis" => "analyze",
            "price" | "quote" => "price",
            "portfolio" | "holdings" => "portfolio",
            "predict" | "prediction" => "predict",
            "news" => "news",
            "help" => "help",
            "buy" | "should" if message.contains("buy") => "buy_advice",
            "sell" if message.contains("sell") => "sell_advice",
            "risk" => "risk",
            "alerts" | "alert" => "alerts",
            _ => "general_query",
        }.to_string();

        // Extract potential stock symbol
        let symbol = words.iter()
            .find(|&&word| word.len() >= 1 && word.len() <= 5 && word.chars().all(|c| c.is_alphabetic()))
            .map(|&s| s.to_uppercase());

        (command, symbol)
    }

    pub fn is_financial_query(message: &str) -> bool {
        let financial_keywords = [
            "stock", "stocks", "invest", "investment", "portfolio", "market", "price",
            "buy", "sell", "trade", "trading", "dividend", "earnings", "analyst",
            "bullish", "bearish", "volatility", "risk", "return", "profit", "loss",
            "crypto", "bitcoin", "ethereum", "bond", "etf", "mutual fund",
            "recession", "inflation", "fed", "interest rate", "gdp", "economy"
        ];

        let lower_message = message.to_lowercase();
        financial_keywords.iter().any(|&keyword| lower_message.contains(keyword))
    }

    pub fn extract_timeframe(message: &str) -> Option<String> {
        let message = message.to_lowercase();
        
        if message.contains("1 day") || message.contains("1d") || message.contains("today") {
            Some("1d".to_string())
        } else if message.contains("1 week") || message.contains("1w") || message.contains("week") {
            Some("1w".to_string())
        } else if message.contains("1 month") || message.contains("1mo") || message.contains("month") {
            Some("1mo".to_string())
        } else if message.contains("3 month") || message.contains("3mo") || message.contains("quarter") {
            Some("3mo".to_string())
        } else if message.contains("1 year") || message.contains("1y") || message.contains("year") {
            Some("1y".to_string())
        } else {
            None
        }
    }
}