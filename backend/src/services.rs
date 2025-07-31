use crate::{
    auth::AuthService,
    cache::Cache,
    config::AppConfig,
    database::Database,
    error::{AppError, Result},
    models::*,
    financial::FinancialService,
    ai::AIService,
    whatsapp::WhatsAppService,
};
use std::sync::Arc;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use std::time::Duration;

pub struct AppServices {
    pub auth: AuthService,
    pub financial: Arc<FinancialService>,
    pub ai: Arc<AIService>,
    pub whatsapp: Arc<WhatsAppService>,
    pub analytics: AnalyticsService,
}

impl AppServices {
    pub async fn new(database: Arc<Database>, cache: Arc<Cache>) -> Result<Self> {
        let config = AppConfig::new()?;
        
        let financial_service = Arc::new(FinancialService::new(cache.clone()));
        let ai_service = Arc::new(AIService::new(financial_service.clone(), cache.clone()));
        let whatsapp_service = Arc::new(WhatsAppService::new(&config));
        
        Ok(Self {
            auth: AuthService::new(&config),
            financial: financial_service,
            ai: ai_service,
            whatsapp: whatsapp_service,
            analytics: AnalyticsService::new(database.clone(), cache.clone()),
        })
    }
}

pub struct AnalyticsService {
    database: Arc<Database>,
    cache: Arc<Cache>,
}

impl AnalyticsService {
    pub fn new(database: Arc<Database>, cache: Arc<Cache>) -> Self {
        Self { database, cache }
    }

    pub async fn get_analytics_data(&self, metric_type: Option<String>) -> Result<AnalyticsResponse> {
        let cache_key = format!("analytics_{}", metric_type.as_deref().unwrap_or("all"));
        
        // Try cache first
        if let Ok(Some(cached_data)) = self.cache.get::<AnalyticsResponse>(&cache_key).await {
            return Ok(cached_data);
        }

        // Generate synthetic analytics data
        let mut data = Vec::new();
        let mut aggregations = HashMap::new();
        
        let metrics = match metric_type.as_deref() {
            Some("user_engagement") => vec!["page_views", "session_duration", "bounce_rate"],
            Some("performance") => vec!["response_time", "throughput", "error_rate"],
            _ => vec!["page_views", "users", "sessions", "conversion_rate"],
        };

        let start_time = Utc::now() - chrono::Duration::hours(24);
        let end_time = Utc::now();

        for metric in metrics {
            let mut total = 0.0;
            for hour in 0..24 {
                let timestamp = start_time + chrono::Duration::hours(hour);
                let value = rand::random::<f64>() * 1000.0 + 100.0;
                total += value;
                
                data.push(AnalyticsData {
                    metric_name: metric.to_string(),
                    value,
                    timestamp,
                    dimensions: {
                        let mut dims = HashMap::new();
                        dims.insert("source".to_string(), "web".to_string());
                        dims.insert("region".to_string(), "us-east-1".to_string());
                        dims
                    },
                });
            }
            aggregations.insert(format!("{}_total", metric), total);
            aggregations.insert(format!("{}_avg", metric), total / 24.0);
        }

        let response = AnalyticsResponse {
            data,
            aggregations,
            time_range: (start_time, end_time),
        };

        // Cache for 5 minutes
        let _ = self.cache.set(&cache_key, &response, Some(Duration::from_secs(300))).await;

        Ok(response)
    }

    pub async fn record_event(&self, event_name: &str, properties: HashMap<String, String>) -> Result<()> {
        let event_key = format!("event_{}_{}", event_name, Utc::now().timestamp());
        let event_data = serde_json::json!({
            "name": event_name,
            "properties": properties,
            "timestamp": Utc::now()
        });

        self.cache.list_push("events_queue", &event_data.to_string()).await?;
        tracing::info!("Event recorded: {} with properties: {:?}", event_name, properties);
        
        Ok(())
    }
}

// Legacy services removed - now using specialized financial AI services