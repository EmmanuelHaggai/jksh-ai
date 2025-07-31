use crate::{config::AppConfig, error::{AppError, Result}};
use redis::{AsyncCommands, Client};
use serde::{Serialize, Deserialize};
use std::time::Duration;

pub struct Cache {
    client: Client,
}

impl Cache {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let client = Client::open(config.redis.url.clone())
            .map_err(AppError::Redis)?;

        // Test connection
        let mut conn = client.get_async_connection().await.map_err(AppError::Redis)?;
        let _: String = conn.ping().await.map_err(AppError::Redis)?;

        Ok(Self { client })
    }

    pub async fn set<T>(&self, key: &str, value: &T, ttl: Option<Duration>) -> Result<()>
    where
        T: Serialize,
    {
        let mut conn = self.client.get_async_connection().await.map_err(AppError::Redis)?;
        let serialized = serde_json::to_string(value)
            .map_err(|e| AppError::Internal(format!("Serialization error: {}", e)))?;

        if let Some(ttl) = ttl {
            conn.set_ex(key, serialized, ttl.as_secs()).await.map_err(AppError::Redis)?;
        } else {
            conn.set(key, serialized).await.map_err(AppError::Redis)?;
        }

        Ok(())
    }

    pub async fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut conn = self.client.get_async_connection().await.map_err(AppError::Redis)?;
        let value: Option<String> = conn.get(key).await.map_err(AppError::Redis)?;

        match value {
            Some(serialized) => {
                let deserialized = serde_json::from_str(&serialized)
                    .map_err(|e| AppError::Internal(format!("Deserialization error: {}", e)))?;
                Ok(Some(deserialized))
            }
            None => Ok(None),
        }
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        let mut conn = self.client.get_async_connection().await.map_err(AppError::Redis)?;
        let _: i32 = conn.del(key).await.map_err(AppError::Redis)?;
        Ok(())
    }

    pub async fn increment(&self, key: &str, delta: i64) -> Result<i64> {
        let mut conn = self.client.get_async_connection().await.map_err(AppError::Redis)?;
        let result: i64 = conn.incr(key, delta).await.map_err(AppError::Redis)?;
        Ok(result)
    }

    pub async fn set_if_not_exists(&self, key: &str, value: &str, ttl: Option<Duration>) -> Result<bool> {
        let mut conn = self.client.get_async_connection().await.map_err(AppError::Redis)?;
        
        let result: bool = if let Some(ttl) = ttl {
            conn.set_nx_ex(key, value, ttl.as_secs()).await.map_err(AppError::Redis)?
        } else {
            conn.set_nx(key, value).await.map_err(AppError::Redis)?
        };

        Ok(result)
    }

    pub async fn list_push(&self, key: &str, value: &str) -> Result<i64> {
        let mut conn = self.client.get_async_connection().await.map_err(AppError::Redis)?;
        let length: i64 = conn.lpush(key, value).await.map_err(AppError::Redis)?;
        Ok(length)
    }

    pub async fn list_pop(&self, key: &str) -> Result<Option<String>> {
        let mut conn = self.client.get_async_connection().await.map_err(AppError::Redis)?;
        let value: Option<String> = conn.lpop(key, None).await.map_err(AppError::Redis)?;
        Ok(value)
    }

    pub async fn health_check(&self) -> Result<bool> {
        let mut conn = self.client.get_async_connection().await.map_err(AppError::Redis)?;
        let _: String = conn.ping().await.map_err(AppError::Redis)?;
        Ok(true)
    }
}