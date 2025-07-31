use crate::{config::AppConfig, models::User, error::{AppError, Result}};
use sqlx::{PgPool, Row};
use uuid::Uuid;
use chrono::Utc;

pub struct Database {
    pool: PgPool,
}

impl Database {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let pool = PgPool::connect(&config.database.url)
            .await
            .map_err(AppError::Database)?;

        // Create tables if they don't exist (in a real app, use migrations)
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(100) NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                is_active BOOLEAN DEFAULT TRUE
            )
            "#,
        )
        .execute(&pool)
        .await
        .map_err(AppError::Database)?;

        Ok(Self { pool })
    }

    pub async fn create_user(&self, email: &str, username: &str, password_hash: &str) -> Result<User> {
        let user = sqlx::query_as::<_, User>(
            r#"
            INSERT INTO users (email, username, password_hash)
            VALUES ($1, $2, $3)
            RETURNING *
            "#,
        )
        .bind(email)
        .bind(username)
        .bind(password_hash)
        .fetch_one(&self.pool)
        .await
        .map_err(AppError::Database)?;

        Ok(user)
    }

    pub async fn find_user_by_email(&self, email: &str) -> Result<Option<User>> {
        let user = sqlx::query_as::<_, User>(
            "SELECT * FROM users WHERE email = $1"
        )
        .bind(email)
        .fetch_optional(&self.pool)
        .await
        .map_err(AppError::Database)?;

        Ok(user)
    }

    pub async fn find_user_by_id(&self, id: Uuid) -> Result<Option<User>> {
        let user = sqlx::query_as::<_, User>(
            "SELECT * FROM users WHERE id = $1"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(AppError::Database)?;

        Ok(user)
    }

    pub async fn update_user_activity(&self, id: Uuid, is_active: bool) -> Result<()> {
        sqlx::query(
            "UPDATE users SET is_active = $1, updated_at = NOW() WHERE id = $2"
        )
        .bind(is_active)
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(AppError::Database)?;

        Ok(())
    }

    pub async fn get_user_statistics(&self) -> Result<(i64, i64)> {
        let row = sqlx::query(
            "SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE is_active = true) as active FROM users"
        )
        .fetch_one(&self.pool)
        .await
        .map_err(AppError::Database)?;

        let total: i64 = row.get("total");
        let active: i64 = row.get("active");

        Ok((total, active))
    }

    pub async fn health_check(&self) -> Result<bool> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map_err(AppError::Database)?;

        Ok(true)
    }
}