use crate::{config::AppConfig, error::*};
use bcrypt::{hash, verify, DEFAULT_COST};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String, // user ID
    pub email: String,
    pub exp: i64, // expiration timestamp
    pub iat: i64, // issued at timestamp
}

pub struct AuthService {
    jwt_secret: String,
    jwt_expiration_hours: i64,
}

impl AuthService {
    pub fn new(config: &AppConfig) -> Self {
        Self {
            jwt_secret: config.jwt.secret.clone(),
            jwt_expiration_hours: config.jwt.expiration_hours,
        }
    }

    pub fn hash_password(&self, password: &str) -> Result<String> {
        hash(password, DEFAULT_COST)
            .map_err(|e| AppError::Internal(format!("Password hashing failed: {}", e)))
    }

    pub fn verify_password(&self, password: &str, hash: &str) -> Result<bool> {
        verify(password, hash)
            .map_err(|e| AppError::Internal(format!("Password verification failed: {}", e)))
    }

    pub fn generate_token(&self, user_id: Uuid, email: &str) -> Result<String> {
        let now = Utc::now();
        let exp = now + Duration::hours(self.jwt_expiration_hours);

        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            exp: exp.timestamp(),
            iat: now.timestamp(),
        };

        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_ref()),
        )
        .map_err(|e| AppError::Auth(format!("Token generation failed: {}", e)))
    }

    pub fn verify_token(&self, token: &str) -> Result<Claims> {
        decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_ref()),
            &Validation::default(),
        )
        .map(|token_data| token_data.claims)
        .map_err(|e| AppError::Auth(format!("Token verification failed: {}", e)))
    }

    pub fn extract_user_id_from_token(&self, token: &str) -> Result<Uuid> {
        let claims = self.verify_token(token)?;
        Uuid::parse_str(&claims.sub)
            .map_err(|e| AppError::Auth(format!("Invalid user ID in token: {}", e)))
    }

    pub fn generate_api_key(&self, user_id: Uuid) -> Result<String> {
        let prefix = "jksh_";
        let key_data = format!("{}_{}", user_id, Utc::now().timestamp());
        let hashed = self.hash_password(&key_data)?;
        
        // Create a more readable API key
        let api_key = format!("{}{}", prefix, &hashed[7..39].replace('/', "_").replace('+', "-"));
        Ok(api_key)
    }

    pub fn validate_api_key(&self, api_key: &str) -> Result<bool> {
        if !api_key.starts_with("jksh_") {
            return Ok(false);
        }

        // In a real implementation, you'd store API keys in the database
        // For now, we'll just validate the format
        Ok(api_key.len() > 10 && api_key.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-'))
    }

    pub fn generate_session_token(&self) -> String {
        format!("sess_{}", Uuid::new_v4().to_string().replace('-', ""))
    }
}