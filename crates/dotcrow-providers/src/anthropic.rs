//! Adaptador da Anthropic (Claude), pela Messages API.

use std::{fmt, sync::Arc, time::Duration};

use dotcrow_vault::Vault;
use reqwest::header::{HeaderValue, RETRY_AFTER};
use serde::{Deserialize, Serialize};

use crate::{CompletionRequest, CompletionResponse, LlmProvider, ProviderError, Role};

const PROVIDER_ID: &str = "anthropic";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MODEL: &str = "claude-opus-5-5";
const API_VERSION: &str = "2023-06-01";
const BETA: &str = "server-side-fallback-2026-07-01";
const TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Provedor Claude, da Anthropic. A chave sai do cofre só na hora de cada pedido.
pub struct Anthropic {
    vault: Arc<Vault>,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl Anthropic {
    /// Cria o provedor com a API oficial e o modelo padrão, lendo a chave `anthropic` do cofre.
    ///
    /// Falha só se o sistema não conseguir preparar a conexão segura (TLS).
    pub fn new(vault: Arc<Vault>) -> Result<Self, ProviderError> {
        let client = reqwest::Client::builder().timeout(TIMEOUT).build().map_err(|_| {
            ProviderError::Unavailable(
                "não consegui preparar a conexão segura com a Anthropic; confira os certificados do sistema".into(),
            )
        })?;
        Ok(Self { vault, base_url: DEFAULT_BASE_URL.into(), model: DEFAULT_MODEL.into(), client })
    }

    /// Troca o modelo pedido à API.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Troca o endereço base da API (por exemplo, um proxy ou um servidor de teste).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

impl fmt::Debug for Anthropic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Anthropic")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl LlmProvider for Anthropic {
    fn id(&self) -> &str {
        PROVIDER_ID
    }

    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        let api_key = {
            let lease = self.vault.lease(PROVIDER_ID).ok_or(ProviderError::NotConfigured)?;
            let mut value = lease.with_secret(HeaderValue::from_str).map_err(|_| ProviderError::Unauthorized)?;
            value.set_sensitive(true);
            value
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", api_key)
            .header("anthropic-version", API_VERSION)
            .header("anthropic-beta", BETA)
            .json(&WireRequest::new(&self.model, request))
            .send()
            .await
            .map_err(transport_error)?;

        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse().ok())
            .map(Duration::from_secs);
        let body = response.bytes().await.map_err(transport_error)?;

        if !(200..300).contains(&status) {
            let error = serde_json::from_slice::<WireErrorBody>(&body).ok().map(|b| b.error.message);
            return Err(classify(status, retry_after, error.as_deref().unwrap_or(""), &self.model));
        }

        let parsed: WireResponse = serde_json::from_slice(&body).map_err(|_| {
            ProviderError::Unavailable("resposta ilegível da Anthropic; tente de novo daqui a pouco".into())
        })?;
        if parsed.stop_reason.as_deref() == Some("refusal") {
            return Err(ProviderError::BadRequest("o modelo recusou o pedido; reformule e tente de novo".into()));
        }
        let text = parsed
            .content
            .into_iter()
            .filter_map(|block| match block {
                WireBlock::Text { text } => Some(text),
                WireBlock::Other => None,
            })
            .collect();
        Ok(CompletionResponse { text, model: parsed.model })
    }
}

fn classify(status: u16, retry_after: Option<Duration>, message: &str, model: &str) -> ProviderError {
    match status {
        400 if message.to_lowercase().contains("credit balance is too low") => ProviderError::QuotaExhausted,
        404 => ProviderError::Unavailable(format!(
            "modelo {model} não encontrado ou indisponível para esta conta; confira o nome do modelo no Console da Anthropic"
        )),
        _ => ProviderError::from_http_status(status, retry_after, message),
    }
}

fn transport_error(error: reqwest::Error) -> ProviderError {
    let reason = if error.is_timeout() { "tempo esgotado" } else { "falha de rede" };
    ProviderError::Unavailable(format!("{reason} ao falar com a Anthropic; confira sua conexão com a internet"))
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<WireMessage<'a>>,
    fallbacks: &'static str,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: Role,
    content: &'a str,
}

impl<'a> WireRequest<'a> {
    fn new(model: &'a str, request: &'a CompletionRequest) -> Self {
        let (system, messages): (Vec<_>, Vec<_>) = request.messages.iter().partition(|m| m.role == Role::System);
        let system =
            (!system.is_empty()).then(|| system.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n\n"));
        let messages = messages.into_iter().map(|m| WireMessage { role: m.role, content: &m.content }).collect();
        Self { model, max_tokens: request.max_tokens, system, messages, fallbacks: "default" }
    }
}

#[derive(Deserialize)]
struct WireResponse {
    model: String,
    content: Vec<WireBlock>,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireBlock {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireErrorBody {
    error: WireError,
}

#[derive(Deserialize)]
struct WireError {
    message: String,
}
