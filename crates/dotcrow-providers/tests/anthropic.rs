//! Testes do adaptador Anthropic contra um servidor HTTP falso com respostas gravadas.

#![allow(clippy::unwrap_used)]

use std::{
    mem::discriminant,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use dotcrow_providers::{CompletionRequest, LlmProvider, Message, ProviderError, Role, anthropic::Anthropic};
use dotcrow_vault::Vault;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};
use zeroize::Zeroizing;

const KEY: &str = "sk-ant-verify-segredo-123456";

struct RawRequest {
    head: String,
    body: Vec<u8>,
}

impl RawRequest {
    fn request_line(&self) -> &str {
        self.head.lines().next().unwrap_or("")
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.head.lines().skip(1).find_map(|line| {
            let (k, v) = line.split_once(':')?;
            k.trim().eq_ignore_ascii_case(name).then(|| v.trim())
        })
    }

    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}

struct FakeServer {
    url: String,
    connections: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<RawRequest>>>,
}

impl FakeServer {
    async fn start(status: u16, headers: &[(&str, &str)], body: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let connections = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let extra: String = headers.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
        let response = format!(
            "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n{extra}\r\n{body}",
            body.len()
        );
        let (conns, reqs) = (connections.clone(), requests.clone());
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else { return };
                conns.fetch_add(1, Ordering::SeqCst);
                let request = read_request(&mut stream).await;
                reqs.lock().unwrap().push(request);
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });
        Self { url, connections, requests }
    }

    async fn with_fixture(status: u16, headers: &[(&str, &str)], name: &str) -> Self {
        Self::start(status, headers, &fixture(name)).await
    }
}

async fn read_request(stream: &mut TcpStream) -> RawRequest {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let n = stream.read(&mut chunk).await.unwrap();
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let partial = RawRequest { head, body: Vec::new() };
    let len: usize = partial.header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    while buf.len() < head_end + len {
        let n = stream.read(&mut chunk).await.unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    RawRequest { head: partial.head, body: buf[head_end..].to_vec() }
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/fixtures/anthropic/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn vault_with_key() -> Arc<Vault> {
    let mut vault = Vault::in_memory();
    vault.add("anthropic", Zeroizing::new(KEY.to_string()), None).unwrap();
    Arc::new(vault)
}

fn provider(url: &str) -> Anthropic {
    Anthropic::new(vault_with_key()).unwrap().with_base_url(url)
}

fn request() -> CompletionRequest {
    CompletionRequest {
        messages: vec![
            Message { role: Role::System, content: "Você é o Huginn.".into() },
            Message { role: Role::User, content: "Oi".into() },
            Message { role: Role::Assistant, content: "Olá".into() },
            Message { role: Role::System, content: "Responda curto.".into() },
            Message { role: Role::User, content: "Tudo bem?".into() },
        ],
        max_tokens: 321,
    }
}

fn assert_no_key(error: &ProviderError) {
    assert!(!format!("{error}").contains(KEY), "a chave vazou no Display: {error}");
    assert!(!format!("{error:?}").contains(KEY), "a chave vazou no Debug: {error:?}");
}

#[tokio::test]
async fn success_returns_joined_text_and_sends_the_messages_api_request() {
    let server = FakeServer::with_fixture(200, &[], "success.json").await;

    let response = provider(&server.url).complete(&request()).await.unwrap();

    assert_eq!(response.text, "Olá, dono!");
    assert_eq!(response.model, "claude-opus-5-5-20261001");

    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    assert!(sent.request_line().starts_with("POST /v1/messages "), "{}", sent.request_line());
    assert_eq!(sent.header("x-api-key"), Some(KEY));
    assert_eq!(sent.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(sent.header("anthropic-beta"), Some("server-side-fallback-2026-07-01"));
    let body = sent.json();
    assert_eq!(body["max_tokens"], 321);
    assert_eq!(body["model"], "claude-opus-5-5");
    assert_eq!(body["fallbacks"], "default");
    assert_eq!(body["system"], "Você é o Huginn.\n\nResponda curto.");
    assert_eq!(
        body["messages"],
        serde_json::json!([
            { "role": "user", "content": "Oi" },
            { "role": "assistant", "content": "Olá" },
            { "role": "user", "content": "Tudo bem?" },
        ])
    );
}

#[tokio::test]
async fn request_without_system_messages_omits_the_system_field() {
    let server = FakeServer::with_fixture(200, &[], "success.json").await;
    let request =
        CompletionRequest { messages: vec![Message { role: Role::User, content: "Oi".into() }], max_tokens: 16 };

    provider(&server.url).complete(&request).await.unwrap();

    let body = server.requests.lock().unwrap()[0].json();
    assert!(body.get("system").is_none(), "{body}");
}

#[tokio::test]
async fn error_responses_are_classified_for_the_router() {
    let unavailable = ProviderError::Unavailable(String::new());
    let bad_request = ProviderError::BadRequest(String::new());
    let rate_limited = ProviderError::RateLimited { retry_after: None };
    let cases: &[(u16, &str, ProviderError)] = &[
        (400, "invalid_request.json", bad_request.clone()),
        (400, "credit_balance.json", ProviderError::QuotaExhausted),
        (402, "billing.json", ProviderError::QuotaExhausted),
        (401, "authentication.json", ProviderError::Unauthorized),
        (403, "permission.json", ProviderError::Unauthorized),
        (404, "not_found.json", unavailable.clone()),
        (429, "rate_limit.json", rate_limited),
        (500, "api_error.json", unavailable.clone()),
        (529, "overloaded.json", unavailable.clone()),
        (503, "api_error.json", unavailable.clone()),
        (200, "refusal.json", bad_request),
    ];

    for (status, name, expected) in cases {
        let server = FakeServer::with_fixture(*status, &[], name).await;

        let error = provider(&server.url).complete(&request()).await.unwrap_err();

        assert_eq!(discriminant(&error), discriminant(expected), "{status} {name}: {error:?}");
        assert_no_key(&error);
    }
}

#[tokio::test]
async fn unparseable_bodies_fall_back_to_the_status() {
    for status in [200, 500, 418] {
        let server = FakeServer::start(status, &[], "<html>gateway</html>").await;

        let error = provider(&server.url).complete(&request()).await.unwrap_err();

        let expected = if status == 418 {
            ProviderError::BadRequest(String::new())
        } else {
            ProviderError::Unavailable(String::new())
        };
        assert_eq!(discriminant(&error), discriminant(&expected), "{status}: {error:?}");
        assert_no_key(&error);
    }
}

#[tokio::test]
async fn rate_limit_reads_retry_after_seconds() {
    let server = FakeServer::with_fixture(429, &[("retry-after", "7")], "rate_limit.json").await;

    let error = provider(&server.url).complete(&request()).await.unwrap_err();

    assert_eq!(error, ProviderError::RateLimited { retry_after: Some(Duration::from_secs(7)) });
    assert_no_key(&error);
}

#[tokio::test]
async fn missing_key_is_not_configured_and_makes_no_request() {
    let server = FakeServer::with_fixture(200, &[], "success.json").await;
    let provider = Anthropic::new(Arc::new(Vault::in_memory())).unwrap().with_base_url(&server.url);

    let error = provider.complete(&request()).await.unwrap_err();

    assert_eq!(error, ProviderError::NotConfigured);
    assert_eq!(server.connections.load(Ordering::SeqCst), 0);
    assert_no_key(&error);
}

#[tokio::test]
async fn connection_refused_is_unavailable() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);

    let error = provider(&url).complete(&request()).await.unwrap_err();

    assert!(matches!(error, ProviderError::Unavailable(_)), "{error:?}");
    assert_no_key(&error);
}

#[test]
fn debug_never_shows_the_key() {
    let provider = Anthropic::new(vault_with_key()).unwrap().with_model("claude-teste");

    let shown = format!("{provider:?}");

    assert!(!shown.contains(KEY), "{shown}");
    assert!(shown.contains("claude-teste"), "{shown}");
}

/// Chama a API real da Anthropic. Fica fora do CI porque precisa de chave.
///
/// Para rodar localmente:
///
/// ```text
/// DOTCROW_ANTHROPIC_KEY=sk-ant-... cargo test -p dotcrow-providers --test anthropic -- --ignored
/// ```
#[tokio::test]
#[ignore = "chama a API real; precisa de DOTCROW_ANTHROPIC_KEY"]
async fn real_api_answers() {
    let key = std::env::var("DOTCROW_ANTHROPIC_KEY").expect("defina DOTCROW_ANTHROPIC_KEY");
    let mut vault = Vault::in_memory();
    vault.add("anthropic", Zeroizing::new(key), None).unwrap();
    let request = CompletionRequest {
        messages: vec![Message { role: Role::User, content: "Responda apenas com a palavra: corvo".into() }],
        max_tokens: 256,
    };

    let response = Anthropic::new(Arc::new(vault)).unwrap().complete(&request).await.unwrap();

    assert!(!response.text.trim().is_empty(), "{response:?}");
    assert!(response.model.starts_with("claude"), "{response:?}");
}
