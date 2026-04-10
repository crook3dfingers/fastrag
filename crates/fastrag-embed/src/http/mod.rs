//! Shared building blocks for HTTP-backed embedders.
//!
//! Each backend uses a blocking `reqwest::Client`. We keep the corpus indexing
//! path synchronous, so an async runtime is never spun up just for embedding.

use std::time::Duration;

use reqwest::blocking::{Client, RequestBuilder, Response};

use crate::EmbedError;

pub mod ollama;
pub mod openai;

/// Build a blocking reqwest client with sane timeouts for embedding APIs.
pub fn build_client() -> Result<Client, EmbedError> {
    Client::builder()
        .timeout(Duration::from_secs(60))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| EmbedError::Http(e.to_string()))
}

/// Send a request with a single retry on connection errors or 5xx responses.
/// Backoff is a fixed 500ms between the two attempts.
pub fn send_with_retry(make: impl Fn() -> RequestBuilder) -> Result<Response, EmbedError> {
    match make().send() {
        Ok(resp) if resp.status().is_server_error() => {
            std::thread::sleep(Duration::from_millis(500));
            make().send().map_err(|e| EmbedError::Http(e.to_string()))
        }
        Ok(resp) => Ok(resp),
        Err(_) => {
            std::thread::sleep(Duration::from_millis(500));
            make().send().map_err(|e| EmbedError::Http(e.to_string()))
        }
    }
}

/// Read a response, returning an `Api` error if the status is not 2xx.
pub fn ensure_success(resp: Response) -> Result<Response, EmbedError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let code = status.as_u16();
    let body = resp.text().unwrap_or_default();
    let mut message = body;
    if message.len() > 500 {
        message.truncate(500);
    }
    Err(EmbedError::Api {
        status: code,
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_client_succeeds() {
        let _c = build_client().expect("client builds");
    }
}
