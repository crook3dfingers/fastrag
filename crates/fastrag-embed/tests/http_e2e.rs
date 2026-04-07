#![cfg(feature = "http-embedders")]

use fastrag_embed::http::ollama::OllamaEmbedder;
use fastrag_embed::http::openai::OpenAIEmbedder;
use fastrag_embed::Embedder;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn openai_embed_through_trait() {
    let rt = rt();
    let (uri, _g) = rt.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({
                    "data": [
                        { "embedding": vec![0.1_f32; 1536] },
                        { "embedding": vec![0.2_f32; 1536] },
                    ]
                })),
            )
            .mount(&server)
            .await;
        (server.uri(), server)
    });
    unsafe { std::env::set_var("OPENAI_API_KEY", "k") };
    let e: Box<dyn Embedder> = Box::new(
        OpenAIEmbedder::new("text-embedding-3-small")
            .unwrap()
            .with_base_url(uri),
    );
    let vecs = e.embed(&["a", "b"]).unwrap();
    assert_eq!(vecs.len(), 2);
    assert_eq!(e.dim(), 1536);
    assert_eq!(e.model_id(), "openai:text-embedding-3-small");
}

#[test]
fn ollama_embed_through_trait() {
    let rt = rt();
    let (uri, _g) = rt.block_on(async {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "embedding": vec![0.3_f32; 6] })),
            )
            .mount(&server)
            .await;
        (server.uri(), server)
    });
    unsafe { std::env::set_var("OLLAMA_HOST", &uri) };
    let e: Box<dyn Embedder> = Box::new(OllamaEmbedder::new("nomic-embed-text").unwrap());
    let vecs = e.embed(&["a", "b"]).unwrap();
    assert_eq!(vecs.len(), 2);
    assert_eq!(vecs[0].len(), 6);
    assert_eq!(e.model_id(), "ollama:nomic-embed-text");
}
