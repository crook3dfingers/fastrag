pub mod baseline;
mod dataset;
pub mod datasets;
mod error;
pub mod gold_set;
pub mod matrix;
pub mod matrix_real;
mod metrics;
mod report;
mod runner;

pub use dataset::{EvalDataset, EvalDocument, EvalQuery, Qrel};
pub use datasets::{
    DatasetName, load_by_name, load_cwe_top25, load_nfcorpus, load_nvd,
    load_nvd_corpus_with_queries, load_scifact,
};
pub use error::{EvalError, EvalResult};
pub use gold_set::{GoldSet, GoldSetEntry};
pub use metrics::{hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k};
pub use report::{EvalReport, LatencyStats, MemoryStats, write_matrix_report};
pub use runner::{Runner, index_documents};
