use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "fastrag", about = "Fast document parser for AI/RAG pipelines")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Parse one or more files
    Parse {
        /// File or directory to parse
        path: String,

        /// Output format
        #[arg(short, long, default_value = "markdown")]
        format: OutputFormatArg,

        /// Output directory (for batch mode)
        #[arg(short, long)]
        output: Option<String>,

        /// Number of parallel workers
        #[arg(short = 'j', long, default_value = "4")]
        workers: usize,
    },

    /// List supported formats
    Formats,
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormatArg {
    Markdown,
    Json,
    Text,
}
