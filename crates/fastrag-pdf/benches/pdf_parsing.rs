use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fastrag_core::{ElementKind, FileFormat, Parser, SourceInfo};
use fastrag_pdf::PdfParser;

// --- fixture bytes loaded at compile time ---
static SAMPLE_PDF: &[u8] = include_bytes!("../../../tests/fixtures/sample.pdf");
static SAMPLE_TABLE: &[u8] = include_bytes!("../../../tests/fixtures/sample_table.pdf");
static SAMPLE_IMAGES: &[u8] = include_bytes!("../../../tests/fixtures/sample_images.pdf");
static COMPLEX_TABLE: &[u8] = include_bytes!("../../../tests/fixtures/complex_table.pdf");
static MIXED_CONTENT: &[u8] = include_bytes!("../../../tests/fixtures/mixed_content.pdf");
static LARGE_TABLES: &[u8] = include_bytes!("../../../tests/fixtures/large_tables.pdf");
static SCANNED_PDF: &[u8] = include_bytes!("../../../tests/fixtures/sample_scanned.pdf");

fn source() -> SourceInfo {
    SourceInfo::new(FileFormat::Pdf)
}

// ─── Baseline: plain text extraction (no feature flags) ─────────────────────

fn bench_baseline(c: &mut Criterion) {
    let parser = PdfParser;
    let mut grp = c.benchmark_group("baseline");

    let fixtures: &[(&str, &[u8])] = &[
        ("sample_3pg", SAMPLE_PDF),
        ("complex_table_2pg", COMPLEX_TABLE),
        ("mixed_content_4pg", MIXED_CONTENT),
        ("large_tables_5pg", LARGE_TABLES),
    ];

    for (name, bytes) in fixtures {
        grp.throughput(Throughput::Bytes(bytes.len() as u64));
        grp.bench_with_input(BenchmarkId::from_parameter(name), bytes, |b, data| {
            b.iter(|| parser.parse(black_box(data), &source()).unwrap())
        });
    }
    grp.finish();
}

// ─── Image extraction ────────────────────────────────────────────────────────

#[cfg(feature = "images")]
fn bench_images(c: &mut Criterion) {
    let parser = PdfParser;
    let mut grp = c.benchmark_group("images");

    let fixtures: &[(&str, &[u8])] = &[
        ("sample_images_2img", SAMPLE_IMAGES),
        ("mixed_content_3img", MIXED_CONTENT),
        ("scanned_1img", SCANNED_PDF),
    ];

    for (name, bytes) in fixtures {
        grp.throughput(Throughput::Bytes(bytes.len() as u64));
        grp.bench_with_input(BenchmarkId::from_parameter(name), bytes, |b, data| {
            b.iter(|| {
                let doc = parser.parse(black_box(data), &source()).unwrap();
                black_box(
                    doc.elements
                        .iter()
                        .filter(|e| e.kind == ElementKind::Image)
                        .count(),
                )
            })
        });
    }
    grp.finish();
}

#[cfg(not(feature = "images"))]
fn bench_images(_c: &mut Criterion) {}

// ─── Table detection ─────────────────────────────────────────────────────────

#[cfg(feature = "table-detect")]
fn bench_tables(c: &mut Criterion) {
    let parser = PdfParser;
    let mut grp = c.benchmark_group("table_detect");

    let fixtures: &[(&str, &[u8])] = &[
        ("sample_table_1tbl", SAMPLE_TABLE),
        ("complex_table_20row", COMPLEX_TABLE),
        ("large_tables_10tbl", LARGE_TABLES),
        ("mixed_content_2tbl", MIXED_CONTENT),
    ];

    for (name, bytes) in fixtures {
        grp.throughput(Throughput::Bytes(bytes.len() as u64));
        grp.bench_with_input(BenchmarkId::from_parameter(name), bytes, |b, data| {
            b.iter(|| {
                let doc = parser.parse(black_box(data), &source()).unwrap();
                black_box(
                    doc.elements
                        .iter()
                        .filter(|e| e.kind == ElementKind::Table)
                        .count(),
                )
            })
        });
    }
    grp.finish();
}

#[cfg(not(feature = "table-detect"))]
fn bench_tables(_c: &mut Criterion) {}

// ─── Combined: images + table-detect ─────────────────────────────────────────

#[cfg(all(feature = "images", feature = "table-detect"))]
fn bench_combined(c: &mut Criterion) {
    let parser = PdfParser;
    let mut grp = c.benchmark_group("combined");

    let fixtures: &[(&str, &[u8])] = &[
        ("mixed_content_full", MIXED_CONTENT),
        ("large_tables_full", LARGE_TABLES),
        ("complex_table_full", COMPLEX_TABLE),
    ];

    for (name, bytes) in fixtures {
        grp.throughput(Throughput::Bytes(bytes.len() as u64));
        grp.bench_with_input(BenchmarkId::from_parameter(name), bytes, |b, data| {
            b.iter(|| {
                let doc = parser.parse(black_box(data), &source()).unwrap();
                black_box(doc.elements.len())
            })
        });
    }
    grp.finish();
}

#[cfg(not(all(feature = "images", feature = "table-detect")))]
fn bench_combined(_c: &mut Criterion) {}

// ─── classify_image micro-benchmark ──────────────────────────────────────────

#[cfg(feature = "images")]
fn bench_classify_image(c: &mut Criterion) {
    use fastrag_pdf::images::classify_image;

    let cases: &[(u32, u32)] = &[
        (400, 200),
        (800, 300),
        (50, 50),
        (300, 600),
        (1920, 800),
        (100, 100),
    ];

    c.bench_function("classify_image_6_cases", |b| {
        b.iter(|| {
            for &(w, h) in cases {
                black_box(classify_image(black_box(w), black_box(h)));
            }
        })
    });
}

#[cfg(not(feature = "images"))]
fn bench_classify_image(_c: &mut Criterion) {}

// ─── table::collect_positioned_text micro-benchmark ──────────────────────────

#[cfg(feature = "table-detect")]
fn bench_collect_positioned(c: &mut Criterion) {
    use fastrag_pdf::table::collect_positioned_text;

    // Simulate ops for a 5-column × 20-row table
    let mut ops = Vec::new();
    let cols: &[f32] = &[50.0, 150.0, 250.0, 350.0, 450.0];
    for row in 0..20u32 {
        let y = 700.0 - row as f32 * 14.0;
        for &x in cols {
            ops.push(pdf::content::Op::SetTextMatrix {
                matrix: pdf::content::Matrix {
                    a: 1.0,
                    b: 0.0,
                    c: 0.0,
                    d: 1.0,
                    e: x,
                    f: y,
                },
            });
            let text_bytes = format!("cell-{row}-{x:.0}").into_bytes();
            ops.push(pdf::content::Op::TextDraw {
                text: pdf::primitive::PdfString::new(text_bytes.into()),
            });
        }
    }

    c.bench_function("collect_positioned_5col_20row", |b| {
        b.iter(|| black_box(collect_positioned_text(black_box(&ops))))
    });
}

#[cfg(not(feature = "table-detect"))]
fn bench_collect_positioned(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_baseline,
    bench_images,
    bench_tables,
    bench_combined,
    bench_classify_image,
    bench_collect_positioned,
);
criterion_main!(benches);
