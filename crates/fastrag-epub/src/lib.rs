use std::io::Read;

use fastrag_core::{Document, FastRagError, FileFormat, Metadata, Parser, SourceInfo};

/// EPUB parser that reads ZIP-based EPUB archives.
pub struct EpubParser;

impl Parser for EpubParser {
    fn supported_formats(&self) -> &[FileFormat] {
        &[FileFormat::Epub]
    }

    fn parse(&self, input: &[u8], source: &SourceInfo) -> Result<Document, FastRagError> {
        let cursor = std::io::Cursor::new(input);
        let mut archive = zip::ZipArchive::new(cursor).map_err(|e| FastRagError::Parse {
            format: FileFormat::Epub,
            message: format!("invalid ZIP/EPUB: {e}"),
        })?;

        let mut metadata = Metadata::new(source.format);
        metadata.source_file = source.filename.clone();

        // Read container.xml to find OPF path
        let opf_path = read_container_xml(&mut archive)?;

        // Determine OPF directory for resolving relative paths
        let opf_dir = opf_path
            .rfind('/')
            .map(|i| &opf_path[..i + 1])
            .unwrap_or("");

        // Read and parse OPF
        let opf_content = read_zip_entry(&mut archive, &opf_path)?;
        let (opf_meta, spine_items) = parse_opf(&opf_content, opf_dir)?;

        if let Some(title) = opf_meta.title {
            metadata.title = Some(title);
        }
        if let Some(author) = opf_meta.author {
            metadata.author = Some(author);
        }
        if !opf_meta.language.is_empty() {
            metadata
                .custom
                .insert("language".to_string(), opf_meta.language);
        }

        // Parse each spine item in order using the HTML parser
        let html_parser = fastrag_html::HtmlParser;
        let mut all_elements = Vec::new();

        for (chapter_idx, item_path) in spine_items.iter().enumerate() {
            let content = match read_zip_entry(&mut archive, item_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let html_source = SourceInfo::new(FileFormat::Html).with_filename(item_path.as_str());
            if let Ok(doc) = html_parser.parse(content.as_bytes(), &html_source) {
                let section_name = format!("Chapter {}", chapter_idx + 1);
                for mut el in doc.elements {
                    el.section = Some(section_name.clone());
                    all_elements.push(el);
                }
            }
        }

        Ok(Document {
            metadata,
            elements: all_elements,
        })
    }
}

struct OpfMetadata {
    title: Option<String>,
    author: Option<String>,
    language: String,
}

fn read_zip_entry(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
    path: &str,
) -> Result<String, FastRagError> {
    let mut file = archive.by_name(path).map_err(|e| FastRagError::Parse {
        format: FileFormat::Epub,
        message: format!("missing entry '{path}': {e}"),
    })?;
    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|e| FastRagError::Parse {
            format: FileFormat::Epub,
            message: format!("read error for '{path}': {e}"),
        })?;
    Ok(content)
}

fn read_container_xml(
    archive: &mut zip::ZipArchive<std::io::Cursor<&[u8]>>,
) -> Result<String, FastRagError> {
    let content = read_zip_entry(archive, "META-INF/container.xml")?;

    // Parse container.xml to find rootfile full-path
    let mut reader = quick_xml::Reader::from_str(&content);
    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Empty(ref e))
            | Ok(quick_xml::events::Event::Start(ref e))
                if e.local_name().as_ref() == b"rootfile" =>
            {
                for attr in e.attributes().flatten() {
                    if attr.key.as_ref() == b"full-path" {
                        return Ok(String::from_utf8_lossy(&attr.value).to_string());
                    }
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(e) => {
                return Err(FastRagError::Parse {
                    format: FileFormat::Epub,
                    message: format!("container.xml parse error: {e}"),
                });
            }
            _ => {}
        }
    }

    Err(FastRagError::Parse {
        format: FileFormat::Epub,
        message: "no rootfile found in container.xml".to_string(),
    })
}

fn parse_opf(content: &str, opf_dir: &str) -> Result<(OpfMetadata, Vec<String>), FastRagError> {
    let mut meta = OpfMetadata {
        title: None,
        author: None,
        language: String::new(),
    };

    // manifest: id -> href
    let mut manifest: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    // spine: ordered list of idref
    let mut spine_idrefs: Vec<String> = Vec::new();

    let mut reader = quick_xml::Reader::from_str(content);
    let mut current_tag = String::new();
    let mut in_metadata = false;

    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(ref e)) => {
                let local = String::from_utf8_lossy(e.local_name().as_ref()).to_string();
                match local.as_str() {
                    "metadata" => in_metadata = true,
                    "title" | "creator" | "language" if in_metadata => {
                        current_tag = local;
                    }
                    "item" => {
                        let mut id = String::new();
                        let mut href = String::new();
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"id" => id = String::from_utf8_lossy(&attr.value).to_string(),
                                b"href" => href = String::from_utf8_lossy(&attr.value).to_string(),
                                _ => {}
                            }
                        }
                        if !id.is_empty() && !href.is_empty() {
                            manifest.insert(id, format!("{opf_dir}{href}"));
                        }
                    }
                    "itemref" => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"idref" {
                                spine_idrefs.push(String::from_utf8_lossy(&attr.value).to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(quick_xml::events::Event::Empty(ref e)) => {
                let local = String::from_utf8_lossy(e.local_name().as_ref()).to_string();
                match local.as_str() {
                    "item" => {
                        let mut id = String::new();
                        let mut href = String::new();
                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"id" => id = String::from_utf8_lossy(&attr.value).to_string(),
                                b"href" => href = String::from_utf8_lossy(&attr.value).to_string(),
                                _ => {}
                            }
                        }
                        if !id.is_empty() && !href.is_empty() {
                            manifest.insert(id, format!("{opf_dir}{href}"));
                        }
                    }
                    "itemref" => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"idref" {
                                spine_idrefs.push(String::from_utf8_lossy(&attr.value).to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(quick_xml::events::Event::Text(ref e)) => {
                if in_metadata {
                    let text = e.unescape().unwrap_or_default().trim().to_string();
                    if !text.is_empty() {
                        match current_tag.as_str() {
                            "title" if meta.title.is_none() => meta.title = Some(text),
                            "creator" if meta.author.is_none() => meta.author = Some(text),
                            "language" if meta.language.is_empty() => meta.language = text,
                            _ => {}
                        }
                    }
                }
                current_tag.clear();
            }
            Ok(quick_xml::events::Event::End(ref e)) => {
                let local = String::from_utf8_lossy(e.local_name().as_ref()).to_string();
                if local == "metadata" {
                    in_metadata = false;
                }
                current_tag.clear();
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(e) => {
                return Err(FastRagError::Parse {
                    format: FileFormat::Epub,
                    message: format!("OPF parse error: {e}"),
                });
            }
            _ => {}
        }
    }

    let spine_paths: Vec<String> = spine_idrefs
        .iter()
        .filter_map(|idref| manifest.get(idref).cloned())
        .collect();

    Ok((meta, spine_paths))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_core::ElementKind;

    #[test]
    fn supported_formats_returns_epub() {
        assert_eq!(EpubParser.supported_formats(), &[FileFormat::Epub]);
    }

    #[test]
    fn invalid_zip_returns_parse_error() {
        let parser = EpubParser;
        let source = SourceInfo::new(FileFormat::Epub);
        let result = parser.parse(b"not a zip", &source);
        assert!(result.is_err());
        match result.unwrap_err() {
            FastRagError::Parse { format, .. } => assert_eq!(format, FileFormat::Epub),
            other => panic!("expected Parse error, got: {other}"),
        }
    }

    fn create_minimal_epub(title: &str, author: &str, chapters: &[(&str, &str)]) -> Vec<u8> {
        use std::io::Write;
        let buf = Vec::new();
        let cursor = std::io::Cursor::new(buf);
        let mut zip = zip::ZipWriter::new(cursor);

        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        // mimetype (must be first, uncompressed)
        zip.start_file("mimetype", options).unwrap();
        zip.write_all(b"application/epub+zip").unwrap();

        // container.xml
        zip.start_file("META-INF/container.xml", options).unwrap();
        zip.write_all(
            br#"<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"#,
        )
        .unwrap();

        // Build manifest items and spine
        let mut manifest_items = String::new();
        let mut spine_refs = String::new();
        for (i, _) in chapters.iter().enumerate() {
            manifest_items.push_str(&format!(
                "    <item id=\"ch{i}\" href=\"chapter{i}.xhtml\" media-type=\"application/xhtml+xml\"/>\n"
            ));
            spine_refs.push_str(&format!("    <itemref idref=\"ch{i}\"/>\n"));
        }

        // content.opf
        zip.start_file("OEBPS/content.opf", options).unwrap();
        let opf = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{title}</dc:title>
    <dc:creator>{author}</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
{manifest_items}  </manifest>
  <spine>
{spine_refs}  </spine>
</package>"#
        );
        zip.write_all(opf.as_bytes()).unwrap();

        // Chapter XHTML files
        for (i, (heading, body)) in chapters.iter().enumerate() {
            zip.start_file(format!("OEBPS/chapter{i}.xhtml"), options)
                .unwrap();
            let xhtml = format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>{heading}</title></head>
<body>
<h1>{heading}</h1>
<p>{body}</p>
</body>
</html>"#
            );
            zip.write_all(xhtml.as_bytes()).unwrap();
        }

        let cursor = zip.finish().unwrap();
        cursor.into_inner()
    }

    #[test]
    fn extracts_metadata_title_author() {
        let epub = create_minimal_epub("Test Book", "Jane Doe", &[("Ch1", "Hello")]);
        let parser = EpubParser;
        let source = SourceInfo::new(FileFormat::Epub).with_filename("test.epub");
        let doc = parser.parse(&epub, &source).unwrap();
        assert_eq!(doc.metadata.title, Some("Test Book".to_string()));
        assert_eq!(doc.metadata.author, Some("Jane Doe".to_string()));
        assert_eq!(doc.metadata.custom.get("language"), Some(&"en".to_string()));
    }

    #[test]
    fn extracts_paragraphs_in_reading_order() {
        let epub = create_minimal_epub(
            "Book",
            "Author",
            &[
                ("Chapter One", "First chapter content"),
                ("Chapter Two", "Second chapter content"),
            ],
        );
        let parser = EpubParser;
        let source = SourceInfo::new(FileFormat::Epub);
        let doc = parser.parse(&epub, &source).unwrap();

        let paras: Vec<&str> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Paragraph)
            .map(|e| e.text.as_str())
            .collect();
        assert_eq!(paras.len(), 2);
        assert_eq!(paras[0], "First chapter content");
        assert_eq!(paras[1], "Second chapter content");
    }

    #[test]
    fn headings_map_to_correct_element_kinds() {
        let epub = create_minimal_epub("Book", "Author", &[("My Heading", "Content")]);
        let parser = EpubParser;
        let source = SourceInfo::new(FileFormat::Epub);
        let doc = parser.parse(&epub, &source).unwrap();

        let titles: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::Title)
            .collect();
        assert!(!titles.is_empty(), "expected at least one Title element");
        assert_eq!(titles[0].text, "My Heading");
    }

    #[test]
    fn chapters_have_section_info() {
        let epub = create_minimal_epub("Book", "Author", &[("Ch1", "First"), ("Ch2", "Second")]);
        let parser = EpubParser;
        let source = SourceInfo::new(FileFormat::Epub);
        let doc = parser.parse(&epub, &source).unwrap();

        let ch1_elements: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.section == Some("Chapter 1".to_string()))
            .collect();
        let ch2_elements: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.section == Some("Chapter 2".to_string()))
            .collect();
        assert!(!ch1_elements.is_empty(), "no Chapter 1 elements found");
        assert!(!ch2_elements.is_empty(), "no Chapter 2 elements found");
    }
}
