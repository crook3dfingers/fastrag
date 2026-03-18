use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::format::FileFormat;

/// A parsed document containing metadata and structured elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub metadata: Metadata,
    pub elements: Vec<Element>,
}

/// Metadata about the source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub source_file: Option<String>,
    pub format: FileFormat,
    pub title: Option<String>,
    pub author: Option<String>,
    pub page_count: Option<usize>,
    pub created_at: Option<String>,
    #[serde(flatten)]
    pub custom: HashMap<String, String>,
}

impl Metadata {
    pub fn new(format: FileFormat) -> Self {
        Self {
            source_file: None,
            format,
            title: None,
            author: None,
            page_count: None,
            created_at: None,
            custom: HashMap::new(),
        }
    }
}

/// A bounding box for an element's position on a page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub page: usize,
}

/// A single structural element extracted from a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    pub kind: ElementKind,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    pub depth: u8,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounding_box: Option<BoundingBox>,
}

impl Element {
    pub fn new(kind: ElementKind, text: impl Into<String>) -> Self {
        Self {
            kind,
            text: text.into(),
            page: None,
            section: None,
            depth: 0,
            attributes: HashMap::new(),
            id: String::new(),
            parent_id: None,
            children: Vec::new(),
            bounding_box: None,
        }
    }

    pub fn with_depth(mut self, depth: u8) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_page(mut self, page: usize) -> Self {
        self.page = Some(page);
        self
    }

    pub fn with_section(mut self, section: impl Into<String>) -> Self {
        self.section = Some(section.into());
        self
    }

    pub fn with_bounding_box(mut self, bbox: BoundingBox) -> Self {
        self.bounding_box = Some(bbox);
        self
    }
}

impl Document {
    /// Assign sequential IDs and build parent-child hierarchy based on heading structure.
    pub fn build_hierarchy(&mut self) {
        // Pass 1: assign sequential IDs
        for (i, el) in self.elements.iter_mut().enumerate() {
            el.id = format!("el-{i}");
        }

        // Pass 2: assign parent_id using heading stack
        // Stack entries: (heading_depth, element_index)
        let mut stack: Vec<(u8, usize)> = Vec::new();

        for i in 0..self.elements.len() {
            let kind = self.elements[i].kind.clone();
            let depth = self.elements[i].depth;

            match kind {
                ElementKind::Title | ElementKind::Heading => {
                    // Effective depth: Title=0, Heading uses its depth field
                    let effective_depth = if kind == ElementKind::Title { 0 } else { depth };

                    // Pop stack entries with depth >= current
                    while let Some(&(d, _)) = stack.last() {
                        if d >= effective_depth {
                            stack.pop();
                        } else {
                            break;
                        }
                    }

                    // If stack is non-empty, this heading is a child of the top
                    if let Some(&(_, parent_idx)) = stack.last() {
                        let parent_id = self.elements[parent_idx].id.clone();
                        self.elements[i].parent_id = Some(parent_id);
                    }

                    stack.push((effective_depth, i));
                }
                _ => {
                    // Content element: parent is top of stack
                    if let Some(&(_, parent_idx)) = stack.last() {
                        let parent_id = self.elements[parent_idx].id.clone();
                        self.elements[i].parent_id = Some(parent_id);
                    }
                }
            }
        }

        // Pass 3: populate children vecs from parent_id references
        // Collect (parent_index, child_id) pairs first to avoid borrow issues
        let mut child_map: Vec<(usize, String)> = Vec::new();
        for el in &self.elements {
            if let Some(ref pid) = el.parent_id
                && let Some(parent_idx) = self.elements.iter().position(|e| e.id == *pid)
            {
                child_map.push((parent_idx, el.id.clone()));
            }
        }
        for (parent_idx, child_id) in child_map {
            self.elements[parent_idx].children.push(child_id);
        }
    }
}

/// Check if text looks like a figure/table caption.
pub fn is_caption_text(text: &str) -> bool {
    let lower = text.to_lowercase();
    let trimmed = lower.trim_start();
    let prefixes = [
        "figure ",
        "figure\u{a0}",
        "fig. ",
        "fig ",
        "table ",
        "image ",
        "illustration ",
        "plate ",
    ];
    for prefix in &prefixes {
        if let Some(rest) = trimmed.strip_prefix(prefix)
            && rest.starts_with(|c: char| c.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

impl Document {
    /// Associate Image elements with adjacent caption paragraphs.
    /// Requires IDs to be assigned (call `build_hierarchy()` first).
    pub fn associate_captions(&mut self) {
        let len = self.elements.len();
        let mut associated: Vec<bool> = vec![false; len];

        for i in 0..len {
            if self.elements[i].kind != ElementKind::Image || associated[i] {
                continue;
            }

            // Scan window i-3..i+3 for caption paragraphs
            let start = i.saturating_sub(3);
            let end = (i + 4).min(len);

            // Prefer after, then before
            let mut best: Option<usize> = None;
            let mut best_distance: usize = usize::MAX;
            let mut prefer_after = true;

            for (j, is_assoc) in associated.iter().enumerate().take(end).skip(start) {
                if j == i || *is_assoc {
                    continue;
                }
                if self.elements[j].kind != ElementKind::Paragraph {
                    continue;
                }
                if !is_caption_text(&self.elements[j].text) {
                    continue;
                }

                let distance = j.abs_diff(i);
                let is_after = j > i;

                let is_better = match best {
                    None => true,
                    Some(_) => {
                        if is_after && !prefer_after {
                            distance <= best_distance
                        } else {
                            distance < best_distance
                        }
                    }
                };

                if is_better {
                    best = Some(j);
                    best_distance = distance;
                    prefer_after = is_after;
                }
            }

            if let Some(caption_idx) = best {
                let image_id = self.elements[i].id.clone();
                let caption_id = self.elements[caption_idx].id.clone();
                self.elements[i]
                    .attributes
                    .insert("associated_caption_id".to_string(), caption_id);
                self.elements[caption_idx]
                    .attributes
                    .insert("associated_image_id".to_string(), image_id);
                associated[i] = true;
                associated[caption_idx] = true;
            }
        }
    }
}

/// The kind of structural element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementKind {
    Title,
    Heading,
    Paragraph,
    Table,
    Code,
    List,
    ListItem,
    Image,
    BlockQuote,
    HorizontalRule,
    FormField,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_new_defaults() {
        let el = Element::new(ElementKind::Paragraph, "hello");
        assert_eq!(el.kind, ElementKind::Paragraph);
        assert_eq!(el.text, "hello");
        assert_eq!(el.depth, 0);
        assert_eq!(el.page, None);
        assert_eq!(el.section, None);
        assert!(el.attributes.is_empty());
    }

    #[test]
    fn element_with_depth() {
        let el = Element::new(ElementKind::Heading, "h").with_depth(2);
        assert_eq!(el.depth, 2);
    }

    #[test]
    fn element_with_page() {
        let el = Element::new(ElementKind::Paragraph, "p").with_page(5);
        assert_eq!(el.page, Some(5));
    }

    #[test]
    fn element_with_section() {
        let el = Element::new(ElementKind::Paragraph, "p").with_section("intro");
        assert_eq!(el.section, Some("intro".to_string()));
    }

    #[test]
    fn element_builder_chaining() {
        let el = Element::new(ElementKind::Code, "x = 1")
            .with_depth(1)
            .with_page(3)
            .with_section("code");
        assert_eq!(el.depth, 1);
        assert_eq!(el.page, Some(3));
        assert_eq!(el.section, Some("code".to_string()));
        assert_eq!(el.text, "x = 1");
    }

    #[test]
    fn element_new_has_empty_hierarchy_fields() {
        let el = Element::new(ElementKind::Paragraph, "hello");
        assert!(el.id.is_empty());
        assert_eq!(el.parent_id, None);
        assert!(el.children.is_empty());
    }

    #[test]
    fn build_hierarchy_assigns_sequential_ids() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Paragraph, "a"),
                Element::new(ElementKind::Paragraph, "b"),
                Element::new(ElementKind::Paragraph, "c"),
            ],
        };
        doc.build_hierarchy();
        assert_eq!(doc.elements[0].id, "el-0");
        assert_eq!(doc.elements[1].id, "el-1");
        assert_eq!(doc.elements[2].id, "el-2");
    }

    #[test]
    fn build_hierarchy_heading_parents_content() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Title, "Title"),
                Element::new(ElementKind::Paragraph, "intro"),
                Element::new(ElementKind::Heading, "Section 1").with_depth(1),
                Element::new(ElementKind::Paragraph, "body"),
            ],
        };
        doc.build_hierarchy();
        // intro's parent is Title
        assert_eq!(doc.elements[1].parent_id, Some("el-0".to_string()));
        // Section 1's parent is Title (depth 1 > depth 0)
        assert_eq!(doc.elements[2].parent_id, Some("el-0".to_string()));
        // body's parent is Section 1
        assert_eq!(doc.elements[3].parent_id, Some("el-2".to_string()));
    }

    #[test]
    fn build_hierarchy_same_level_resets() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Heading, "H1-a").with_depth(1),
                Element::new(ElementKind::Paragraph, "under a"),
                Element::new(ElementKind::Heading, "H1-b").with_depth(1),
                Element::new(ElementKind::Paragraph, "under b"),
            ],
        };
        doc.build_hierarchy();
        // "under a" is child of H1-a
        assert_eq!(doc.elements[1].parent_id, Some("el-0".to_string()));
        // H1-b has no parent (same level, stack popped)
        assert_eq!(doc.elements[2].parent_id, None);
        // "under b" is child of H1-b
        assert_eq!(doc.elements[3].parent_id, Some("el-2".to_string()));
    }

    #[test]
    fn build_hierarchy_nested_headings() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Heading, "H1").with_depth(1),
                Element::new(ElementKind::Heading, "H2").with_depth(2),
                Element::new(ElementKind::Paragraph, "content"),
            ],
        };
        doc.build_hierarchy();
        // H2's parent is H1
        assert_eq!(doc.elements[1].parent_id, Some("el-0".to_string()));
        // content's parent is H2
        assert_eq!(doc.elements[2].parent_id, Some("el-1".to_string()));
    }

    #[test]
    fn build_hierarchy_children_populated() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Title, "Doc Title"),
                Element::new(ElementKind::Paragraph, "intro"),
                Element::new(ElementKind::Heading, "Sec").with_depth(1),
            ],
        };
        doc.build_hierarchy();
        // Title should have children: el-1 (Paragraph) and el-2 (Heading)
        assert_eq!(doc.elements[0].children, vec!["el-1", "el-2"]);
    }

    #[test]
    fn json_output_includes_hierarchy() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Title, "Title"),
                Element::new(ElementKind::Paragraph, "text"),
            ],
        };
        doc.build_hierarchy();
        let json = serde_json::to_string(&doc).unwrap();
        assert!(json.contains("\"id\":\"el-0\""), "missing id in json");
        assert!(
            json.contains("\"parent_id\":\"el-0\""),
            "missing parent_id in json"
        );
        assert!(
            json.contains("\"children\":[\"el-1\"]"),
            "missing children in json: {json}"
        );
    }

    #[test]
    fn is_caption_text_matches_common_patterns() {
        assert!(is_caption_text("Figure 1: Revenue Growth"));
        assert!(is_caption_text("Fig. 2. Comparison"));
        assert!(is_caption_text("Table 3: Results"));
        assert!(is_caption_text("Image 4 - Photo"));
        assert!(is_caption_text("Illustration 1: Diagram"));
        assert!(is_caption_text("Plate 5: Specimen"));
        assert!(is_caption_text("fig 7 something"));
        assert!(is_caption_text("FIGURE 1: UPPERCASE"));
    }

    #[test]
    fn is_caption_text_rejects_normal_text() {
        assert!(!is_caption_text("Regular paragraph"));
        assert!(!is_caption_text("The figure below shows"));
        assert!(!is_caption_text("Table of contents"));
        assert!(!is_caption_text(""));
        assert!(!is_caption_text("Figure without number"));
    }

    #[test]
    fn associate_captions_links_adjacent() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Image, "photo.jpg"),
                Element::new(ElementKind::Paragraph, "Figure 1: A chart"),
            ],
        };
        doc.build_hierarchy();
        doc.associate_captions();
        assert_eq!(
            doc.elements[0].attributes.get("associated_caption_id"),
            Some(&"el-1".to_string())
        );
        assert_eq!(
            doc.elements[1].attributes.get("associated_image_id"),
            Some(&"el-0".to_string())
        );
    }

    #[test]
    fn associate_captions_ignores_distant() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Image, "photo.jpg"),
                Element::new(ElementKind::Paragraph, "a"),
                Element::new(ElementKind::Paragraph, "b"),
                Element::new(ElementKind::Paragraph, "c"),
                Element::new(ElementKind::Paragraph, "d"),
                Element::new(ElementKind::Paragraph, "e"),
                Element::new(ElementKind::Paragraph, "f"),
                Element::new(ElementKind::Paragraph, "g"),
                Element::new(ElementKind::Paragraph, "h"),
                Element::new(ElementKind::Paragraph, "i"),
                Element::new(ElementKind::Paragraph, "Figure 1: distant caption"),
            ],
        };
        doc.build_hierarchy();
        doc.associate_captions();
        assert!(
            doc.elements[0]
                .attributes
                .get("associated_caption_id")
                .is_none()
        );
    }

    #[test]
    fn associate_captions_prefers_after_over_before() {
        let mut doc = Document {
            metadata: Metadata::new(FileFormat::Html),
            elements: vec![
                Element::new(ElementKind::Paragraph, "Figure 1: before"),
                Element::new(ElementKind::Image, "photo.jpg"),
                Element::new(ElementKind::Paragraph, "Figure 2: after"),
            ],
        };
        doc.build_hierarchy();
        doc.associate_captions();
        // Should prefer the one after
        assert_eq!(
            doc.elements[1].attributes.get("associated_caption_id"),
            Some(&"el-2".to_string())
        );
    }

    #[test]
    fn metadata_new_defaults() {
        let m = Metadata::new(FileFormat::Html);
        assert_eq!(m.format, FileFormat::Html);
        assert_eq!(m.source_file, None);
        assert_eq!(m.title, None);
        assert_eq!(m.author, None);
        assert_eq!(m.page_count, None);
        assert_eq!(m.created_at, None);
        assert!(m.custom.is_empty());
    }

    // --- BoundingBox tests ---

    #[test]
    fn bounding_box_default_none() {
        let el = Element::new(ElementKind::Paragraph, "text");
        assert!(el.bounding_box.is_none());
    }

    #[test]
    fn with_bounding_box_builder() {
        let bbox = BoundingBox {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
            page: 1,
        };
        let el = Element::new(ElementKind::Paragraph, "text").with_bounding_box(bbox);
        let bb = el.bounding_box.unwrap();
        assert_eq!(bb.x, 10.0);
        assert_eq!(bb.y, 20.0);
        assert_eq!(bb.width, 100.0);
        assert_eq!(bb.height, 50.0);
        assert_eq!(bb.page, 1);
    }

    #[test]
    fn bounding_box_serializes_when_present() {
        let bbox = BoundingBox {
            x: 5.0,
            y: 10.0,
            width: 200.0,
            height: 30.0,
            page: 2,
        };
        let el = Element::new(ElementKind::Paragraph, "text").with_bounding_box(bbox);
        let json = serde_json::to_string(&el).unwrap();
        assert!(json.contains("\"bounding_box\""), "json: {json}");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["bounding_box"]["x"], 5.0);
        assert_eq!(parsed["bounding_box"]["page"], 2);
    }

    #[test]
    fn bounding_box_absent_when_none() {
        let el = Element::new(ElementKind::Paragraph, "text");
        let json = serde_json::to_string(&el).unwrap();
        assert!(
            !json.contains("bounding_box"),
            "should omit bounding_box when None, got: {json}"
        );
    }
}
