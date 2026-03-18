use fastrag_core::{Element, ElementKind};
use pdf::object::Resolve;

/// Extract form fields from a PDF's interactive form dictionary.
pub fn extract_form_fields(
    pdf: &pdf::file::CachedFile<impl AsRef<[u8]>>,
    resolver: &impl Resolve,
) -> Vec<Element> {
    let mut elements = Vec::new();

    let forms = match pdf.trailer.root.forms {
        Some(ref f) => f,
        None => return elements,
    };

    for field_ref in &forms.fields {
        extract_field(&**field_ref, resolver, &mut elements);
    }

    elements
}

fn field_type_str(ft: &pdf::object::FieldType) -> &'static str {
    match ft {
        pdf::object::FieldType::Button => "Button",
        pdf::object::FieldType::Text => "Text",
        pdf::object::FieldType::Choice => "Choice",
        pdf::object::FieldType::Signature => "Signature",
        pdf::object::FieldType::SignatureReference => "SignatureReference",
    }
}

fn primitive_to_string(p: &pdf::primitive::Primitive) -> String {
    match p {
        pdf::primitive::Primitive::Null => String::new(),
        pdf::primitive::Primitive::Integer(i) => i.to_string(),
        pdf::primitive::Primitive::Number(n) => n.to_string(),
        pdf::primitive::Primitive::Boolean(b) => b.to_string(),
        pdf::primitive::Primitive::String(s) => s.to_string().unwrap_or_default(),
        pdf::primitive::Primitive::Name(n) => n.to_string(),
        pdf::primitive::Primitive::Array(a) => {
            let items: Vec<String> = a.iter().map(primitive_to_string).collect();
            items.join(", ")
        }
        _ => String::new(),
    }
}

fn extract_field(
    field: &pdf::object::FieldDictionary,
    resolver: &impl Resolve,
    elements: &mut Vec<Element>,
) {
    let name = field
        .name
        .as_ref()
        .and_then(|n| n.to_string().ok())
        .unwrap_or_default();

    let value = primitive_to_string(&field.value);

    let type_str = field.typ.as_ref().map(field_type_str);

    // If the field has a name and either a value or a type, emit it as an element
    if !name.is_empty() || !value.is_empty() {
        let mut el = Element::new(ElementKind::FormField, &value);
        el.attributes.insert("field_name".to_string(), name.clone());
        if let Some(t) = type_str {
            el.attributes
                .insert("field_type".to_string(), t.to_string());
        }
        elements.push(el);
    }

    // Recurse into kids
    for kid_ref in &field.kids {
        if let Ok(kid) = resolver.get(*kid_ref) {
            if let Some(ref kid_field) = kid.a {
                extract_field(kid_field, resolver, elements);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrag_core::{FileFormat, Parser, SourceInfo};

    #[test]
    fn extract_form_fields_from_fixture() {
        let pdf_bytes = include_bytes!("../../../tests/fixtures/sample_form.pdf");
        let parser = crate::PdfParser;
        let source = SourceInfo::new(FileFormat::Pdf).with_filename("sample_form.pdf");
        let doc = parser.parse(pdf_bytes, &source).unwrap();

        let form_fields: Vec<_> = doc
            .elements
            .iter()
            .filter(|e| e.kind == ElementKind::FormField)
            .collect();
        assert!(
            !form_fields.is_empty(),
            "expected form field elements, got none"
        );

        // Verify attributes are populated
        for field in &form_fields {
            assert!(
                field.attributes.contains_key("field_name"),
                "missing field_name attribute"
            );
        }
    }

    #[test]
    fn form_fields_no_forms_returns_empty() {
        // sample.pdf has no forms
        let pdf_bytes = include_bytes!("../../../tests/fixtures/sample.pdf");
        let pdf = pdf::file::FileOptions::cached()
            .load(pdf_bytes.to_vec())
            .unwrap();
        let resolver = pdf.resolver();
        let elements = extract_form_fields(&pdf, &resolver);
        assert!(elements.is_empty());
    }
}
