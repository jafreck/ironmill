//! Reader for `.mlmodel` files (single protobuf binary).

use std::path::Path;

use prost::Message;

use crate::error::Result;
use crate::proto::specification::{Model, ModelDescription};

/// Read a `.mlmodel` file and deserialize it into a [`Model`] struct.
///
/// `.mlmodel` files are single protobuf binary files containing the full
/// CoreML model specification.
///
/// # Errors
///
/// Returns [`MilError::Io`] if the file cannot be read, or
/// [`MilError::Protobuf`] if the bytes are not valid protobuf.
///
/// # Examples
///
/// ```no_run
/// use mil_rs::reader::read_mlmodel;
///
/// let model = read_mlmodel("path/to/model.mlmodel").unwrap();
/// println!("spec version: {}", model.specification_version);
/// ```
pub fn read_mlmodel(path: impl AsRef<Path>) -> Result<Model> {
    let bytes = std::fs::read(path.as_ref())?;
    let model = Model::decode(&bytes[..])?;
    Ok(model)
}

/// Print a human-readable summary of a CoreML model to stdout.
pub fn print_model_summary(model: &Model) {
    tracing::info!("CoreML Model Summary");
    tracing::info!("====================");
    tracing::info!("Specification version: {}", model.specification_version);
    tracing::info!("Updatable: {}", model.is_updatable);

    if let Some(desc) = &model.description {
        print_description(desc);
    }

    if let Some(ref ty) = model.r#type {
        tracing::info!("Model type: {}", model_type_name(ty));
    }
}

fn print_description(desc: &ModelDescription) {
    if let Some(meta) = &desc.metadata {
        if !meta.short_description.is_empty() {
            tracing::info!("Description: {}", meta.short_description);
        }
        if !meta.author.is_empty() {
            tracing::info!("Author: {}", meta.author);
        }
        if !meta.version_string.is_empty() {
            tracing::info!("Version: {}", meta.version_string);
        }
        if !meta.license.is_empty() {
            tracing::info!("License: {}", meta.license);
        }
    }

    if !desc.input.is_empty() {
        tracing::info!("Inputs ({}):", desc.input.len());
        for f in &desc.input {
            tracing::info!("  - {}", feature_summary(f));
        }
    }

    if !desc.output.is_empty() {
        tracing::info!("Outputs ({}):", desc.output.len());
        for f in &desc.output {
            tracing::info!("  - {}", feature_summary(f));
        }
    }
}

fn feature_summary(f: &crate::proto::specification::FeatureDescription) -> String {
    if f.short_description.is_empty() {
        f.name.clone()
    } else {
        format!("{} — {}", f.name, f.short_description)
    }
}

fn model_type_name(ty: &crate::proto::specification::model::Type) -> &'static str {
    use crate::proto::specification::model::Type;
    match ty {
        Type::PipelineClassifier(_) => "PipelineClassifier",
        Type::PipelineRegressor(_) => "PipelineRegressor",
        Type::Pipeline(_) => "Pipeline",
        Type::GlmRegressor(_) => "GLMRegressor",
        Type::SupportVectorRegressor(_) => "SupportVectorRegressor",
        Type::TreeEnsembleRegressor(_) => "TreeEnsembleRegressor",
        Type::NeuralNetworkRegressor(_) => "NeuralNetworkRegressor",
        Type::BayesianProbitRegressor(_) => "BayesianProbitRegressor",
        Type::GlmClassifier(_) => "GLMClassifier",
        Type::SupportVectorClassifier(_) => "SupportVectorClassifier",
        Type::TreeEnsembleClassifier(_) => "TreeEnsembleClassifier",
        Type::NeuralNetworkClassifier(_) => "NeuralNetworkClassifier",
        Type::NeuralNetwork(_) => "NeuralNetwork",
        Type::CustomModel(_) => "CustomModel",
        Type::LinkedModel(_) => "LinkedModel",
        Type::MlProgram(_) => "MLProgram",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MilError;

    fn fixture_path(name: &str) -> std::path::PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest)
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    #[test]
    #[ignore] // Requires test fixture files
    fn read_mobilenet_mlmodel() {
        let path = fixture_path("MobileNet.mlmodel");
        let model = read_mlmodel(&path).expect("failed to read MobileNet.mlmodel");

        assert!(model.specification_version > 0);
        let desc = model
            .description
            .as_ref()
            .expect("model has no description");
        assert!(!desc.input.is_empty(), "model should have inputs");
        assert!(!desc.output.is_empty(), "model should have outputs");
    }

    #[test]
    #[ignore] // Requires test fixture files
    fn read_mobilenet_metadata() {
        let path = fixture_path("MobileNet.mlmodel");
        let model = read_mlmodel(&path).unwrap();

        // Just verify print_model_summary doesn't panic.
        print_model_summary(&model);
    }

    #[test]
    fn error_on_missing_file() {
        let result = read_mlmodel("nonexistent.mlmodel");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, MilError::Io(_)),
            "expected Io error, got: {err}"
        );
    }

    #[test]
    fn error_on_invalid_protobuf() {
        let dir = env!("CARGO_MANIFEST_DIR");
        // Use Cargo.toml as an invalid protobuf file.
        let path = Path::new(dir).join("Cargo.toml");
        let result = read_mlmodel(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, MilError::ProtobufDecode(_)),
            "expected ProtobufDecode error, got: {err}"
        );
    }
}
