#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(target_os = "macos"))]
compile_error!("ironmill-coreml-sys only supports macOS");

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, bail};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureType, MLFeatureValue,
    MLModel, MLModelConfiguration, MLMultiArray, MLMultiArrayDataType as ObjcMLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};

// ── ComputeUnits ──────────────────────────────────────────────────

/// Which hardware compute units a CoreML model may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ComputeUnits {
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
    All,
}

impl fmt::Display for ComputeUnits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ComputeUnits::CpuOnly => "cpu",
            ComputeUnits::CpuAndGpu => "gpu",
            ComputeUnits::CpuAndNeuralEngine => "ane",
            ComputeUnits::All => "all",
        };
        f.write_str(s)
    }
}

impl FromStr for ComputeUnits {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cpu" => Ok(ComputeUnits::CpuOnly),
            "gpu" => Ok(ComputeUnits::CpuAndGpu),
            "ane" => Ok(ComputeUnits::CpuAndNeuralEngine),
            "all" => Ok(ComputeUnits::All),
            _ => bail!("unknown compute units: {s:?} (expected cpu, gpu, ane, or all)"),
        }
    }
}

impl ComputeUnits {
    fn to_ml(self) -> MLComputeUnits {
        match self {
            ComputeUnits::CpuOnly => MLComputeUnits::CPUOnly,
            ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            ComputeUnits::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
            ComputeUnits::All => MLComputeUnits::All,
        }
    }
}

// ── MultiArrayDataType ────────────────────────────────────────────

/// Element data type for CoreML multi-array tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MultiArrayDataType {
    Float32,
    Float16,
    Int32,
    Double,
}

impl MultiArrayDataType {
    fn to_objc(self) -> ObjcMLMultiArrayDataType {
        match self {
            MultiArrayDataType::Float32 => ObjcMLMultiArrayDataType::Float32,
            MultiArrayDataType::Float16 => ObjcMLMultiArrayDataType::Float16,
            MultiArrayDataType::Int32 => ObjcMLMultiArrayDataType::Int32,
            MultiArrayDataType::Double => ObjcMLMultiArrayDataType::Double,
        }
    }

    fn from_objc(dt: ObjcMLMultiArrayDataType) -> anyhow::Result<Self> {
        match dt {
            ObjcMLMultiArrayDataType::Float32 => Ok(MultiArrayDataType::Float32),
            ObjcMLMultiArrayDataType::Float16 => Ok(MultiArrayDataType::Float16),
            ObjcMLMultiArrayDataType::Int32 => Ok(MultiArrayDataType::Int32),
            ObjcMLMultiArrayDataType::Double => Ok(MultiArrayDataType::Double),
            other => bail!("unsupported MLMultiArrayDataType: {other:?}"),
        }
    }
}

// ── InputFeature / InputDescription ───────────────────────────────

/// A single multi-array input feature with name, shape, and element type.
#[derive(Debug, Clone)]
pub struct InputFeature {
    /// Feature name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Element data type.
    pub data_type: MultiArrayDataType,
}

/// Description of a CoreML model's input features.
#[derive(Debug, Clone)]
pub struct InputDescription {
    /// Multi-array input features required by the model.
    pub features: Vec<InputFeature>,
}

// ── Model ─────────────────────────────────────────────────────────

/// A loaded CoreML model backed by `MLModel`.
pub struct Model {
    inner: Retained<MLModel>,
}

impl Model {
    /// Load a compiled CoreML model (.mlmodelc) with the specified compute units.
    pub fn load(path: &Path, compute_units: ComputeUnits) -> anyhow::Result<Self> {
        let path_str = path.to_str().context("model path is not valid UTF-8")?;
        let ns_path = NSString::from_str(path_str);
        let url = NSURL::fileURLWithPath(&ns_path);

        // SAFETY: MLModelConfiguration::new() is an ObjC alloc+init with no preconditions.
        let config = unsafe { MLModelConfiguration::new() };
        // SAFETY: Setting a simple enum property on a freshly created configuration object.
        unsafe { config.setComputeUnits(compute_units.to_ml()) };

        // SAFETY: Loading a model from a file URL with a valid configuration.
        // The URL and config are valid ObjC objects created above.
        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
            .map_err(|e| anyhow::anyhow!("failed to load CoreML model: {e}"))?;

        Ok(Self { inner: model })
    }

    /// Query the model's input description.
    pub fn input_description(&self) -> anyhow::Result<InputDescription> {
        // SAFETY: Accessing the model description from a successfully loaded model.
        let desc = unsafe { self.inner.modelDescription() };
        // SAFETY: Accessing inputDescriptionsByName from a valid model description.
        let inputs = unsafe { desc.inputDescriptionsByName() };

        let mut features = Vec::new();

        let all_keys = inputs.allKeys();
        for i in 0..all_keys.count() {
            let key: &NSString = &all_keys.objectAtIndex(i);
            let feat_desc = inputs
                .objectForKey(key)
                .context("missing feature description")?;

            // SAFETY: Querying the feature type enum from a valid feature description.
            if unsafe { feat_desc.r#type() } != MLFeatureType::MultiArray {
                continue;
            }

            // SAFETY: Accessing the multi-array constraint from a multi-array feature.
            let constraint = unsafe { feat_desc.multiArrayConstraint() }
                .context("multi-array feature missing constraint")?;

            // SAFETY: Accessing the shape array from a valid constraint.
            let shape_arr = unsafe { constraint.shape() };
            let mut shape = Vec::new();
            for j in 0..shape_arr.count() {
                let num: &NSNumber = &shape_arr.objectAtIndex(j);
                let dim = num.as_isize();
                if dim < 0 {
                    bail!("invalid negative dimension {} in output shape", dim);
                }
                shape.push(dim as usize);
            }

            // SAFETY: Accessing the data type enum from a valid constraint.
            let data_type = MultiArrayDataType::from_objc(unsafe { constraint.dataType() })?;

            features.push(InputFeature {
                name: key.to_string(),
                shape,
                data_type,
            });
        }

        Ok(InputDescription { features })
    }

    /// Run prediction with the given feature provider input.
    pub fn predict(&self, input: &PredictionInput) -> anyhow::Result<PredictionOutput> {
        let provider: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(&*input.inner);
        // SAFETY: Running prediction with a valid model and feature provider.
        // The model was successfully loaded, and the input was constructed via safe API.
        let result = unsafe { self.inner.predictionFromFeatures_error(provider) }
            .map_err(|e| anyhow::anyhow!("prediction failed: {e}"))?;

        Ok(PredictionOutput { inner: result })
    }

    /// Extract all multi-array outputs from a prediction result as f32 data.
    ///
    /// Uses the model's output description to discover output feature names,
    /// then extracts each multi-array output into an [`OutputTensorData`].
    /// Non-multi-array outputs (e.g. dictionaries, images) are skipped.
    pub fn extract_outputs(
        &self,
        output: &PredictionOutput,
    ) -> anyhow::Result<Vec<OutputTensorData>> {
        // SAFETY: Accessing the model description from a successfully loaded model.
        let desc = unsafe { self.inner.modelDescription() };
        // SAFETY: Accessing outputDescriptionsByName from a valid model description.
        let outputs = unsafe { desc.outputDescriptionsByName() };
        let all_keys = outputs.allKeys();

        let mut result = Vec::new();

        for i in 0..all_keys.count() {
            let key: &NSString = &all_keys.objectAtIndex(i);
            let name = key.to_string();

            let feat_value = output
                .feature_value(&name)
                .with_context(|| format!("missing output feature: {name}"))?;

            // SAFETY: Querying the feature type enum from a valid feature value.
            if unsafe { feat_value.r#type() } != MLFeatureType::MultiArray {
                continue;
            }

            // SAFETY: Accessing the multi-array value from a multi-array feature value.
            let multi_array = unsafe { feat_value.multiArrayValue() }
                .with_context(|| format!("output feature '{name}' has no multi-array value"))?;

            // SAFETY: Accessing the shape array from a valid multi-array.
            let shape_arr = unsafe { multi_array.shape() };
            let mut shape = Vec::new();
            for j in 0..shape_arr.count() {
                let num: &NSNumber = &shape_arr.objectAtIndex(j);
                let dim = num.as_isize();
                if dim < 0 {
                    bail!("invalid negative dimension {} in output shape", dim);
                }
                shape.push(dim as usize);
            }

            // SAFETY: Accessing the element count from a valid multi-array.
            let count = unsafe { multi_array.count() } as usize;
            let mut data = Vec::with_capacity(count);
            for j in 0..count {
                // SAFETY: Accessing elements by index within the valid range [0, count).
                let val = unsafe { multi_array.objectAtIndexedSubscript(j as isize) };
                data.push(val.as_f32());
            }

            result.push(OutputTensorData { name, shape, data });
        }

        Ok(result)
    }
}

// ── PredictionInput ───────────────────────────────────────────────

/// Builder for CoreML prediction inputs backed by `MLDictionaryFeatureProvider`.
pub struct PredictionInput {
    inner: Retained<MLDictionaryFeatureProvider>,
}

impl PredictionInput {
    /// Create an empty prediction input.
    pub fn new() -> anyhow::Result<Self> {
        let empty_dict: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::new();
        // SAFETY: Creating a dictionary feature provider from a valid (empty) NSDictionary.
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &empty_dict,
            )
        }
        .map_err(|e| anyhow::anyhow!("failed to create empty MLDictionaryFeatureProvider: {e}"))?;

        Ok(Self { inner: provider })
    }

    /// Add a multi-array feature with the given name, shape, type, and f32 data.
    pub fn add_multi_array(
        &mut self,
        name: &str,
        shape: &[usize],
        data_type: MultiArrayDataType,
        data: &[f32],
    ) -> anyhow::Result<()> {
        let total_elements: usize = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| anyhow::anyhow!("tensor shape {:?} causes size overflow", shape))?;
        if data.len() != total_elements {
            bail!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                total_elements
            );
        }

        // Build NSArray<NSNumber> for shape
        let ns_shape_items: Vec<Retained<NSNumber>> = shape
            .iter()
            .map(|&d| NSNumber::new_isize(d as isize))
            .collect();
        let ns_shape = NSArray::from_retained_slice(&ns_shape_items);

        // SAFETY: Creating an MLMultiArray with a valid shape array and data type.
        let multi_array = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &ns_shape,
                data_type.to_objc(),
            )
        }
        .map_err(|e| anyhow::anyhow!("failed to create MLMultiArray: {e}"))?;

        // Fill the multi-array with data via indexed subscript (flat/linear indexing)
        for (i, &val) in data.iter().enumerate() {
            let value = NSNumber::new_f32(val);
            // SAFETY: Setting elements by index within the valid range [0, total_elements).
            unsafe { multi_array.setObject_atIndexedSubscript(&value, i as isize) };
        }

        // SAFETY: Creating a feature value from a valid, populated MLMultiArray.
        let feature_value = unsafe { MLFeatureValue::featureValueWithMultiArray(&multi_array) };

        // Build a new dictionary merging existing entries with this one
        // SAFETY: Accessing the dictionary from a valid feature provider.
        let existing = unsafe { self.inner.dictionary() };
        let existing_keys = existing.allKeys();

        let mut owned_keys: Vec<Retained<NSString>> = Vec::new();
        let mut all_values: Vec<Retained<objc2::runtime::AnyObject>> = Vec::new();

        for i in 0..existing_keys.count() {
            let k: Retained<NSString> = existing_keys.objectAtIndex(i);
            if let Some(v) = existing.objectForKey(&k) {
                owned_keys.push(k);
                all_values.push(v.clone().into());
            }
        }

        let ns_name = NSString::from_str(name);
        let fv_as_any: Retained<objc2::runtime::AnyObject> =
            Retained::into_super(Retained::into_super(feature_value));
        owned_keys.push(ns_name);
        all_values.push(fv_as_any);

        let all_keys: Vec<&NSString> = owned_keys.iter().map(|k| &**k).collect();
        let all_value_refs: Vec<&objc2::runtime::AnyObject> =
            all_values.iter().map(|v| &**v).collect();

        let merged: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::from_slices(&all_keys, &all_value_refs);

        // SAFETY: Creating a dictionary feature provider from a valid merged NSDictionary.
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &merged,
            )
        }
        .map_err(|e| anyhow::anyhow!("failed to create MLDictionaryFeatureProvider: {e}"))?;

        self.inner = provider;
        Ok(())
    }
}

// ── PredictionOutput ──────────────────────────────────────────────

/// Output of a CoreML model prediction, wrapping an `MLFeatureProvider`.
pub struct PredictionOutput {
    inner: Retained<ProtocolObject<dyn MLFeatureProvider>>,
}

/// An extracted multi-array output with name, shape, and f32 data.
#[derive(Debug, Clone)]
pub struct ExtractedOutput {
    /// Output feature name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Tensor data as f32 values.
    pub data: Vec<f32>,
}

impl PredictionOutput {
    /// Get the underlying feature provider.
    pub fn as_feature_provider(&self) -> &ProtocolObject<dyn MLFeatureProvider> {
        &self.inner
    }

    /// Get a feature value by name.
    pub fn feature_value(&self, name: &str) -> Option<Retained<MLFeatureValue>> {
        let ns_name = NSString::from_str(name);
        // SAFETY: Querying a feature value by name from a valid feature provider.
        unsafe { self.inner.featureValueForName(&ns_name) }
    }

    /// Get the names of all output features.
    pub fn feature_names(&self) -> Vec<String> {
        // SAFETY: Accessing the feature names set from a valid feature provider.
        let names = unsafe { self.inner.featureNames() };
        let all_objects = names.allObjects();
        let mut result = Vec::new();
        for i in 0..all_objects.count() {
            let name_ns: &NSString = &all_objects.objectAtIndex(i);
            result.push(name_ns.to_string());
        }
        result
    }

    /// Extract all multi-array outputs as f32 vectors.
    ///
    /// Non-multi-array features are skipped. Each element is extracted
    /// via indexed subscript from the underlying `MLMultiArray`.
    pub fn extract_multi_arrays(&self) -> anyhow::Result<Vec<ExtractedOutput>> {
        let names = self.feature_names();
        let mut outputs = Vec::new();

        for name in &names {
            let Some(feature_value) = self.feature_value(name) else {
                continue;
            };
            // SAFETY: Querying the feature type enum from a valid feature value.
            let ft = unsafe { feature_value.r#type() };
            if ft != MLFeatureType::MultiArray {
                continue;
            }
            // SAFETY: Accessing the multi-array value from a multi-array feature value.
            let Some(multi_array) = (unsafe { feature_value.multiArrayValue() }) else {
                continue;
            };

            // SAFETY: Accessing the shape array from a valid multi-array.
            let shape_arr = unsafe { multi_array.shape() };
            let mut shape = Vec::new();
            for j in 0..shape_arr.count() {
                let num: &NSNumber = &shape_arr.objectAtIndex(j);
                let dim = num.as_isize();
                if dim < 0 {
                    bail!("invalid negative dimension {} in output shape", dim);
                }
                shape.push(dim as usize);
            }

            let total: usize = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| anyhow::anyhow!("tensor shape {:?} causes size overflow", shape))?;
            let mut data = Vec::with_capacity(total);
            for idx in 0..total {
                // SAFETY: Accessing elements by index within the valid range [0, total).
                let val = unsafe { multi_array.objectAtIndexedSubscript(idx as isize) };
                data.push(val.as_f32());
            }

            outputs.push(ExtractedOutput {
                name: name.clone(),
                shape,
                data,
            });
        }

        Ok(outputs)
    }
}

// ── OutputTensorData ──────────────────────────────────────────────

/// Data from a single output tensor, extracted as f32 values.
#[derive(Debug, Clone)]
pub struct OutputTensorData {
    /// Feature name from the model.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Flattened data converted to f32.
    pub data: Vec<f32>,
}

// ── build_dummy_input ─────────────────────────────────────────────

/// Build a dummy input (zeros) from a model's input description, suitable for benchmarking.
pub fn build_dummy_input(desc: &InputDescription) -> anyhow::Result<PredictionInput> {
    let mut input = PredictionInput::new()?;
    for feat in &desc.features {
        let total: usize = feat
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| anyhow::anyhow!("tensor shape {:?} causes size overflow", feat.shape))?;
        let data = vec![0.0f32; total];
        input.add_multi_array(&feat.name, &feat.shape, feat.data_type, &data)?;
    }
    Ok(input)
}
