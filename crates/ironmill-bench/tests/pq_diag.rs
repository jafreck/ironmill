#[cfg(feature = "metal")]
#[test]
#[ignore]
fn tq_short_vs_long() {
    use ironmill_compile::weights::{SafeTensorsProvider, WeightProvider};
    use ironmill_inference::InferenceEngine;
    use ironmill_inference::gpu::{GpuConfig, GpuInference};
    let manifest = env!("CARGO_MANIFEST_DIR");
    let model_dir = std::path::PathBuf::from(manifest).join("../../tests/fixtures/Qwen3-0.6B");
    let st = SafeTensorsProvider::load(&model_dir).unwrap();
    let dp = std::path::PathBuf::from(manifest).join("../../tests/fixtures/quality/wikitext2-qwen3.json");
    let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&dp).unwrap()).unwrap();
    let seq: Vec<u32> = data["sequences"][0].as_array().unwrap().iter()
        .map(|v| v.as_u64().unwrap() as u32).collect();

    let eval = |config: GpuConfig, n_tokens: usize, label: &str| {
        let mut engine = GpuInference::new(config.clone()).unwrap();
        engine.load_weights(&st, config).unwrap();
        engine.reset();
        let mut ce = 0.0f64;
        for pos in 0..n_tokens.min(seq.len()-1) {
            let logits = engine.decode_step(seq[pos]).unwrap();
            if logits.iter().any(|v| v.is_nan()) { println!("{label}: NaN at step {pos}"); return; }
            let t = seq[pos+1] as usize;
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let se: f64 = logits.iter().map(|&x| ((x-mx) as f64).exp()).sum();
            ce -= (logits[t]-mx) as f64 - se.ln();
        }
        let n = n_tokens.min(seq.len()-1);
        let ppl = (ce / n as f64).exp();
        println!("{label:<45} PPL={ppl:>8.2}  ({n} tokens)");
    };

    for n in [20, 50, 100, 150] {
        eval(GpuConfig { n_bits: 8, ..GpuConfig::default() }, n, &format!("INT8 KV ({n} tok)"));
        eval(GpuConfig { n_bits: 4, ..GpuConfig::default() }, n, &format!("INT4 KV ({n} tok)"));
    }
}
