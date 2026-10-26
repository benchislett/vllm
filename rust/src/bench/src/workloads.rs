// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::cli::{BackendKind, BenchServeArgs, DatasetName};
use crate::config::BenchConfig;
use crate::error::{BenchError, Result};

/// Fixed, versioned random chat workloads shared with the Python client.
#[derive(clap::ValueEnum, Debug, Clone, Copy, serde::Serialize)]
pub enum Workload {
    #[value(name = "1k1k-v1")]
    #[serde(rename = "1k1k-v1")]
    OneKOneK,
    #[value(name = "8k1k-v1")]
    #[serde(rename = "8k1k-v1")]
    EightKOneK,
}

/// On-disk measurement format; the benchmark return value remains unchanged.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum ReportFormat {
    #[default]
    Legacy,
    Byog,
}

impl Workload {
    pub fn apply(self, args: &mut BenchServeArgs) -> Result<()> {
        let concurrency = args.max_concurrency.filter(|n| *n > 0).ok_or_else(|| {
            BenchError::Config("--workload requires a positive --max-concurrency".into())
        })?;
        if let Some(body) = &args.extra_body {
            let body: Value = serde_json::from_str(body)?;
            if [
                "prompt",
                "messages",
                "max_tokens",
                "max_completion_tokens",
                "ignore_eos",
                "n",
            ]
            .iter()
            .any(|key| body.get(key).is_some())
            {
                return Err(BenchError::Config(
                    "--extra-body overrides fixed workload request fields".into(),
                ));
            }
        }
        args.backend = BackendKind::OpenaiChat;
        args.endpoint = Some("/v1/chat/completions".into());
        args.dataset_name = DatasetName::Random;
        args.random_input_len = match self {
            Self::OneKOneK => 1024,
            Self::EightKOneK => 8192,
        };
        args.random_output_len = 1024;
        args.random_range_ratio = "0.2".into();
        args.random_prefix_len = 0;
        args.seed = 0;
        args.num_warmups = 1;
        args.num_prompts = concurrency
            .checked_mul(10)
            .ok_or_else(|| BenchError::Config("Workload request count overflow".into()))?
            .max(20);
        args.request_rate = f64::INFINITY;
        Ok(())
    }
}

/// Require disabled caching and read hit counters across all engines.
pub async fn prefix_cache_snapshot(
    client: &reqwest::Client,
    config: &BenchConfig,
) -> Result<BTreeMap<String, f64>> {
    let mut request = client
        .get(format!("{}/metrics", config.base_url.trim_end_matches('/')))
        .timeout(std::time::Duration::from_secs(10));
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        request = request.bearer_auth(key);
    }
    if let Some(headers) = &config.extra_headers {
        for (name, value) in headers {
            request = request.header(name, value);
        }
    }
    let text = request.send().await?.error_for_status()?.text().await?;
    let mut configurations = 0;
    let mut hits = BTreeMap::new();
    for line in text.lines().map(str::trim).filter(|line| !line.starts_with('#')) {
        let name = line.split(['{', ' ']).next().unwrap_or("");
        if name == "vllm:cache_config_info" {
            configurations += 1;
            if !line.contains("enable_prefix_caching=\"False\"") {
                return Err(BenchError::Config(
                    "--workload requires --no-enable-prefix-caching on the server".into(),
                ));
            }
        }
        if matches!(
            name,
            "vllm:prefix_cache_hits_total" | "vllm:external_prefix_cache_hits_total"
        ) {
            let value = line
                .rsplit_once('}')
                .map_or_else(|| line.strip_prefix(name).unwrap_or(""), |(_, value)| value)
                .split_whitespace()
                .next();
            let value = value
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|value| value.is_finite());
            if let Some(value) = value {
                let series = line.rsplit_once('}').map_or(name, |(labels, _)| labels);
                hits.insert(series.to_owned(), value);
            } else {
                return Err(BenchError::Config("Invalid prefix cache counter".into()));
            }
        }
    }
    if configurations == 0 || hits.is_empty() {
        return Err(BenchError::Config("Cannot verify disabled prefix caching: expose cache configuration and counters at /metrics".into()));
    }
    Ok(hits)
}

pub fn metadata(config: &BenchConfig) -> Value {
    json!({"name": config.workload, "resolved": {
        "dataset_name": "random", "backend": "openai-chat", "endpoint": "/v1/chat/completions",
        "random_input_len": config.random_input_len, "random_output_len": config.random_output_len,
        "random_range_ratio": 0.2, "random_prefix_len": 0, "seed": 0,
        "num_prompts": config.num_prompts, "num_warmups": 1, "request_rate": "inf"
    }})
}

/// Wrap the native result while preserving raw per-request records.
pub fn measurement(config: &BenchConfig, result: &Value, cache_error: Option<&str>) -> Value {
    let mut metrics = result.clone();
    let mut records = serde_json::Map::new();
    for key in [
        "input_lens",
        "output_lens",
        "start_times",
        "ttfts",
        "itls",
        "generated_texts",
        "errors",
        "server_queue_times",
        "server_ttfts",
    ] {
        if let Some(value) = metrics.as_object_mut().unwrap().remove(key) {
            records.insert(key.into(), value);
        }
    }
    let mut settings = json!({
        "model_id": result["model_id"], "tokenizer_id": result["tokenizer_id"],
        "workload": config.workload.map(|_| metadata(config)),
        "client_args": config.report_settings
    });
    settings.sort_all_objects();
    // Match Python's sorted compact JSON with ensure_ascii=True.
    let canonical: String = settings
        .to_string()
        .chars()
        .map(|character| {
            if character.is_ascii() {
                character.to_string()
            } else {
                character
                    .encode_utf16(&mut [0; 2])
                    .iter()
                    .map(|unit| format!("\\u{unit:04x}"))
                    .collect()
            }
        })
        .collect();
    let hash = format!("{:x}", Sha256::digest(canonical.as_bytes()));
    let complete =
        cache_error.is_none() && result["completed"].as_u64() == Some(config.num_prompts as u64);
    json!({
        "schema": "byog.measurement/v1", "kind": "benchmark", "name": "serve",
        "created_at": result["date"], "status": if complete {"completed"} else {"partial"},
        "client": {"vllm_version": vllm_build_info::VERSION, "implementation": "rust"},
        "server": null, "settings": settings, "settings_sha256": hash,
        "metrics": metrics, "records": records
    })
}
