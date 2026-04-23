//! Slow smoke: embedded `hub_gendoc` vs `lua tools/gen_docs.lua` on full
//! `algocline-bundled-packages`. Run via **`just hub-gendoc-smoke`** (or
//! `ALG_BUNDLED_PACKAGES=… cargo test -p algocline-app … -- --ignored`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::service::config::{AppConfig, LogDirSource};
use crate::service::AppService;

fn bundled_packages_root() -> PathBuf {
    std::env::var_os("ALG_BUNDLED_PACKAGES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            // `CARGO_MANIFEST_DIR` = `.../algocline/crates/algocline-app`
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../algocline-bundled-packages")
        })
}

#[test]
#[ignore = "slow; run `just hub-gendoc-smoke` until promoted to CI"]
fn hub_gendoc_matches_standalone_lua_on_bundled() {
    let bundled = bundled_packages_root();
    if !bundled.join("hub_index.json").is_file() || !bundled.join("alc_shapes/init.lua").is_file() {
        eprintln!(
            "skip: no hub_index.json / alc_shapes under {}",
            bundled.display()
        );
        return;
    }
    let root = bundled.to_str().expect("utf-8 bundled path");

    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path();
    let config = AppConfig {
        log_dir: Some(dir.to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(PathBuf::from("."))),
        card_store: Arc::new(algocline_engine::FileCardStore::new(PathBuf::from("."))),
        eval_sessions: Arc::new(std::sync::Mutex::new(HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(HashMap::new())),
        search_paths: vec![],
    };

    let rust_out = dir.join("rust_docs");
    std::fs::create_dir_all(&rust_out).expect("mkdir rust_docs");
    let rust_out_s = rust_out.to_str().unwrap();
    let gendoc_json = app
        .hub_gendoc(root, Some(rust_out_s), None, None, None)
        .unwrap_or_else(|e| panic!("hub_gendoc: {e}"));
    assert!(
        gendoc_json.contains("source_dir"),
        "expected JSON response body, got: {gendoc_json}"
    );

    let lua_out = dir.join("lua_docs");
    std::fs::create_dir_all(&lua_out).expect("mkdir lua_docs");
    let lua_st = std::process::Command::new("lua")
        .current_dir(&bundled)
        .args(["tools/gen_docs.lua", root, lua_out.to_str().unwrap()])
        .status()
        .expect("spawn lua tools/gen_docs.lua");
    assert!(lua_st.success(), "lua gen_docs failed: {lua_st}");

    let rust_llms = std::fs::read_to_string(rust_out.join("llms-full.txt")).expect("rust llms");
    let lua_llms = std::fs::read_to_string(lua_out.join("llms-full.txt")).expect("lua llms");
    let rl = rust_llms.lines().count();
    let ll = lua_llms.lines().count();
    let rr = rust_llms.matches("## Result").count();
    let lr = lua_llms.matches("## Result").count();
    eprintln!("embedded hub_gendoc: llms-full lines={rl}  ## Result={rr}");
    eprintln!("standalone lua:       llms-full lines={ll}  ## Result={lr}");
    assert_eq!(rr, lr, "## Result count: embedded vs standalone lua");
    let d = (rl as isize - ll as isize).abs();
    assert!(
        d <= 5,
        "llms-full line count drift embedded={rl} lua={ll} (Δ={d})"
    );
}
