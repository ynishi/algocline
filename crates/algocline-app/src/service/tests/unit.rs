use std::path::{Path, PathBuf};
use std::sync::Arc;

use algocline_core::ExecutionObserver;
use std::io::Write;

use crate::service::config::{AppConfig, LogDirSource};
use crate::service::path::{copy_dir, ContainedPath};
use crate::service::resolve::{make_require_code, packages_dir, resolve_code};
use crate::service::transcript::{append_note, write_transcript_log};
use crate::service::AppService;

// ─── resolve_code tests ───

#[test]
fn resolve_code_inline() {
    let result = resolve_code(Some("return 1".into()), None);
    assert_eq!(result.unwrap(), "return 1");
}

#[test]
fn resolve_code_from_file() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    write!(tmp, "return 42").unwrap();

    let result = resolve_code(None, Some(tmp.path().to_string_lossy().into()));
    assert_eq!(result.unwrap(), "return 42");
}

#[test]
fn resolve_code_both_provided_error() {
    let result = resolve_code(Some("code".into()), Some("file.lua".into()));
    let err = result.unwrap_err();
    assert!(err.contains("not both"), "error: {err}");
}

#[test]
fn resolve_code_neither_provided_error() {
    let result = resolve_code(None, None);
    let err = result.unwrap_err();
    assert!(err.contains("must be provided"), "error: {err}");
}

#[test]
fn resolve_code_nonexistent_file_error() {
    let result = resolve_code(
        None,
        Some("/tmp/algocline_nonexistent_test_file.lua".into()),
    );
    assert!(result.is_err());
}

// ─── make_require_code tests ───

#[test]
fn make_require_code_basic() {
    let code = make_require_code("ucb");
    assert!(code.contains(r#"require("ucb")"#), "code: {code}");
    assert!(code.contains("pkg.run(ctx)"), "code: {code}");
}

#[test]
fn make_require_code_different_names() {
    for name in &["panel", "cot", "sc", "cove", "reflect", "calibrate"] {
        let code = make_require_code(name);
        assert!(
            code.contains(&format!(r#"require("{name}")"#)),
            "code for {name}: {code}"
        );
    }
}

// ─── packages_dir tests ───

#[test]
fn packages_dir_ends_with_expected_path() {
    let tmp = tempfile::tempdir().unwrap();
    let app_dir = algocline_core::AppDir::new(tmp.path().to_path_buf());
    let dir = packages_dir(&app_dir);
    assert!(dir.ends_with("packages"), "dir: {}", dir.display());
}

// ─── append_note tests ───

#[test]
fn append_note_to_existing_log() {
    let dir = tempfile::tempdir().unwrap();
    let session_id = "s-test-001";
    let log = serde_json::json!({
        "session_id": session_id,
        "stats": { "elapsed_ms": 100 },
        "transcript": [],
    });
    let path = dir.path().join(format!("{session_id}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&log).unwrap()).unwrap();

    let count = append_note(dir.path(), session_id, "Step 2 was weak", Some("Step 2")).unwrap();
    assert_eq!(count, 1);

    let count = append_note(dir.path(), session_id, "Overall good", None).unwrap();
    assert_eq!(count, 2);

    let raw = std::fs::read_to_string(&path).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let notes = doc["notes"].as_array().unwrap();
    assert_eq!(notes.len(), 2);
    assert_eq!(notes[0]["content"], "Step 2 was weak");
    assert_eq!(notes[0]["title"], "Step 2");
    assert_eq!(notes[1]["content"], "Overall good");
    assert!(notes[1]["title"].is_null());
    assert!(notes[0]["timestamp"].is_number());
}

#[test]
fn append_note_missing_log_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let result = append_note(dir.path(), "s-nonexistent", "note", None);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

// ─── log_list / log_view tests ───

#[test]
fn log_list_from_dir() {
    let dir = tempfile::tempdir().unwrap();

    // Create two log files
    let log1 = serde_json::json!({
        "session_id": "s-001",
        "task_hint": "What is 2+2?",
        "stats": { "elapsed_ms": 100, "rounds": 1, "llm_calls": 1 },
        "transcript": [{ "prompt": "What is 2+2?", "response": "4" }],
    });
    let log2 = serde_json::json!({
        "session_id": "s-002",
        "task_hint": "Explain ownership",
        "stats": { "elapsed_ms": 5000, "rounds": 3, "llm_calls": 3 },
        "transcript": [],
        "notes": [{ "timestamp": 0, "content": "good" }],
    });

    std::fs::write(
        dir.path().join("s-001.json"),
        serde_json::to_string(&log1).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("s-002.json"),
        serde_json::to_string(&log2).unwrap(),
    )
    .unwrap();
    // Non-json file should be ignored
    std::fs::write(dir.path().join("README.txt"), "ignore me").unwrap();

    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };

    // Use log_list directly via the free function path
    let entries = std::fs::read_dir(config.log_dir.as_ref().unwrap()).unwrap();
    let mut count = 0;
    for entry in entries.flatten() {
        if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
            count += 1;
        }
    }
    assert_eq!(count, 2);
}

// ─── ContainedPath tests ───

#[test]
fn contained_path_accepts_simple_name() {
    let dir = tempfile::tempdir().unwrap();
    let result = ContainedPath::child(dir.path(), "s-abc123.json");
    assert!(result.is_ok());
    assert!(result.unwrap().as_ref().ends_with("s-abc123.json"));
}

#[test]
fn contained_path_rejects_parent_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let result = ContainedPath::child(dir.path(), "../../../etc/passwd");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("path traversal"), "err: {err}");
}

#[test]
fn contained_path_rejects_absolute_path() {
    let dir = tempfile::tempdir().unwrap();
    let result = ContainedPath::child(dir.path(), "/etc/passwd");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("path traversal"), "err: {err}");
}

#[test]
fn contained_path_rejects_dot_dot_in_middle() {
    let dir = tempfile::tempdir().unwrap();
    let result = ContainedPath::child(dir.path(), "foo/../bar");
    assert!(result.is_err());
}

#[test]
fn contained_path_accepts_nested_normal() {
    let dir = tempfile::tempdir().unwrap();
    let result = ContainedPath::child(dir.path(), "sub/file.json");
    assert!(result.is_ok());
}

#[test]
fn append_note_rejects_traversal_session_id() {
    let dir = tempfile::tempdir().unwrap();
    let result = append_note(dir.path(), "../../../etc/passwd", "evil", None);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("path traversal"));
}

// ─── meta file tests ───

#[test]
fn write_transcript_log_creates_meta_file() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };

    let metrics = algocline_core::ExecutionMetrics::new();
    let observer = metrics.create_observer();
    observer.on_paused(&[algocline_core::LlmQuery {
        id: algocline_core::QueryId::single(),
        prompt: "What is 2+2?".into(),
        system: None,
        max_tokens: 100,
        grounded: false,
        underspecified: false,
    }]);
    observer.on_response_fed(&algocline_core::QueryId::single(), "4", None);
    observer.on_resumed();
    observer.on_completed(&serde_json::json!(null));

    write_transcript_log(&config, "s-meta-test", &metrics, Some("ucb"))
        .expect("write_transcript_log");

    // Main log should exist
    assert!(dir.path().join("s-meta-test.json").exists());

    // Meta file should exist
    let meta_path = dir.path().join("s-meta-test.meta.json");
    assert!(meta_path.exists());

    let raw = std::fs::read_to_string(&meta_path).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(meta["session_id"], "s-meta-test");
    assert_eq!(meta["notes_count"], 0);
    assert!(meta.get("elapsed_ms").is_some());
    assert!(meta.get("rounds").is_some());
    assert!(meta.get("llm_calls").is_some());
    assert_eq!(meta["strategy"], "ucb");
    assert!(meta.get("total_prompt_chars").is_some());
    assert!(meta.get("total_response_chars").is_some());
    // Meta should NOT contain transcript
    assert!(meta.get("transcript").is_none());

    // Full log should also contain strategy
    let log_raw = std::fs::read_to_string(dir.path().join("s-meta-test.json")).unwrap();
    let log: serde_json::Value = serde_json::from_str(&log_raw).unwrap();
    assert_eq!(log["strategy"], "ucb");
}

#[test]
fn write_transcript_log_strategy_none() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };

    let metrics = algocline_core::ExecutionMetrics::new();
    let observer = metrics.create_observer();
    observer.on_paused(&[algocline_core::LlmQuery {
        id: algocline_core::QueryId::single(),
        prompt: "hello".into(),
        system: None,
        max_tokens: 100,
        grounded: false,
        underspecified: false,
    }]);
    observer.on_response_fed(&algocline_core::QueryId::single(), "world", None);
    observer.on_resumed();
    observer.on_completed(&serde_json::json!(null));

    write_transcript_log(&config, "s-no-strat", &metrics, None).expect("write_transcript_log");

    let meta_path = dir.path().join("s-no-strat.meta.json");
    let raw = std::fs::read_to_string(&meta_path).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert!(meta["strategy"].is_null());
}

#[test]
fn append_note_updates_meta_notes_count() {
    let dir = tempfile::tempdir().unwrap();
    let session_id = "s-meta-note";

    // Create main log
    let log = serde_json::json!({
        "session_id": session_id,
        "stats": { "elapsed_ms": 100 },
        "transcript": [],
    });
    std::fs::write(
        dir.path().join(format!("{session_id}.json")),
        serde_json::to_string_pretty(&log).unwrap(),
    )
    .unwrap();

    // Create meta file
    let meta = serde_json::json!({
        "session_id": session_id,
        "task_hint": "test",
        "elapsed_ms": 100,
        "rounds": 1,
        "llm_calls": 1,
        "notes_count": 0,
    });
    std::fs::write(
        dir.path().join(format!("{session_id}.meta.json")),
        serde_json::to_string(&meta).unwrap(),
    )
    .unwrap();

    append_note(dir.path(), session_id, "first note", None).unwrap();

    let raw = std::fs::read_to_string(dir.path().join(format!("{session_id}.meta.json"))).unwrap();
    let updated: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(updated["notes_count"], 1);

    append_note(dir.path(), session_id, "second note", None).unwrap();

    let raw = std::fs::read_to_string(dir.path().join(format!("{session_id}.meta.json"))).unwrap();
    let updated: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(updated["notes_count"], 2);
}

// ─── TranscriptConfig tests ───

#[test]
fn transcript_config_default_enabled() {
    // Without env vars, should default to enabled
    let config = AppConfig {
        log_dir: Some(PathBuf::from("/tmp/test")),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    assert!(config.log_enabled);
}

#[test]
fn write_transcript_log_disabled_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: false,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    let metrics = algocline_core::ExecutionMetrics::new();
    let observer = metrics.create_observer();
    observer.on_paused(&[algocline_core::LlmQuery {
        id: algocline_core::QueryId::single(),
        prompt: "test".into(),
        system: None,
        max_tokens: 10,
        grounded: false,
        underspecified: false,
    }]);
    observer.on_response_fed(&algocline_core::QueryId::single(), "r", None);
    observer.on_resumed();
    observer.on_completed(&serde_json::json!(null));

    write_transcript_log(&config, "s-disabled", &metrics, None).expect("write_transcript_log");

    // No file should be created
    assert!(!dir.path().join("s-disabled.json").exists());
    assert!(!dir.path().join("s-disabled.meta.json").exists());
}

#[test]
fn write_transcript_log_empty_transcript_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // Metrics with no observer events → empty transcript
    let metrics = algocline_core::ExecutionMetrics::new();
    write_transcript_log(&config, "s-empty", &metrics, None).expect("write_transcript_log");
    assert!(!dir.path().join("s-empty.json").exists());
}

// ─── copy_dir tests ───

#[test]
fn copy_dir_basic() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();

    std::fs::write(src.path().join("a.txt"), "hello").unwrap();
    std::fs::create_dir(src.path().join("sub")).unwrap();
    std::fs::write(src.path().join("sub/b.txt"), "world").unwrap();

    let dst_path = dst.path().join("copied");
    copy_dir(src.path(), &dst_path).unwrap();

    assert_eq!(
        std::fs::read_to_string(dst_path.join("a.txt")).unwrap(),
        "hello"
    );
    assert_eq!(
        std::fs::read_to_string(dst_path.join("sub/b.txt")).unwrap(),
        "world"
    );
}

#[test]
fn copy_dir_empty() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let dst_path = dst.path().join("empty_copy");
    copy_dir(src.path(), &dst_path).unwrap();
    assert!(dst_path.exists());
    assert!(dst_path.is_dir());
}

// ─── task_hint truncation in write_transcript_log ───

#[test]
fn write_transcript_log_truncates_long_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    let metrics = algocline_core::ExecutionMetrics::new();
    let observer = metrics.create_observer();
    let long_prompt = "x".repeat(300);
    observer.on_paused(&[algocline_core::LlmQuery {
        id: algocline_core::QueryId::single(),
        prompt: long_prompt,
        system: None,
        max_tokens: 10,
        grounded: false,
        underspecified: false,
    }]);
    observer.on_response_fed(&algocline_core::QueryId::single(), "r", None);
    observer.on_resumed();
    observer.on_completed(&serde_json::json!(null));

    write_transcript_log(&config, "s-long", &metrics, None).expect("write_transcript_log");

    let raw = std::fs::read_to_string(dir.path().join("s-long.json")).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let hint = doc["task_hint"].as_str().unwrap();
    // Should be truncated to ~100 chars + "..."
    assert!(hint.len() <= 104, "hint too long: {} chars", hint.len());
    assert!(hint.ends_with("..."));
}

#[test]
fn log_list_prefers_meta_file() {
    let dir = tempfile::tempdir().unwrap();

    // Create a full log (large, with transcript)
    let log = serde_json::json!({
        "session_id": "s-big",
        "task_hint": "full log hint",
        "stats": { "elapsed_ms": 999, "rounds": 5, "llm_calls": 5 },
        "transcript": [{"prompt": "x".repeat(10000), "response": "y".repeat(10000)}],
    });
    std::fs::write(
        dir.path().join("s-big.json"),
        serde_json::to_string(&log).unwrap(),
    )
    .unwrap();

    // Create corresponding meta
    let meta = serde_json::json!({
        "session_id": "s-big",
        "task_hint": "full log hint",
        "elapsed_ms": 999,
        "rounds": 5,
        "llm_calls": 5,
        "notes_count": 0,
    });
    std::fs::write(
        dir.path().join("s-big.meta.json"),
        serde_json::to_string(&meta).unwrap(),
    )
    .unwrap();

    // Create a legacy log (no meta file)
    let legacy = serde_json::json!({
        "session_id": "s-legacy",
        "task_hint": "legacy hint",
        "stats": { "elapsed_ms": 100, "rounds": 1, "llm_calls": 1 },
        "transcript": [],
    });
    std::fs::write(
        dir.path().join("s-legacy.json"),
        serde_json::to_string(&legacy).unwrap(),
    )
    .unwrap();

    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    let result = app.log_list(50).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let sessions = parsed["sessions"].as_array().unwrap();

    assert_eq!(sessions.len(), 2);

    // Both sessions should have session_id and task_hint
    let ids: Vec<&str> = sessions
        .iter()
        .map(|s| s["session_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"s-big"));
    assert!(ids.contains(&"s-legacy"));
}

// ─── stats tests ───

#[test]
fn stats_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    let result = app.stats(None, None).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["total_sessions"], 0);
}

#[test]
fn stats_aggregates_by_strategy() {
    let dir = tempfile::tempdir().unwrap();

    // Create meta files for different strategies
    let meta1 = serde_json::json!({
        "session_id": "s-001", "strategy": "ucb",
        "elapsed_ms": 1000, "llm_calls": 10, "rounds": 5,
        "total_prompt_chars": 500, "total_response_chars": 300,
    });
    let meta2 = serde_json::json!({
        "session_id": "s-002", "strategy": "ucb",
        "elapsed_ms": 2000, "llm_calls": 12, "rounds": 6,
        "total_prompt_chars": 600, "total_response_chars": 400,
    });
    let meta3 = serde_json::json!({
        "session_id": "s-003", "strategy": "cove",
        "elapsed_ms": 500, "llm_calls": 4, "rounds": 2,
        "total_prompt_chars": 200, "total_response_chars": 150,
    });

    for (name, meta) in [("s-001", &meta1), ("s-002", &meta2), ("s-003", &meta3)] {
        std::fs::write(
            dir.path().join(format!("{name}.meta.json")),
            serde_json::to_string(meta).unwrap(),
        )
        .unwrap();
    }

    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    // All strategies
    let result = app.stats(None, None).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["total_sessions"], 3);
    assert_eq!(parsed["strategies"]["ucb"]["count"], 2);
    assert_eq!(parsed["strategies"]["ucb"]["avg_elapsed_ms"], 1500);
    assert_eq!(parsed["strategies"]["ucb"]["avg_llm_calls"], 11);
    assert_eq!(parsed["strategies"]["cove"]["count"], 1);

    // Filter by strategy
    let result = app.stats(Some("ucb"), None).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["total_sessions"], 2);
    assert!(parsed["strategies"]["cove"].is_null());

    // Filter by nonexistent strategy
    let result = app.stats(Some("nonexistent"), None).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["total_sessions"], 0);
}

#[test]
fn stats_legacy_logs_without_strategy() {
    let dir = tempfile::tempdir().unwrap();

    // Legacy log without strategy field (pre-0.6.0)
    let legacy = serde_json::json!({
        "session_id": "s-legacy",
        "stats": { "elapsed_ms": 300, "llm_calls": 2, "rounds": 1,
                    "total_prompt_chars": 100, "total_response_chars": 50 },
        "transcript": [],
    });
    std::fs::write(
        dir.path().join("s-legacy.json"),
        serde_json::to_string(&legacy).unwrap(),
    )
    .unwrap();

    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    let result = app.stats(None, None).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["total_sessions"], 1);
    assert_eq!(parsed["strategies"]["unknown"]["count"], 1);
}

// ─── info / require_log_dir / LogDirSource tests ───

#[test]
fn info_returns_valid_json_with_expected_keys() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppConfig {
        log_dir: Some(dir.path().to_path_buf()),
        log_dir_source: LogDirSource::Home,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    let result = app.info();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert!(parsed["version"].is_string());
    assert!(parsed["log_dir"]["resolved"].is_string());
    assert_eq!(
        parsed["log_dir"]["source"].as_str().unwrap(),
        "~/.algocline/logs"
    );
    assert!(parsed["log_enabled"].as_bool().unwrap());
    assert_eq!(parsed["tracing"].as_str().unwrap(), "file + stderr");
}

#[test]
fn info_stderr_only_when_no_log_dir() {
    let config = AppConfig {
        log_dir: None,
        log_dir_source: LogDirSource::None,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    let result = app.info();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert!(parsed["log_dir"]["resolved"].is_null());
    assert_eq!(
        parsed["log_dir"]["source"].as_str().unwrap(),
        "none (stderr only)"
    );
    assert_eq!(parsed["tracing"].as_str().unwrap(), "stderr only");
}

#[test]
fn require_log_dir_returns_path_when_present() {
    let config = AppConfig {
        log_dir: Some(PathBuf::from("/tmp/test-logs")),
        log_dir_source: LogDirSource::EnvVar,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    assert_eq!(app.require_log_dir().unwrap(), Path::new("/tmp/test-logs"));
}

#[test]
fn require_log_dir_returns_err_when_none() {
    let config = AppConfig {
        log_dir: None,
        log_dir_source: LogDirSource::None,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    assert!(app.require_log_dir().is_err());
}

#[test]
fn write_transcript_log_noop_when_log_dir_none() {
    let dir = tempfile::tempdir().unwrap();
    let config = AppConfig {
        log_dir: None,
        log_dir_source: LogDirSource::None,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    let metrics = algocline_core::ExecutionMetrics::new();
    let observer = metrics.create_observer();
    observer.on_paused(&[algocline_core::LlmQuery {
        id: algocline_core::QueryId::single(),
        prompt: "test".into(),
        system: None,
        max_tokens: 10,
        grounded: false,
        underspecified: false,
    }]);
    observer.on_response_fed(&algocline_core::QueryId::single(), "r", None);
    observer.on_resumed();
    observer.on_completed(&serde_json::json!(null));

    write_transcript_log(&config, "s-none-dir", &metrics, None).expect("write_transcript_log");

    // No file anywhere — dir is unused, just verifying no panic
    assert!(!dir.path().join("s-none-dir.json").exists());
}

#[test]
fn log_dir_source_display_all_variants() {
    assert_eq!(LogDirSource::EnvVar.to_string(), "ALC_LOG_DIR");
    assert_eq!(LogDirSource::Home.to_string(), "~/.algocline/logs");
    assert_eq!(LogDirSource::StateDir.to_string(), "state_dir");
    assert_eq!(LogDirSource::CurrentDir.to_string(), "current_dir");
    assert_eq!(LogDirSource::None.to_string(), "none (stderr only)");
}

#[test]
fn log_list_returns_empty_when_no_log_dir() {
    let config = AppConfig {
        log_dir: None,
        log_dir_source: LogDirSource::None,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    let result = app.log_list(50).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["sessions"].as_array().unwrap().len(), 0);
}

#[test]
fn stats_returns_zero_when_no_log_dir() {
    let config = AppConfig {
        log_dir: None,
        log_dir_source: LogDirSource::None,
        log_enabled: true,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    // NOTE: Build `AppService` as a struct literal (not `AppService::new`)
    // so the sync `#[test]` path does not require a Tokio runtime context —
    // `AppService::new` spawns a GC task via `tokio::spawn`, which panics
    // when no reactor is running.  Subtask 2c swaps the relative
    // `.algocline/` paths for a tempdir.
    let app = AppService {
        executor: Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async { algocline_engine::Executor::new(vec![]).await.unwrap() }),
        ),
        registry: Arc::new(algocline_engine::SessionRegistry::new()),
        log_config: config,
        state_store: Arc::new(algocline_engine::JsonFileStore::new(
            std::path::PathBuf::from("."),
        )),
        card_store: Arc::new(algocline_engine::FileCardStore::new(
            std::path::PathBuf::from("."),
        )),
        eval_sessions: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        session_strategies: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        search_paths: vec![],
    };

    let result = app.stats(None, None).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed["total_sessions"].as_u64().unwrap(), 0);
}

// ─── status(pending_filter) tests ───
//
// These cover the `resolve_pending_filter` dispatch branches at the app
// layer. With an empty registry the status response is `{active_sessions:
// 0, sessions: []}` regardless of filter shape, so a filter error surfaces
// on the Err path even without any Paused session in place.

async fn make_status_test_app() -> AppService {
    let executor = Arc::new(algocline_engine::Executor::new(vec![]).await.unwrap());
    let log_config = AppConfig {
        log_dir: None,
        log_dir_source: LogDirSource::None,
        log_enabled: false,
        prompt_preview_chars: algocline_engine::DEFAULT_PROMPT_PREVIEW_CHARS,
        ..Default::default()
    };
    AppService::new(executor, log_config, vec![])
}

#[tokio::test]
async fn status_without_filter_returns_empty_active_sessions() {
    let app = make_status_test_app().await;
    let out = app.status(None, None).await.unwrap();
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["active_sessions"], 0);
    assert_eq!(v["sessions"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn status_with_known_preset_string_ok() {
    let app = make_status_test_app().await;
    for preset in ["meta", "preview", "full"] {
        let out = app
            .status(None, Some(serde_json::json!(preset)))
            .await
            .unwrap_or_else(|e| panic!("preset '{preset}' should resolve, got err: {e}"));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["active_sessions"], 0);
    }
}

#[tokio::test]
async fn status_with_unknown_preset_returns_error() {
    let app = make_status_test_app().await;
    let err = app
        .status(None, Some(serde_json::json!("bogus")))
        .await
        .expect_err("unknown preset must error");
    assert!(
        err.contains("unknown pending_filter preset"),
        "err should explain the preset, got: {err}"
    );
    assert!(
        err.contains("bogus"),
        "err should echo the bad name, got: {err}"
    );
}

#[tokio::test]
async fn status_with_custom_object_filter_ok() {
    let app = make_status_test_app().await;
    let filter = serde_json::json!({
        "query_id": true,
        "prompt": { "mode": "preview", "chars": 80 }
    });
    let out = app.status(None, Some(filter)).await.unwrap();
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["active_sessions"], 0);
}

#[tokio::test]
async fn status_with_non_object_non_string_filter_errors() {
    let app = make_status_test_app().await;
    for (label, bad) in [
        ("null", serde_json::json!(null)),
        ("bool", serde_json::json!(true)),
        ("number", serde_json::json!(42)),
        ("array", serde_json::json!(["meta"])),
    ] {
        let result = app.status(None, Some(bad)).await;
        let err = result.expect_err(&format!("{label} filter must error"));
        assert!(
            err.contains("pending_filter must be a preset name"),
            "err for {label} should explain shape, got: {err}"
        );
        assert!(
            err.contains(label),
            "err for {label} should name the bad type, got: {err}"
        );
    }
}
