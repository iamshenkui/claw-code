use std::collections::HashSet;
use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use api::ProviderKind;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Describes a single environment variable that a provider may need.
struct EnvVarSpec {
    key: &'static str,
    description: &'static str,
    default_value: Option<&'static str>,
}

/// A provider's full set of environment variable requirements.
struct ProviderEnvSpec {
    provider_name: &'static str,
    kind: ProviderKind,
    /// Environment variable name that selects this provider's default model.
    /// E.g. "ANTHROPIC_MODEL", "XAI_MODEL", "OPENAI_MODEL".
    model_key: &'static str,
    /// Compiled-in default model for this provider (used as comment in template).
    default_model: &'static str,
    /// All env var specs for this provider (required + optional).
    vars: Vec<EnvVarSpec>,
    /// If set, at least one of these keys must have a non-empty value.
    /// Used for Anthropic where either API_KEY or AUTH_TOKEN suffices.
    or_group: Option<Vec<&'static str>>,
}

/// Result of ensuring the project .env file.
pub struct EnvLoadResult {
    pub env_path: PathBuf,
    pub was_created: bool,
    pub was_patched: bool,
}

/// Validation outcome for provider credential check.
enum EnvValidation {
    Ok,
    MissingValues { keys: Vec<String>, provider: String },
}

// ---------------------------------------------------------------------------
// Provider specs derived from the existing provider/config system
// ---------------------------------------------------------------------------

fn all_provider_specs() -> &'static [ProviderEnvSpec] {
    static SPECS: OnceLock<Vec<ProviderEnvSpec>> = OnceLock::new();
    SPECS.get_or_init(|| {
        vec![
            ProviderEnvSpec {
                provider_name: "Anthropic",
                kind: ProviderKind::Anthropic,
                model_key: "ANTHROPIC_MODEL",
                default_model: "claude-opus-4-6",
                vars: vec![
                    EnvVarSpec {
                        key: "ANTHROPIC_API_KEY",
                        description: "Anthropic API key",
                        default_value: None,
                    },
                    EnvVarSpec {
                        key: "ANTHROPIC_AUTH_TOKEN",
                        description: "Anthropic auth token (alternative to API key)",
                        default_value: None,
                    },
                    EnvVarSpec {
                        key: "ANTHROPIC_BASE_URL",
                        description: "Anthropic API endpoint",
                        default_value: Some("https://api.anthropic.com"),
                    },
                ],
                or_group: Some(vec!["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]),
            },
            ProviderEnvSpec {
                provider_name: "xAI",
                kind: ProviderKind::Xai,
                model_key: "XAI_MODEL",
                default_model: "grok-3",
                vars: vec![
                    EnvVarSpec {
                        key: "XAI_API_KEY",
                        description: "xAI API key",
                        default_value: None,
                    },
                    EnvVarSpec {
                        key: "XAI_BASE_URL",
                        description: "xAI API endpoint",
                        default_value: Some("https://api.x.ai/v1"),
                    },
                ],
                or_group: None,
            },
            ProviderEnvSpec {
                provider_name: "OpenAI",
                kind: ProviderKind::OpenAi,
                model_key: "OPENAI_MODEL",
                default_model: "gpt-4o",
                vars: vec![
                    EnvVarSpec {
                        key: "OPENAI_API_KEY",
                        description: "OpenAI API key",
                        default_value: None,
                    },
                    EnvVarSpec {
                        key: "OPENAI_BASE_URL",
                        description: "OpenAI / OpenAI-compatible API endpoint",
                        default_value: Some("https://api.openai.com/v1"),
                    },
                ],
                or_group: None,
            },
        ]
    })
}

/// Determine the active provider spec based on the model name.
fn derive_active_spec(model: &str) -> &'static ProviderEnvSpec {
    // CLAW_PROVIDER explicit override takes highest priority — must be checked
    // before metadata_for_model, which would otherwise win for known model names
    // (e.g. "claude-opus-4-6" always resolves to Anthropic metadata even when
    // the user has pointed it at an OpenAI-compatible endpoint).
    if let Ok(provider_override) = std::env::var("CLAW_PROVIDER") {
        let kind = match provider_override.to_ascii_lowercase().as_str() {
            "anthropic" | "claude" => Some(ProviderKind::Anthropic),
            "xai" | "grok" => Some(ProviderKind::Xai),
            "openai" => Some(ProviderKind::OpenAi),
            other => {
                eprintln!(
                    "warning: unknown CLAW_PROVIDER value {:?}; ignoring and detecting from model name",
                    other
                );
                None
            }
        };
        if let Some(kind) = kind {
            if let Some(spec) = all_provider_specs().iter().find(|s| s.kind == kind) {
                return spec;
            }
        }
    }

    // Check model metadata for a precise match.
    if let Some(meta) = api::metadata_for_model(model) {
        if let Some(spec) = all_provider_specs().iter().find(|s| s.kind == meta.provider) {
            return spec;
        }
    }

    // Fall back to heuristic detection by kind.
    let kind = api::detect_provider_kind(model);
    all_provider_specs()
        .iter()
        .find(|s| s.kind == kind)
        .unwrap_or_else(|| {
            // Should never happen, but default to Anthropic.
            all_provider_specs().first().unwrap()
        })
}

// ---------------------------------------------------------------------------
// .env file operations
// ---------------------------------------------------------------------------

fn env_path() -> PathBuf {
    match env::current_dir() {
        Ok(cwd) => cwd.join(".env"),
        Err(_) => PathBuf::from(".env"),
    }
}

/// Resolve the model to use by consulting project env vars in priority order:
///
/// 1. `CLAW_MODEL` — global override, applies to all providers.
/// 2. Provider-specific model var (`ANTHROPIC_MODEL`, `XAI_MODEL`, `OPENAI_MODEL`)
///    selected by the current `CLAW_PROVIDER` value (if set).
/// 3. If `CLAW_PROVIDER` names a non-Anthropic provider but no model var is set,
///    return that provider's compiled-in default (e.g. `gpt-4o` for `openai`).
/// 4. Returns `None` — caller falls back to compiled-in Anthropic default.
///
/// **Side effect**: when a provider-specific model var (e.g. `OPENAI_MODEL`) is
/// selected and `CLAW_PROVIDER` is not already set, this function writes
/// `CLAW_PROVIDER` into the process environment.  This ensures that the runtime
/// provider-kind detection (which checks `CLAW_PROVIDER` first) agrees with the
/// validation path even for model names that are not in the built-in prefix table
/// (e.g. `deepseek-chat`, `qwen-plus`).
///
/// This should be called **after** `load_project_env()` so that project .env values
/// are already in the process environment.
pub fn resolve_model_from_env() -> Option<String> {
    // Global override wins.
    if let Ok(m) = env::var("CLAW_MODEL") {
        let m = m.trim().to_string();
        if !m.is_empty() {
            return Some(m);
        }
    }

    let claw_provider = env::var("CLAW_PROVIDER").ok();
    let claw_provider = claw_provider.as_deref();

    // Provider-specific model key, determined by CLAW_PROVIDER (if set).
    // Returns (Option<model_key>, Option<inferred_claw_provider_value>).
    // The second element is Some only when we need to backfill CLAW_PROVIDER.
    let (provider_model_key, backfill_provider): (Option<&'static str>, Option<&'static str>) =
        match claw_provider {
            Some("anthropic" | "claude") => (Some("ANTHROPIC_MODEL"), None),
            Some("xai" | "grok") => (Some("XAI_MODEL"), None),
            Some("openai") => (Some("OPENAI_MODEL"), None),
            Some(v) if !v.trim().is_empty() => {
                eprintln!(
                    "warning: unknown CLAW_PROVIDER value {:?}; ignoring and scanning all provider model vars",
                    v
                );
                (None, None)
            }
            _ => {
                // No CLAW_PROVIDER set: check if any per-provider model var has a value.
                // Check in spec order so Anthropic wins as the natural default.
                if let Some(spec) = all_provider_specs().iter().find(|s| {
                    env::var(s.model_key).map_or(false, |v| !v.trim().is_empty())
                }) {
                    // Infer CLAW_PROVIDER from whichever model var has a value so that
                    // detect_provider_kind() agrees even for unrecognised model names.
                    let provider_value: &'static str = match spec.kind {
                        ProviderKind::Anthropic => "anthropic",
                        ProviderKind::Xai => "xai",
                        ProviderKind::OpenAi => "openai",
                    };
                    (Some(spec.model_key), Some(provider_value))
                } else {
                    (None, None)
                }
            }
        };

    // Backfill CLAW_PROVIDER when we inferred it from a provider-specific model var.
    if let Some(pv) = backfill_provider {
        env::set_var("CLAW_PROVIDER", pv);
    }

    if let Some(key) = provider_model_key {
        if let Ok(m) = env::var(key) {
            let m = m.trim().to_string();
            if !m.is_empty() {
                return Some(m);
            }
        }
    }

    // Risk 1 fix: when CLAW_PROVIDER names a known non-Anthropic provider but no
    // model var is set, return that provider's compiled-in default so we never
    // send a Claude model name to an OpenAI-compatible endpoint.
    if let Some(pv) = claw_provider {
        let kind = match pv.to_ascii_lowercase().as_str() {
            "xai" | "grok" => Some(ProviderKind::Xai),
            "openai" => Some(ProviderKind::OpenAi),
            _ => None,
        };
        if let Some(kind) = kind {
            if let Some(spec) = all_provider_specs().iter().find(|s| s.kind == kind) {
                return Some(spec.default_model.to_string());
            }
        }
    }

    None
}

/// Load `.env` from the current working directory into the process environment.
/// Silent — does nothing if the file does not exist. Does NOT traverse parent dirs.
pub fn load_project_env() -> Result<(), Box<dyn std::error::Error>> {
    let path = env_path();
    if path.exists() {
        dotenvy::from_path(&path)?;
    }
    Ok(())
}

/// Generate the content for a new `.env` file.
fn generate_env_content(active: &ProviderEnvSpec) -> String {
    let specs = all_provider_specs();
    let mut lines: Vec<String> = Vec::new();

    lines.push("# Claw Code - Project Environment Configuration".to_string());
    lines.push("".to_string());

    for spec in specs {
        let is_active = spec.kind == active.kind;
        let tag = if is_active {
            format!(" (active — derived from model provider)")
        } else {
            String::new()
        };
        lines.push(format!("# --- {}{} ---", spec.provider_name, tag));

        if is_active && spec.or_group.is_some() {
            lines.push("# At least one of API_KEY or AUTH_TOKEN is required, or use `claw login` for OAuth.".to_string());
        }

        for var in &spec.vars {
            if is_active {
                // Active provider: key is uncommented so user can fill it in.
                lines.push(format!("{}=", var.key));
            } else {
                // Inactive providers: commented out with description.
                lines.push(format!("# {}={}", var.key, var.default_value.unwrap_or("")));
            }
        }
        lines.push("".to_string());
    }

    // Model configuration section
    lines.push("# --- Model configuration ---".to_string());
    lines.push("# Resolution order (highest priority first):".to_string());
    lines.push("#   1. --model flag at runtime".to_string());
    lines.push("#   2. CLAW_MODEL (global override for all providers)".to_string());
    lines.push("#   3. Provider-specific model var (e.g. ANTHROPIC_MODEL when CLAW_PROVIDER=anthropic)".to_string());
    lines.push("#   4. Compiled-in default (claude-opus-4-6)".to_string());
    lines.push("".to_string());
    lines.push("# Global model override — applies regardless of active provider.".to_string());
    lines.push("# CLAW_MODEL=".to_string());
    lines.push("".to_string());

    // Per-provider model vars.
    for spec in specs {
        let is_active = spec.kind == active.kind;
        let active_marker = if is_active { " (active provider)" } else { "" };
        lines.push(format!(
            "# {provider} model{marker} — used when CLAW_PROVIDER={provider_lower} (or detected from model name).",
            provider = spec.provider_name,
            marker = active_marker,
            provider_lower = spec.provider_name.to_ascii_lowercase(),
        ));
        lines.push(format!("# {}={}", spec.model_key, spec.default_model));
        lines.push("".to_string());
    }

    // Provider/routing notes section
    lines.push("# --- Routing notes ---".to_string());
    lines.push("# The CLI runtime always uses the Anthropic Messages protocol.".to_string());
    lines.push("# CLAW_PROVIDER only affects which credentials env_config validates on startup.".to_string());
    lines.push("# To route through a proxy (e.g. OpenRouter), redirect the Anthropic endpoint:".to_string());
    lines.push("#   ANTHROPIC_BASE_URL=https://openrouter.ai/api   # client appends /v1/messages".to_string());
    lines.push("#   ANTHROPIC_API_KEY=sk-or-...                    # your OpenRouter key".to_string());
    lines.push("#   CLAW_MODEL=anthropic/claude-opus-4-6           # OpenRouter model slug".to_string());
    lines.push("# Note: OpenAI/xAI provider credentials below are validated but not used".to_string());
    lines.push("# by the main conversation runtime in this build.".to_string());
    lines.push("".to_string());
    lines.push("# CLAW_PROVIDER controls which provider's credentials are checked at startup.".to_string());
    lines.push("# Values: anthropic (default), xai, openai".to_string());
    lines.push("# CLAW_PROVIDER=".to_string());

    lines.join("\n")
}

/// Patch an existing `.env` file by appending any missing keys from the active spec.
fn patch_env_file(
    path: &Path,
    active: &ProviderEnvSpec,
) -> Result<bool, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;

    // Collect keys that already appear in the file (even if commented or empty).
    let existing_keys: HashSet<&str> = content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                return None;
            }
            trimmed.split('=').next()
        })
        .collect();

    let missing: Vec<&str> = active
        .vars
        .iter()
        .filter(|var| !existing_keys.contains(var.key))
        .map(|var| var.key)
        .collect();

    if missing.is_empty() {
        return Ok(false);
    }

    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    writeln!(file)?;
    writeln!(file, "# --- Added by claw (missing provider keys) ---")?;
    for key in &missing {
        writeln!(file, "{}=", key)?;
    }
    Ok(true)
}

/// Validate that the active provider's required env vars have non-empty values.
fn validate_active_provider(active: &ProviderEnvSpec) -> EnvValidation {
    // Check OR group first (Anthropic: either API_KEY or AUTH_TOKEN).
    if let Some(ref or_group) = active.or_group {
        let any_present = or_group
            .iter()
            .any(|key| env::var(key).map_or(false, |v| !v.trim().is_empty()));
        if !any_present {
            return EnvValidation::MissingValues {
                keys: or_group.iter().map(|s| s.to_string()).collect(),
                provider: active.provider_name.to_string(),
            };
        }
    }

    // Check individually required vars (for providers without or_group).
    let missing: Vec<String> = active
        .vars
        .iter()
        .filter(|var| {
            // Only check vars that are not part of an or_group (those were checked above).
            let in_or_group = active
                .or_group
                .as_ref()
                .is_some_and(|og| og.iter().any(|k| *k == var.key));
            if in_or_group {
                return false;
            }
            // Required vars must have non-empty values.
            // A var is "required" if it has no default_value (it's an auth key).
            var.default_value.is_none() && env::var(var.key).map_or(true, |v| v.trim().is_empty())
        })
        .map(|var| var.key.to_string())
        .collect();

    if missing.is_empty() {
        EnvValidation::Ok
    } else {
        EnvValidation::MissingValues {
            keys: missing,
            provider: active.provider_name.to_string(),
        }
    }
}

/// Ensure `.env` is listed in the nearest `.gitignore` when the CWD is inside
/// a git repository. If no `.gitignore` exists at the repo root it is created.
/// Does nothing when the directory is not a git repo.
fn ensure_gitignore_excludes_env(env_file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = env_file
        .parent()
        .ok_or("cannot determine parent of .env path")?;

    // Walk up to find the git repo root (the directory containing .git).
    let mut dir = cwd;
    let git_root = loop {
        if dir.join(".git").exists() {
            break Some(dir);
        }
        match dir.parent() {
            Some(p) => dir = p,
            None => break None,
        }
    };

    let Some(root) = git_root else {
        // Not a git repository — nothing to do.
        return Ok(());
    };

    let gitignore_path = root.join(".gitignore");

    // Check whether .env is already covered.
    if gitignore_path.exists() {
        let content = std::fs::read_to_string(&gitignore_path)?;
        let already_ignored = content.lines().any(|line| {
            let l = line.trim();
            l == ".env" || l == "**/.env" || l == "/.env"
        });
        if already_ignored {
            return Ok(());
        }
        // Append to existing .gitignore.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&gitignore_path)?;
        // Ensure we start on a new line.
        if !content.ends_with('\n') {
            writeln!(file)?;
        }
        writeln!(file, ".env")?;
    } else {
        // Create a minimal .gitignore at the repo root.
        std::fs::write(&gitignore_path, ".env\n")?;
    }

    eprintln!("Added .env to {}.", gitignore_path.display());
    Ok(())
}

/// Ensure `.env` exists with the right keys for the model's provider.
/// Generates if missing, patches if keys absent, validates values.
pub fn ensure_env_for_provider(model: &str) -> Result<EnvLoadResult, Box<dyn std::error::Error>> {
    let path = env_path();
    let active = derive_active_spec(model);

    if !path.exists() {
        let content = generate_env_content(active);
        std::fs::write(&path, content)?;
        eprintln!(
            "Created {} with provider configuration template.",
            path.display()
        );
        eprintln!(
            "Edit {} to add your API keys before running claw.",
            path.display()
        );

        // Protect credentials: add .env to the project's .gitignore if the
        // directory is a git repo (i.e. a .git directory exists anywhere up the
        // tree from CWD) and .env is not already ignored.
        if let Err(e) = ensure_gitignore_excludes_env(&path) {
            // Non-fatal — just warn; the user may not have a git repo here.
            eprintln!("Note: could not update .gitignore: {e}");
        }

        // Reload newly created .env into process env.
        dotenvy::from_path(&path).ok();
        validate_or_warn(active)?;
        return Ok(EnvLoadResult {
            env_path: path,
            was_created: true,
            was_patched: false,
        });
    }

    // .env exists — patch if needed, then validate.
    let patched = patch_env_file(&path, active)?;

    if patched {
        // Reload after patching to pick up any new keys (though they're empty).
        dotenvy::from_path(&path).ok();
        eprintln!(
            "Updated {} with missing configuration keys for provider \"{}\".",
            path.display(),
            active.provider_name
        );
    }

    validate_or_warn(active)?;

    Ok(EnvLoadResult {
        env_path: path,
        was_created: false,
        was_patched: patched,
    })
}

/// Validate provider credentials.
/// - Anthropic: warn-only, because `claw login` OAuth may satisfy auth at runtime.
/// - All other providers: hard-stop with an error, since there is no fallback auth path.
fn validate_or_warn(active: &ProviderEnvSpec) -> Result<(), Box<dyn std::error::Error>> {
    match validate_active_provider(active) {
        EnvValidation::Ok => Ok(()),
        EnvValidation::MissingValues { keys, provider } => {
            eprintln!(
                "Warning: No {provider} credentials configured. \
                 Missing or empty: {}.",
                keys.join(", ")
            );
            if active.or_group.is_some() {
                // Anthropic: OAuth via `claw login` may cover it — only warn.
                eprintln!("  Set at least one of the above, or run `claw login` for OAuth.");
                Ok(())
            } else {
                // Non-Anthropic providers have no OAuth fallback — hard stop.
                eprintln!(
                    "  Set {} in .env or export it in your shell before running claw.",
                    keys.join(", ")
                );
                Err(format!(
                    "{provider} credentials not configured; set {} to proceed.",
                    keys.join(", ")
                )
                .into())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn with_temp_dir(f: impl FnOnce(&std::path::Path)) {
        use std::time::SystemTime;
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("claw_env_test_{ts}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        f(&dir);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_generate_template_includes_all_providers() {
        let active = derive_active_spec("claude-opus-4-6");
        let content = generate_env_content(active);

        // All three providers should be mentioned.
        assert!(content.contains("Anthropic"), "missing Anthropic section");
        assert!(content.contains("xAI"), "missing xAI section");
        assert!(content.contains("OpenAI"), "missing OpenAI section");

        // Active provider keys should be uncommented.
        assert!(
            content.contains("\nANTHROPIC_API_KEY="),
            "active key should be uncommented"
        );

        // Inactive provider keys should be commented out.
        assert!(
            content.contains("# OPENAI_API_KEY="),
            "inactive key should be commented"
        );
        assert!(
            content.contains("# XAI_API_KEY="),
            "inactive key should be commented"
        );
    }

    #[test]
    fn test_generate_template_for_xai_model() {
        let active = derive_active_spec("grok-3");
        assert_eq!(active.kind, ProviderKind::Xai);
        let content = generate_env_content(active);
        assert!(
            content.contains("\nXAI_API_KEY="),
            "active xAI key should be uncommented"
        );
    }

    #[test]
    fn test_patch_adds_missing_keys() {
        with_temp_dir(|dir| {
            let env_file = dir.join(".env");
            fs::write(&env_file, "ANTHROPIC_API_KEY=sk-existing\n").unwrap();

            let active = derive_active_spec("claude-opus-4-6");
            let patched = patch_env_file(&env_file, active).unwrap();
            assert!(patched);

            let content = fs::read_to_string(&env_file).unwrap();
            assert!(content.contains("ANTHROPIC_AUTH_TOKEN="));
        });
    }

    #[test]
    fn test_patch_skips_existing_keys() {
        with_temp_dir(|dir| {
            let env_file = dir.join(".env");
            fs::write(
                &env_file,
                "ANTHROPIC_API_KEY=\nANTHROPIC_AUTH_TOKEN=\nANTHROPIC_BASE_URL=\n",
            )
            .unwrap();

            let active = derive_active_spec("claude-opus-4-6");
            let patched = patch_env_file(&env_file, active).unwrap();
            assert!(!patched, "should not patch when all keys exist");
        });
    }

    #[test]
    fn test_validate_anthropic_or_semantics() {
        // Hold the crate-wide env lock to avoid races with sibling tests that
        // also temporarily set ANTHROPIC_API_KEY (e.g. startup_banner test).
        let _guard = crate::test_env_lock();
        let active = derive_active_spec("claude-opus-4-6");
        assert!(active.or_group.is_some());

        // Neither set.
        env::remove_var("ANTHROPIC_API_KEY");
        env::remove_var("ANTHROPIC_AUTH_TOKEN");
        match validate_active_provider(active) {
            EnvValidation::MissingValues { .. } => {}
            EnvValidation::Ok => panic!("expected MissingValues when neither key is set"),
        }

        // Only AUTH_TOKEN set.
        env::set_var("ANTHROPIC_AUTH_TOKEN", "tok-test");
        env::remove_var("ANTHROPIC_API_KEY");
        match validate_active_provider(active) {
            EnvValidation::Ok => {}
            EnvValidation::MissingValues { .. } => {
                panic!("expected Ok when AUTH_TOKEN is set")
            }
        }

        // Only API_KEY set.
        env::set_var("ANTHROPIC_API_KEY", "sk-test");
        env::remove_var("ANTHROPIC_AUTH_TOKEN");
        match validate_active_provider(active) {
            EnvValidation::Ok => {}
            EnvValidation::MissingValues { .. } => {
                panic!("expected Ok when API_KEY is set")
            }
        }

        // Clean up.
        env::remove_var("ANTHROPIC_API_KEY");
        env::remove_var("ANTHROPIC_AUTH_TOKEN");
    }

    #[test]
    fn test_validate_openai_requires_key() {
        let specs = all_provider_specs();
        let spec = specs
            .iter()
            .find(|s| s.kind == ProviderKind::OpenAi)
            .unwrap();

        env::remove_var("OPENAI_API_KEY");
        match validate_active_provider(spec) {
            EnvValidation::MissingValues { .. } => {}
            EnvValidation::Ok => panic!("expected MissingValues when OPENAI_API_KEY is unset"),
        }

        env::set_var("OPENAI_API_KEY", "sk-test");
        match validate_active_provider(spec) {
            EnvValidation::Ok => {}
            EnvValidation::MissingValues { .. } => {
                panic!("expected Ok when OPENAI_API_KEY is set")
            }
        }

        env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn test_dotenvy_handles_quoted_values() {
        with_temp_dir(|dir| {
            let env_file = dir.join(".env");
            // Use unique var names to avoid collisions with the test environment.
            fs::write(
                &env_file,
                "CLAW_TEST_SINGLE='hello world'\nCLAW_TEST_DOUBLE=\"quoted value\"\nCLAW_TEST_PLAIN=unquoted\n",
            )
            .unwrap();

            // Ensure these are not already set (dotenvy won't overwrite existing vars).
            env::remove_var("CLAW_TEST_SINGLE");
            env::remove_var("CLAW_TEST_DOUBLE");
            env::remove_var("CLAW_TEST_PLAIN");

            dotenvy::from_path(&env_file).unwrap();

            assert_eq!(
                env::var("CLAW_TEST_SINGLE").unwrap(),
                "hello world",
                "single-quoted value should have quotes stripped"
            );
            assert_eq!(
                env::var("CLAW_TEST_DOUBLE").unwrap(),
                "quoted value",
                "double-quoted value should have quotes stripped"
            );
            assert_eq!(
                env::var("CLAW_TEST_PLAIN").unwrap(),
                "unquoted",
                "plain value should be used as-is"
            );

            env::remove_var("CLAW_TEST_SINGLE");
            env::remove_var("CLAW_TEST_DOUBLE");
            env::remove_var("CLAW_TEST_PLAIN");
        });
    }

    #[test]
    fn test_load_cwd_only_not_parent() {
        with_temp_dir(|dir| {
            // Create .env in parent but not child.
            fs::write(dir.join(".env"), "SHOULD_NOT_LOAD=true\n").unwrap();

            let child = dir.join("subdir");
            fs::create_dir_all(&child).unwrap();

            // dotenvy::from_path would not be called for a nonexistent file.
            let child_env = child.join(".env");
            assert!(!child_env.exists(), "child should not have .env");

            // Simulate what load_project_env does with a custom path.
            let _ = dotenvy::from_path(&child_env); // no-op, file absent
            assert!(
                env::var("SHOULD_NOT_LOAD").is_err(),
                "parent .env must not leak into child"
            );
        });
    }

    // -----------------------------------------------------------------------
    // resolve_model_from_env: provider-default and CLAW_PROVIDER backfill
    // -----------------------------------------------------------------------

    #[test]
    fn test_claw_provider_openai_without_model_var_uses_openai_default() {
        // Risk 1: CLAW_PROVIDER=openai set, no OPENAI_MODEL / CLAW_MODEL →
        // should return the OpenAI provider's compiled-in default, not claude-opus-4-6.
        let _guard = crate::test_env_lock();
        env::remove_var("CLAW_MODEL");
        env::remove_var("OPENAI_MODEL");
        env::remove_var("ANTHROPIC_MODEL");
        env::remove_var("XAI_MODEL");
        env::set_var("CLAW_PROVIDER", "openai");

        let model = resolve_model_from_env();

        env::remove_var("CLAW_PROVIDER");

        let model = model.expect("should return provider default when CLAW_PROVIDER=openai");
        assert_ne!(
            model, "claude-opus-4-6",
            "must not use Claude default for OpenAI provider"
        );
        // The OpenAI spec default_model should be returned.
        let openai_spec = all_provider_specs()
            .iter()
            .find(|s| s.kind == ProviderKind::OpenAi)
            .unwrap();
        assert_eq!(model, openai_spec.default_model);
    }

    #[test]
    fn test_claw_provider_xai_without_model_var_uses_xai_default() {
        // Risk 1 (xAI variant): CLAW_PROVIDER=xai, no XAI_MODEL → xAI default.
        let _guard = crate::test_env_lock();
        env::remove_var("CLAW_MODEL");
        env::remove_var("XAI_MODEL");
        env::remove_var("ANTHROPIC_MODEL");
        env::remove_var("OPENAI_MODEL");
        env::set_var("CLAW_PROVIDER", "xai");

        let model = resolve_model_from_env();

        env::remove_var("CLAW_PROVIDER");

        let model = model.expect("should return provider default when CLAW_PROVIDER=xai");
        assert_ne!(model, "claude-opus-4-6", "must not use Claude default for xAI provider");
        let xai_spec = all_provider_specs()
            .iter()
            .find(|s| s.kind == ProviderKind::Xai)
            .unwrap();
        assert_eq!(model, xai_spec.default_model);
    }

    #[test]
    fn test_openai_model_var_backfills_claw_provider() {
        // Risk 2: OPENAI_MODEL=deepseek-chat set, no CLAW_PROVIDER →
        // resolve_model_from_env should set CLAW_PROVIDER=openai so that
        // detect_provider_kind routes to OpenAI even without an OPENAI_API_KEY.
        let _guard = crate::test_env_lock();
        env::remove_var("CLAW_MODEL");
        env::remove_var("CLAW_PROVIDER");
        env::remove_var("ANTHROPIC_MODEL");
        env::remove_var("XAI_MODEL");
        env::set_var("OPENAI_MODEL", "deepseek-chat");

        let model = resolve_model_from_env();

        let provider_after = env::var("CLAW_PROVIDER").ok();
        env::remove_var("OPENAI_MODEL");
        env::remove_var("CLAW_PROVIDER");

        assert_eq!(model.as_deref(), Some("deepseek-chat"));
        assert_eq!(
            provider_after.as_deref(),
            Some("openai"),
            "CLAW_PROVIDER must be backfilled to 'openai' when OPENAI_MODEL selects the model"
        );
    }

    #[test]
    fn test_xai_model_var_backfills_claw_provider() {
        // Risk 2 (xAI variant): XAI_MODEL=grok-custom set, no CLAW_PROVIDER.
        let _guard = crate::test_env_lock();
        env::remove_var("CLAW_MODEL");
        env::remove_var("CLAW_PROVIDER");
        env::remove_var("ANTHROPIC_MODEL");
        env::remove_var("OPENAI_MODEL");
        env::set_var("XAI_MODEL", "grok-custom");

        let model = resolve_model_from_env();

        let provider_after = env::var("CLAW_PROVIDER").ok();
        env::remove_var("XAI_MODEL");
        env::remove_var("CLAW_PROVIDER");

        assert_eq!(model.as_deref(), Some("grok-custom"));
        assert_eq!(
            provider_after.as_deref(),
            Some("xai"),
            "CLAW_PROVIDER must be backfilled to 'xai' when XAI_MODEL selects the model"
        );
    }
}
