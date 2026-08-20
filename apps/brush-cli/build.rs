use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn git_output(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .ok()?;

    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn git_is_dirty(repo: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["diff-index", "--quiet", "HEAD", "--"])
        .status()
        .is_ok_and(|status| !status.success())
}

fn track_git_state(repo: &Path) {
    let Some(git_dir) = git_output(repo, &["rev-parse", "--absolute-git-dir"]).map(PathBuf::from)
    else {
        return;
    };

    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
    println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
    println!(
        "cargo:rerun-if-changed={}",
        git_dir.join("packed-refs").display()
    );

    if let Some(head_ref) = git_output(repo, &["symbolic-ref", "-q", "HEAD"]) {
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join(head_ref).display()
        );
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=BRUSH_BUILD_ID");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    track_git_state(&manifest_dir);

    let build_id = env::var("BRUSH_BUILD_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            let revision = git_output(&manifest_dir, &["rev-parse", "HEAD"])?;
            Some(if git_is_dirty(&manifest_dir) {
                format!("{revision}-dirty")
            } else {
                revision
            })
        })
        .unwrap_or_else(|| "unknown".to_owned());
    let version = format!("{} ({build_id})", env!("CARGO_PKG_VERSION"));

    println!("cargo:rustc-env=BRUSH_BUILD_ID={build_id}");
    println!("cargo:rustc-env=BRUSH_VERSION={version}");
}
