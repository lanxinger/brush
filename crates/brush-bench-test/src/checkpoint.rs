use anyhow::{Result, bail};
use brush_vfs::BrushVfs;
use std::path::PathBuf;

/// Select the only PLY in a mounted checkpoint path.
///
/// A direct PLY path produces a one-file VFS. A directory may contain sidecar
/// files or several exports, so selecting its first hash-map entry is neither
/// type-safe nor deterministic.
pub fn select_checkpoint_ply(vfs: &BrushVfs) -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = vfs.files_with_extension("ply").collect();
    candidates.sort_unstable();

    match candidates.len() {
        0 => bail!(
            "checkpoint contains no .ply file (found {} file(s))",
            vfs.file_count()
        ),
        1 => Ok(candidates.remove(0)),
        count => bail!(
            "checkpoint contains {count} .ply files and cannot choose deterministically: {}. Pass the single .ply you mean.",
            candidates
                .iter()
                .map(|path| format!("'{}'", path.display()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vfs(paths: &[&str]) -> BrushVfs {
        BrushVfs::create_test_vfs(paths.iter().map(PathBuf::from).collect())
    }

    #[test]
    fn selects_the_only_ply_past_sidecars() {
        let mounted = vfs(&["args.txt", "export_30000.ply", "export_30000_dig_mlp.json"]);

        assert_eq!(
            select_checkpoint_ply(&mounted).unwrap(),
            PathBuf::from("export_30000.ply")
        );
    }

    #[test]
    fn rejects_multiple_plys_with_a_stable_candidate_list() {
        let mut messages = std::collections::BTreeSet::new();
        for _ in 0..100 {
            let error = select_checkpoint_ply(&vfs(&[
                "export_30000.ply",
                "args.txt",
                "export_10000.ply",
                "export_20000.ply",
            ]))
            .unwrap_err();
            messages.insert(error.to_string());
        }

        assert_eq!(messages.len(), 1);
        let message = messages.first().unwrap();
        for name in ["export_10000.ply", "export_20000.ply", "export_30000.ply"] {
            assert!(message.contains(name));
        }
    }

    #[test]
    fn rejects_a_checkpoint_without_a_ply() {
        assert!(select_checkpoint_ply(&vfs(&["args.txt", "meta.json"])).is_err());
    }
}
