#[cfg(target_os = "android")]
pub mod android;

#[cfg(target_family = "wasm")]
pub mod wasm;

#[cfg(not(target_family = "wasm"))]
use std::path::PathBuf;
use tokio::io::AsyncRead;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PickFileError {
    #[error("No file was selected")]
    NoFileSelected,
    #[error("No directory was selected")]
    NoDirectorySelected,
    #[error("IO error while saving file.")]
    IoError(#[from] std::io::Error),
}

/// Result of picking a file - contains filename and reader
pub struct PickedFile<R: AsyncRead + Unpin> {
    pub name: String,
    pub reader: R,
}

/// Pick a file and return the name & reader of the file.
pub async fn pick_file() -> Result<PickedFile<impl AsyncRead + Unpin>, PickFileError> {
    #[cfg(target_os = "ios")]
    {
        Err::<PickedFile<tokio::fs::File>, PickFileError>(PickFileError::NoFileSelected)
    }

    #[cfg(all(
        not(target_os = "android"),
        not(target_os = "ios"),
        not(target_family = "wasm")
    ))]
    {
        let file = rfd::AsyncFileDialog::new()
            .pick_file()
            .await
            .ok_or(PickFileError::NoFileSelected)?;

        let name = file.file_name();
        let file = tokio::fs::File::open(file.path()).await?;
        Ok(PickedFile { name, reader: file })
    }

    #[cfg(target_family = "wasm")]
    {
        wasm::pick_file().await
    }

    #[cfg(target_os = "android")]
    {
        let file = android::pick_file().await?;
        Ok(PickedFile {
            name: "file".to_owned(), // Android doesn't easily give us the filename
            reader: file,
        })
    }
}

/// A file being saved incrementally, so large outputs never have to exist in
/// memory as a whole. Natively the user picks a path up front and writes go
/// straight to disk; on the web they are collected into chunked Blob parts
/// and offered as a download by [`SaveTarget::finish`].
///
/// File dialogs are currently unavailable on Android and iOS.
pub struct SaveTarget {
    #[cfg(all(
        not(target_os = "android"),
        not(target_os = "ios"),
        not(target_family = "wasm")
    ))]
    inner: std::io::BufWriter<std::fs::File>,
    #[cfg(target_family = "wasm")]
    inner: wasm::BlobPartsWriter,
    #[cfg(target_family = "wasm")]
    name: String,
    #[cfg(any(target_os = "android", target_os = "ios"))]
    inner: std::io::Sink,
}

impl SaveTarget {
    pub async fn pick(default_name: &str) -> Result<Self, PickFileError> {
        #[cfg(all(
            not(target_os = "android"),
            not(target_os = "ios"),
            not(target_family = "wasm")
        ))]
        {
            let file = rfd::AsyncFileDialog::new()
                .set_file_name(default_name)
                .save_file()
                .await
                .ok_or(PickFileError::NoFileSelected)?;
            let file = std::fs::File::create(file.path())?;
            Ok(Self {
                inner: std::io::BufWriter::with_capacity(1 << 20, file),
            })
        }

        #[cfg(target_family = "wasm")]
        {
            Ok(Self {
                inner: wasm::BlobPartsWriter::new(),
                name: default_name.to_owned(),
            })
        }

        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            let _ = default_name;
            Err(PickFileError::NoFileSelected)
        }
    }

    pub fn finish(self) -> Result<(), PickFileError> {
        #[cfg(all(
            not(target_os = "android"),
            not(target_os = "ios"),
            not(target_family = "wasm")
        ))]
        {
            use std::io::Write;
            let mut inner = self.inner;
            inner.flush()?;
            Ok(())
        }

        #[cfg(target_family = "wasm")]
        {
            self.inner.save(&self.name)
        }

        #[cfg(any(target_os = "android", target_os = "ios"))]
        {
            Err(PickFileError::NoFileSelected)
        }
    }
}

impl std::io::Write for SaveTarget {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(all(
    test,
    not(target_family = "wasm"),
    not(target_os = "android"),
    not(target_os = "ios")
))]
mod save_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn finish_flushes_buffered_bytes_and_reports_write_errors() {
        let path = std::env::temp_dir().join(format!(
            "brush-save-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let file = std::fs::File::create_new(&path).unwrap();
        let mut target = SaveTarget {
            inner: std::io::BufWriter::new(file),
        };
        target.write_all(b"PLY bytes").unwrap();
        target.finish().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"PLY bytes");

        let file = std::fs::File::open(&path).unwrap();
        let mut target = SaveTarget {
            inner: std::io::BufWriter::new(file),
        };
        target.write_all(b"buffered").unwrap();
        assert!(matches!(target.finish(), Err(PickFileError::IoError(_))));
        assert_eq!(std::fs::read(&path).unwrap(), b"PLY bytes");
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(not(target_family = "wasm"))]
pub async fn pick_directory() -> Result<PathBuf, PickFileError> {
    #[cfg(target_os = "ios")]
    {
        Err(PickFileError::NoDirectorySelected)
    }

    #[cfg(all(not(target_os = "android"), not(target_os = "ios")))]
    {
        let dir = rfd::AsyncFileDialog::new()
            .pick_folder()
            .await
            .ok_or(PickFileError::NoDirectorySelected)?;

        Ok(dir.path().to_path_buf())
    }

    #[cfg(target_os = "android")]
    {
        panic!("No picking directories on Android yet.")
    }
}

/// Saves data to a file and returns the filename the data was saved too.
///
/// Nb: Does not work on Android currently.
pub async fn save_file(default_name: &str, data: Vec<u8>) -> Result<(), PickFileError> {
    #[cfg(target_os = "ios")]
    {
        let _ = default_name;
        let _ = data;
        Err(PickFileError::NoFileSelected)
    }

    #[cfg(all(
        not(target_os = "android"),
        not(target_os = "ios"),
        not(target_family = "wasm")
    ))]
    {
        let file = rfd::AsyncFileDialog::new()
            .set_file_name(default_name)
            .save_file()
            .await
            .ok_or(PickFileError::NoFileSelected)?;

        tokio::fs::write(file.path(), data).await?;

        Ok(())
    }

    #[cfg(target_family = "wasm")]
    {
        wasm::save_file(default_name, &data).await
    }

    #[cfg(target_os = "android")]
    {
        let _ = default_name;
        let _ = data;
        panic!("No saving on Android yet.")
    }
}
