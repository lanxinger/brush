use bytes::Bytes;
use futures_channel;
use futures_util::StreamExt;
use js_sys::Uint8Array;
use std::io;
use std::path::PathBuf;
use tokio::io::AsyncRead;
use tokio_util::io::StreamReader;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;

use wasm_streams::ReadableStream as WasmReadableStream;
use web_sys::{Blob, Event, HtmlAnchorElement, HtmlInputElement};

use crate::PickFileError;

/// A handle to a directory picked by the user via the File System Access API.
/// Can be used to read files on demand by path.
#[derive(Clone, Debug)]
pub struct DirectoryHandle {
    handle: web_sys::FileSystemDirectoryHandle,
}

impl DirectoryHandle {
    /// Wrap a `FileSystemDirectoryHandle` obtained from JS (e.g. `showDirectoryPicker`).
    pub fn from_handle(handle: web_sys::FileSystemDirectoryHandle) -> Self {
        Self { handle }
    }

    /// Get a file handle for the given path within this directory.
    /// The path can contain subdirectories (e.g., "subdir/file.txt").
    pub async fn get_file(&self, path: &std::path::Path) -> Result<web_sys::File, PickFileError> {
        let components: Vec<_> = path.components().collect();
        if components.is_empty() {
            return Err(PickFileError::NoFileSelected);
        }

        // Navigate through subdirectories
        let mut current_dir = self.handle.clone();
        for component in &components[..components.len() - 1] {
            let dir_name = component
                .as_os_str()
                .to_str()
                .ok_or(PickFileError::NoFileSelected)?;
            let promise = current_dir.get_directory_handle(dir_name);
            let result = wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|_e| PickFileError::NoFileSelected)?;
            current_dir = result.unchecked_into();
        }

        // Get the file from the final directory
        let file_name = components
            .last()
            .unwrap()
            .as_os_str()
            .to_str()
            .ok_or(PickFileError::NoFileSelected)?;

        let promise = current_dir.get_file_handle(file_name);
        let file_handle: web_sys::FileSystemFileHandle =
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|_e| PickFileError::NoFileSelected)?
                .unchecked_into();

        let promise = file_handle.get_file();
        let file: web_sys::File = wasm_bindgen_futures::JsFuture::from(promise)
            .await
            .map_err(|_e| PickFileError::NoFileSelected)?
            .unchecked_into();

        Ok(file)
    }

    /// List all files in the directory recursively.
    pub async fn list_files(&self) -> Result<Vec<PathBuf>, PickFileError> {
        let mut files = Vec::new();
        self.list_files_recursive(&self.handle, PathBuf::new(), &mut files)
            .await?;
        Ok(files)
    }

    fn list_files_recursive<'a>(
        &'a self,
        dir: &'a web_sys::FileSystemDirectoryHandle,
        prefix: PathBuf,
        files: &'a mut Vec<PathBuf>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), PickFileError>> + 'a>> {
        Box::pin(async move {
            // Get async iterator from entries()
            let entries = js_sys::Reflect::get(dir, &"entries".into())
                .map_err(|_e| PickFileError::NoDirectorySelected)?
                .dyn_into::<js_sys::Function>()
                .map_err(|_e| PickFileError::NoDirectorySelected)?
                .call0(dir)
                .map_err(|_e| PickFileError::NoDirectorySelected)?;

            loop {
                // Call next() on the async iterator
                let next_fn = js_sys::Reflect::get(&entries, &"next".into())
                    .map_err(|_e| PickFileError::NoDirectorySelected)?
                    .dyn_into::<js_sys::Function>()
                    .map_err(|_e| PickFileError::NoDirectorySelected)?;

                let promise = next_fn
                    .call0(&entries)
                    .map_err(|_e| PickFileError::NoDirectorySelected)?
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_e| PickFileError::NoDirectorySelected)?;

                let result = wasm_bindgen_futures::JsFuture::from(promise)
                    .await
                    .map_err(|_e| PickFileError::NoDirectorySelected)?;

                let done = js_sys::Reflect::get(&result, &"done".into())
                    .map_err(|_e| PickFileError::NoDirectorySelected)?
                    .as_bool()
                    .unwrap_or(true);

                if done {
                    break;
                }

                let value = js_sys::Reflect::get(&result, &"value".into())
                    .map_err(|_e| PickFileError::NoDirectorySelected)?;

                let array: js_sys::Array = value.unchecked_into();
                let name: String = array
                    .get(0)
                    .as_string()
                    .ok_or(PickFileError::NoDirectorySelected)?;
                let handle: web_sys::FileSystemHandle = array.get(1).unchecked_into();

                // Get kind as string property
                let kind = js_sys::Reflect::get(&handle, &"kind".into())
                    .map_err(|_e| PickFileError::NoDirectorySelected)?
                    .as_string()
                    .ok_or(PickFileError::NoDirectorySelected)?;

                let path = prefix.join(&name);

                if kind == "file" {
                    files.push(path);
                } else if kind == "directory" {
                    let subdir: web_sys::FileSystemDirectoryHandle = handle.unchecked_into();
                    self.list_files_recursive(&subdir, path, files).await?;
                }
            }
            Ok(())
        })
    }
}

/// Pick a directory using the File System Access API (`showDirectoryPicker`).
/// Returns a [`DirectoryHandle`] that can be used to read files on demand.
pub async fn pick_directory_handle() -> Result<DirectoryHandle, PickFileError> {
    let window = web_sys::window().ok_or(PickFileError::NoDirectorySelected)?;

    // Call showDirectoryPicker()
    let promise = js_sys::Reflect::get(&window, &"showDirectoryPicker".into())
        .map_err(|_e| PickFileError::NoDirectorySelected)?
        .dyn_into::<js_sys::Function>()
        .map_err(|_e| PickFileError::NoDirectorySelected)?
        .call0(&window)
        .map_err(|_e| PickFileError::NoDirectorySelected)?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_e| PickFileError::NoDirectorySelected)?;

    let handle = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_e| PickFileError::NoDirectorySelected)?
        .dyn_into::<web_sys::FileSystemDirectoryHandle>()
        .map_err(|_e| PickFileError::NoDirectorySelected)?;

    Ok(DirectoryHandle { handle })
}

pub async fn save_file(default_name: &str, data: &[u8]) -> Result<(), PickFileError> {
    let blob_parts = js_sys::Array::new();
    for chunk in data.chunks(BlobPartsWriter::PART_SIZE) {
        blob_parts.push(&Uint8Array::from(chunk));
    }
    download_blob_parts(default_name, &blob_parts)
}

/// `io::Write` sink that copies data into JS-side `Uint8Array` parts as it
/// goes, so the wasm heap only ever holds one part-sized buffer.
pub struct BlobPartsWriter {
    parts: js_sys::Array,
    buf: Vec<u8>,
}

impl BlobPartsWriter {
    const PART_SIZE: usize = 16 * 1024 * 1024;

    pub fn new() -> Self {
        Self {
            parts: js_sys::Array::new(),
            buf: Vec::with_capacity(Self::PART_SIZE),
        }
    }

    fn flush_part(&mut self) {
        if !self.buf.is_empty() {
            self.parts.push(&Uint8Array::from(self.buf.as_slice()));
            self.buf.clear();
        }
    }

    /// Offer the collected data to the user as a download.
    pub fn save(mut self, default_name: &str) -> Result<(), PickFileError> {
        self.flush_part();
        drop(std::mem::take(&mut self.buf));
        download_blob_parts(default_name, &self.parts)
    }
}

impl Default for BlobPartsWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl io::Write for BlobPartsWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let n = data.len().min(Self::PART_SIZE - self.buf.len());
        self.buf.extend_from_slice(&data[..n]);
        if self.buf.len() >= Self::PART_SIZE {
            self.flush_part();
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_part();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use wasm_bindgen_test::wasm_bindgen_test;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn blob_parts_preserve_bytes_across_flushes_and_part_boundaries() {
        let bytes: Vec<u8> = (0..BlobPartsWriter::PART_SIZE + 97)
            .map(|i| (i % 251) as u8)
            .collect();
        let mut writer = BlobPartsWriter::new();
        writer.write_all(&bytes[..3]).unwrap();
        writer.flush().unwrap();
        writer.write_all(&bytes[3..]).unwrap();
        assert!(writer.buf.len() < BlobPartsWriter::PART_SIZE);
        writer.flush().unwrap();
        assert!(writer.buf.is_empty());
        assert_eq!(writer.parts.length(), 3);
        let blob = Blob::new_with_u8_array_sequence(&writer.parts).unwrap();
        assert_eq!(blob.size() as usize, bytes.len());
        let buffer = wasm_bindgen_futures::JsFuture::from(blob.array_buffer())
            .await
            .unwrap();
        assert!(
            Uint8Array::new(&buffer).to_vec() == bytes,
            "Blob bytes changed"
        );
    }
}

fn download_blob_parts(
    default_name: &str,
    blob_parts: &js_sys::Array,
) -> Result<(), PickFileError> {
    let window = web_sys::window().ok_or(PickFileError::NoFileSelected)?;
    let document = window.document().ok_or(PickFileError::NoFileSelected)?;
    let body = document.body().ok_or(PickFileError::NoFileSelected)?;

    let blob =
        Blob::new_with_u8_array_sequence(blob_parts).map_err(|_e| PickFileError::NoFileSelected)?;

    let anchor = document
        .create_element("a")
        .map_err(|_e| PickFileError::NoFileSelected)?
        .dyn_into::<HtmlAnchorElement>()
        .map_err(|_e| PickFileError::NoFileSelected)?;

    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|_e| PickFileError::NoFileSelected)?;
    anchor.set_href(&url);
    anchor.set_download(default_name);

    // Append to body, click, then remove - some browsers ignore clicks on detached elements.
    if body.append_child(&anchor).is_err() {
        let _ = web_sys::Url::revoke_object_url(&url);
        return Err(PickFileError::NoFileSelected);
    }
    anchor.click();
    let _ = body.remove_child(&anchor);

    // Revoking right after click() can abort large downloads before the
    // browser has started reading the blob, so release it a bit later.
    let delayed_url = url.clone();
    let revoke = Closure::once(move || {
        let _ = web_sys::Url::revoke_object_url(&delayed_url);
    });
    if window
        .set_timeout_with_callback_and_timeout_and_arguments_0(
            revoke.as_ref().unchecked_ref(),
            60_000,
        )
        .is_ok()
    {
        revoke.forget();
    } else {
        let _ = web_sys::Url::revoke_object_url(&url);
    }
    Ok(())
}

pub async fn pick_file() -> Result<crate::PickedFile<impl AsyncRead + Unpin>, PickFileError> {
    let window = web_sys::window().ok_or(PickFileError::NoFileSelected)?;
    let document = window.document().ok_or(PickFileError::NoFileSelected)?;

    let input = document
        .create_element("input")
        .map_err(|_e| PickFileError::NoFileSelected)?
        .dyn_into::<HtmlInputElement>()
        .map_err(|_e| PickFileError::NoFileSelected)?;

    input.set_type("file");

    let (sender, receiver) = futures_channel::oneshot::channel();
    let sender = std::rc::Rc::new(std::cell::RefCell::new(Some(sender)));

    let onchange = {
        let sender = sender.clone();
        let input = input.clone();
        Closure::wrap(Box::new(move |_: Event| {
            if let Some(sender) = sender.borrow_mut().take() {
                let files = input.files();
                let _ = sender.send(files);
            }
        }) as Box<dyn FnMut(_)>)
    };
    let oncancel = {
        let sender = sender.clone();
        Closure::wrap(Box::new(move |_: Event| {
            if let Some(sender) = sender.borrow_mut().take() {
                let _ = sender.send(None);
            }
        }) as Box<dyn FnMut(_)>)
    };

    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    input.set_oncancel(Some(oncancel.as_ref().unchecked_ref()));
    input.click();

    let files = receiver
        .await
        .map_err(|_e| PickFileError::NoFileSelected)?
        .ok_or(PickFileError::NoFileSelected)?;

    let file = files.get(0).ok_or(PickFileError::NoFileSelected)?;
    let name = file.name();

    let readable_stream = file.stream();
    let wasm_stream = WasmReadableStream::from_raw(readable_stream);

    let byte_stream = wasm_stream.into_stream().map(|result| {
        result
            .map_err(|e| io::Error::other(format!("Stream error: {e:?}")))
            .and_then(|chunk| {
                if let Ok(uint8_array) = chunk.dyn_into::<Uint8Array>() {
                    let mut data = vec![0; uint8_array.length() as usize];
                    uint8_array.copy_to(&mut data);
                    Ok(Bytes::from(data))
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Invalid chunk type",
                    ))
                }
            })
    });

    Ok(crate::PickedFile {
        name,
        reader: StreamReader::new(byte_stream),
    })
}
