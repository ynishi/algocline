//! Rewriting a safetensors bundle's header with a metadata map, and
//! hashing its tensor data on the way through.
//!
//! The tensors are not re-serialised. A safetensors file is a u64
//! little-endian header length, the header's JSON, and a data section
//! whose tensor offsets are relative to its own start — so a new header
//! in front of the same data section is a valid file with the same
//! tensors, and nothing has to reproduce the order `safetensors` sorts
//! tensors into (which the crate does not expose).

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::train::ckpt::read_header_json;

/// Key under which [`rewrite_with_metadata`] records the tensor digest.
///
/// Lowercase hex SHA-256 of the data section — every byte after the
/// header. That is the range ModelSpec's `hash_sha256` covers (without
/// its `0x` prefix), and it is **not** the Hugging Face Hub's LFS
/// `sha256`, which covers the whole file, header included.
pub const TENSOR_SHA256_KEY: &str = "alc.tensor_sha256";

/// Width of a SHA-256 digest in hex.
const DIGEST_HEX_LEN: usize = 64;

/// The header length is padded to a multiple of this, with spaces —
/// what `safetensors` 0.8 does (`tensor.rs`, `prepare`), so the data
/// section starts 8-byte aligned.
const HEADER_ALIGN: usize = 8;

/// Size of the little-endian header-length prefix.
const LEN_PREFIX: u64 = 8;

/// One tensor entry of a safetensors header, as read from it.
#[derive(Debug)]
struct TensorEntry {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
}

impl TensorEntry {
    /// Read one entry, refusing a field the format does not define: the
    /// entry is rewritten field by field, and a field this does not
    /// know would be dropped from the new header without a word.
    fn parse(src: &Path, name: &str, value: &Value) -> Result<Self, String> {
        let bad = |what: &str| format!("{}: tensor {name:?}: {what}", src.display());
        let obj = value
            .as_object()
            .ok_or_else(|| bad("entry is not a JSON object"))?;
        if let Some(extra) = obj
            .keys()
            .find(|k| !matches!(k.as_str(), "dtype" | "shape" | "data_offsets"))
        {
            return Err(bad(&format!(
                "unknown field {extra:?}; a rewrite would drop it"
            )));
        }
        let dtype = obj
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("`dtype` is not a string"))?
            .to_string();
        let shape = obj
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("`shape` is not an array"))?
            .iter()
            .map(|d| d.as_u64().ok_or_else(|| bad("`shape` holds a non-integer")))
            .collect::<Result<Vec<u64>, String>>()?;
        let offsets = obj
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("`data_offsets` is not an array"))?;
        let [begin, end] = offsets.as_slice() else {
            return Err(bad("`data_offsets` does not hold two numbers"));
        };
        let begin = begin
            .as_u64()
            .ok_or_else(|| bad("`data_offsets` holds a non-integer"))?;
        let end = end
            .as_u64()
            .ok_or_else(|| bad("`data_offsets` holds a non-integer"))?;
        if end < begin {
            return Err(bad(&format!(
                "`data_offsets` [{begin}, {end}] runs backwards"
            )));
        }
        Ok(Self {
            name: name.to_string(),
            dtype,
            shape,
            begin,
            end,
        })
    }
}

/// A JSON string literal for `s`.
///
/// `serde_json` escapes a `&str` the same way whatever features it was
/// built with — the features change how *maps* are ordered, and no map
/// passes through here.
fn json_str(s: &str) -> Result<String, String> {
    serde_json::to_string(s).map_err(|e| format!("encode {s:?} as JSON: {e}"))
}

/// The new header, byte for byte, and where the digest goes in it.
///
/// The layout depends on nothing but its inputs: `__metadata__` first
/// with its keys in sorted order, then the tensors in data-offset order,
/// each with its fields as `dtype`, `shape`, `data_offsets`. It does not
/// go through a `serde_json` map, whose order depends on whether
/// something in the build turned on `preserve_order`.
///
/// The digest is written as 64 placeholder characters; the returned
/// offset (into the header bytes) is where the real one is patched in
/// once the data section has been read. A hex digest is always 64
/// characters and needs no escaping, so patching it in changes no
/// length and the file is the one a direct write would produce.
fn serialize_header(
    metadata: &BTreeMap<String, String>,
    tensors: &[TensorEntry],
) -> Result<(Vec<u8>, usize), String> {
    let mut out = String::from("{\"__metadata__\":{");
    let mut digest_at = None;
    for (i, (key, value)) in metadata.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&json_str(key)?);
        out.push(':');
        if key == TENSOR_SHA256_KEY {
            // Past the opening quote of the value.
            digest_at = Some(out.len() + 1);
        }
        out.push_str(&json_str(value)?);
    }
    out.push('}');
    for t in tensors {
        out.push(',');
        out.push_str(&json_str(&t.name)?);
        out.push_str(":{\"dtype\":");
        out.push_str(&json_str(&t.dtype)?);
        out.push_str(",\"shape\":[");
        let dims: Vec<String> = t.shape.iter().map(u64::to_string).collect();
        out.push_str(&dims.join(","));
        out.push_str(&format!("],\"data_offsets\":[{},{}]}}", t.begin, t.end));
    }
    out.push('}');
    let digest_at =
        digest_at.ok_or_else(|| format!("internal: the header has no {TENSOR_SHA256_KEY} slot"))?;
    let mut bytes = out.into_bytes();
    let padded = bytes.len().next_multiple_of(HEADER_ALIGN);
    bytes.resize(padded, b' ');
    Ok((bytes, digest_at))
}

/// Attach the outcome of removing `tmp` to `primary`.
///
/// A temporary left behind is reported with the error that caused it
/// rather than dropped: the caller sees both what failed and what is
/// still on disk because of it.
fn with_cleanup(primary: String, tmp: &Path) -> String {
    match fs::remove_file(tmp) {
        Ok(()) => primary,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => primary,
        Err(e) => format!(
            "{primary}; the temporary {} could not be removed: {e}",
            tmp.display()
        ),
    }
}

/// Write `src`'s tensors to `dst` under a new header whose
/// `__metadata__` is `metadata` plus [`TENSOR_SHA256_KEY`], and return
/// that digest.
///
/// - `src`'s own `__metadata__`, if any, is dropped: the map given here
///   replaces it wholly.
/// - The tensor data section is copied byte for byte, and hashed while
///   it is copied, so the digest is of the bytes that were written.
/// - The header layout is fixed (see `serialize_header`), so two
///   rewrites of one `src` with one map produce byte-identical files.
/// - `dst` is written through a temporary file in its own directory and
///   published with [`publish_new`], so a reader never sees half a file
///   and a file already at `dst` — including one that appears while the
///   rewrite runs — is never replaced.
///
/// # Errors
///
/// `dst` already existing (checked up front, and again atomically at
/// publish), a link failure on a filesystem without hard links, a caller
/// map that already holds [`TENSOR_SHA256_KEY`] (the value is computed,
/// not given), a `src` that is not a safetensors file — including one
/// whose tensors do not tile its data section exactly, which a reader
/// would refuse — and any I/O failure.
pub fn rewrite_with_metadata(
    src: &Path,
    dst: &Path,
    metadata: &BTreeMap<String, String>,
) -> Result<String, String> {
    if metadata.contains_key(TENSOR_SHA256_KEY) {
        return Err(format!(
            "{TENSOR_SHA256_KEY} is computed from the data section and cannot be supplied"
        ));
    }
    match fs::symlink_metadata(dst) {
        Ok(_) => {
            return Err(format!(
                "{} already exists; a rewrite does not replace a file",
                dst.display()
            ))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("stat {}: {e}", dst.display())),
    }

    let (header_len, header) = read_header_json(src)?;
    let Value::Object(entries) = header else {
        return Err(format!(
            "{}: the header is not a JSON object",
            src.display()
        ));
    };
    let mut tensors = entries
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
        .map(|(name, value)| TensorEntry::parse(src, name, value))
        .collect::<Result<Vec<_>, _>>()?;
    // Offset order is the file's own order. Ties are zero-length
    // tensors sharing an offset, broken by name so the order is total.
    tensors
        .sort_by(|a, b| (a.begin, a.end, a.name.as_str()).cmp(&(b.begin, b.end, b.name.as_str())));

    let data_start = LEN_PREFIX
        .checked_add(header_len)
        .ok_or_else(|| format!("{}: header length overflows", src.display()))?;
    let file_len = fs::metadata(src)
        .map_err(|e| format!("stat {}: {e}", src.display()))?
        .len();
    let data_len = file_len.checked_sub(data_start).ok_or_else(|| {
        format!(
            "{}: the file ends inside its own header ({file_len} bytes, header ends at {data_start})",
            src.display()
        )
    })?;
    // The tensors have to tile the data section with no gap and no
    // overlap, which is what a safetensors reader checks. A file that
    // fails it would be copied into another file that fails it.
    let mut cursor = 0u64;
    for t in &tensors {
        if t.begin != cursor {
            return Err(format!(
                "{}: tensor {:?} starts at {} where the previous one ended at {cursor}",
                src.display(),
                t.name,
                t.begin
            ));
        }
        cursor = t.end;
    }
    if cursor != data_len {
        return Err(format!(
            "{}: the tensors cover {cursor} bytes of a {data_len}-byte data section",
            src.display()
        ));
    }

    let mut full = metadata.clone();
    full.insert(TENSOR_SHA256_KEY.to_string(), "0".repeat(DIGEST_HEX_LEN));
    let (header_bytes, digest_at) = serialize_header(&full, &tensors)?;

    let tmp = temp_path_for(dst)?;
    // `create_new`: a file already at this name is not ours to remove,
    // so a failure here returns before anything would clean up after it.
    let out = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| format!("create {}: {e}", tmp.display()))?;
    let digest = match write_rewritten(
        out,
        &tmp,
        src,
        data_start,
        data_len,
        &header_bytes,
        digest_at,
    ) {
        Ok(digest) => digest,
        Err(e) => return Err(with_cleanup(e, &tmp)),
    };
    publish_new(&tmp, dst)?;
    Ok(digest)
}

/// A temporary name for `dst`, in `dst`'s own directory (so publishing
/// it with [`publish_new`] never crosses a filesystem): a dot-file
/// `.<name>.tmp-<pid>`.
///
/// # Errors
///
/// A `dst` that names no file.
pub fn temp_path_for(dst: &Path) -> Result<PathBuf, String> {
    let parent = match dst.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let file_name = dst
        .file_name()
        .ok_or_else(|| format!("{} names no file", dst.display()))?
        .to_string_lossy()
        .into_owned();
    Ok(parent.join(format!(".{file_name}.tmp-{}", std::process::id())))
}

/// Publish the finished file `tmp` at `dst`, refusing — atomically —
/// if anything is already at `dst`, then remove `tmp`.
///
/// A hard link rather than a rename: `rename` replaces whatever `dst`
/// names, so a check for an existing file followed by a rename leaves a
/// window in which a file that appeared in between is overwritten.
/// `link` fails with `AlreadyExists` instead, in one step. The cost is
/// that the directory has to be on a filesystem with hard links; on one
/// without, the link fails and nothing is published.
///
/// `tmp` is removed on every path. If it cannot be removed after the
/// link succeeded, the file just published at `dst` is withdrawn again
/// (it is this call's own inode) and the error says so, so the caller
/// is never told "failed" about a file that is in fact there.
///
/// # Errors
///
/// `dst` already existing, a link failure, and a failure to remove
/// `tmp`, each with whatever was or was not cleaned up after it.
pub fn publish_new(tmp: &Path, dst: &Path) -> Result<(), String> {
    if let Err(e) = fs::hard_link(tmp, dst) {
        let primary = if e.kind() == std::io::ErrorKind::AlreadyExists {
            format!("{} already exists; the file is not replaced", dst.display())
        } else {
            format!("link {} -> {}: {e}", tmp.display(), dst.display())
        };
        return Err(with_cleanup(primary, tmp));
    }
    if let Err(e) = fs::remove_file(tmp) {
        let withdrawn = match fs::remove_file(dst) {
            Ok(()) => "the published file was withdrawn".to_string(),
            Err(r) => format!(
                "the published file {} could not be withdrawn either: {r}",
                dst.display()
            ),
        };
        return Err(format!(
            "remove temporary {} after publishing {}: {e}; {withdrawn}",
            tmp.display(),
            dst.display()
        ));
    }
    Ok(())
}

/// Write the length prefix, `header_bytes` and `src`'s data section to
/// `out`, hashing the data as it goes, then patch the digest into the
/// header at `digest_at`. Returns the digest.
fn write_rewritten(
    mut out: fs::File,
    tmp: &Path,
    src: &Path,
    data_start: u64,
    data_len: u64,
    header_bytes: &[u8],
    digest_at: usize,
) -> Result<String, String> {
    let write_err = |e: std::io::Error| format!("write {}: {e}", tmp.display());
    let read_err = |e: std::io::Error| format!("read {}: {e}", src.display());

    out.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .map_err(write_err)?;
    out.write_all(header_bytes).map_err(write_err)?;

    let mut input = fs::File::open(src).map_err(read_err)?;
    input.seek(SeekFrom::Start(data_start)).map_err(read_err)?;
    let mut hasher = Sha256::new();
    let mut remaining = data_len;
    let mut buf = vec![0u8; 1 << 20];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        let got = input.read(&mut buf[..want]).map_err(read_err)?;
        if got == 0 {
            return Err(format!(
                "{}: the data section ended {remaining} bytes early; the file changed while \
                 it was being read",
                src.display()
            ));
        }
        hasher.update(&buf[..got]);
        out.write_all(&buf[..got]).map_err(write_err)?;
        remaining -= got as u64;
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    out.seek(SeekFrom::Start(LEN_PREFIX + digest_at as u64))
        .map_err(write_err)?;
    out.write_all(digest.as_bytes()).map_err(write_err)?;
    out.sync_all().map_err(write_err)?;
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::read_bundle_header;
    use candle_core::{DType, Device, Tensor};
    use tempfile::TempDir;

    /// A small bundle with tensors of two dtypes, so the data section
    /// holds more than one run of one element size.
    fn write_src(dir: &Path, metadata: Option<BTreeMap<String, String>>) -> PathBuf {
        let a = Tensor::arange(0f32, 6f32, &Device::Cpu)
            .unwrap()
            .reshape((2, 3))
            .unwrap();
        let b = Tensor::new(&[7u32, 8, 9], &Device::Cpu).unwrap();
        let c = Tensor::new(&[[0.5f32], [1.5]], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let path = dir.join("src.safetensors");
        let tensors: BTreeMap<String, Tensor> = [
            ("layer.a".to_string(), a),
            ("layer.b".to_string(), b),
            ("head.c".to_string(), c),
        ]
        .into_iter()
        .collect();
        safetensors::serialize_to_file(
            tensors.iter().map(|(k, v)| (k.clone(), v)),
            metadata.map(|m| m.into_iter().collect()),
            &path,
        )
        .unwrap();
        path
    }

    fn sample_map() -> BTreeMap<String, String> {
        [
            ("format", "pt"),
            ("alc.schema", "1"),
            ("alc.card_id", "demo_1"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    /// The SHA-256 of everything after the header, computed here from
    /// the file's own bytes rather than through the function under test.
    fn data_section_digest(path: &Path) -> String {
        let bytes = fs::read(path).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        Sha256::digest(&bytes[8 + n..])
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    #[test]
    fn the_digest_is_the_sha256_of_the_written_data_section() {
        let tmp = TempDir::new().unwrap();
        let src = write_src(tmp.path(), None);
        let dst = tmp.path().join("dst.safetensors");
        let digest = rewrite_with_metadata(&src, &dst, &sample_map()).expect("rewrite");
        assert_eq!(digest.len(), 64);
        assert!(digest
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
        assert_eq!(digest, data_section_digest(&dst));
        // The data section is the source's, byte for byte.
        assert_eq!(data_section_digest(&src), digest);
    }

    #[test]
    fn the_metadata_reads_back_as_given_plus_the_digest() {
        let tmp = TempDir::new().unwrap();
        let src = write_src(tmp.path(), None);
        let dst = tmp.path().join("dst.safetensors");
        let digest = rewrite_with_metadata(&src, &dst, &sample_map()).unwrap();
        let header = read_bundle_header(&dst).unwrap().expect("a header");
        let mut expected = sample_map();
        expected.insert(TENSOR_SHA256_KEY.into(), digest);
        assert_eq!(header, expected);
    }

    #[test]
    fn the_tensors_load_back_equal_to_the_source() {
        let tmp = TempDir::new().unwrap();
        let src = write_src(tmp.path(), None);
        let dst = tmp.path().join("dst.safetensors");
        rewrite_with_metadata(&src, &dst, &sample_map()).unwrap();

        let before = candle_core::safetensors::load(&src, &Device::Cpu).unwrap();
        let after = candle_core::safetensors::load(&dst, &Device::Cpu).unwrap();
        assert_eq!(before.len(), after.len());
        for (name, t) in &before {
            let u = &after[name];
            assert_eq!(t.dims(), u.dims(), "{name}");
            assert_eq!(t.dtype(), u.dtype(), "{name}");
            let a = t.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
            let b = u.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
            assert_eq!(
                a.to_vec1::<f64>().unwrap(),
                b.to_vec1::<f64>().unwrap(),
                "{name}"
            );
        }
        // And the mmap reader, which validates offsets against the file.
        let mm = unsafe { candle_core::safetensors::MmapedSafetensors::new(&dst) }.unwrap();
        assert_eq!(mm.tensors().len(), 3);
    }

    #[test]
    fn two_rewrites_of_one_source_are_byte_identical() {
        let tmp = TempDir::new().unwrap();
        let src = write_src(tmp.path(), None);
        let one = tmp.path().join("one.safetensors");
        let two = tmp.path().join("two.safetensors");
        rewrite_with_metadata(&src, &one, &sample_map()).unwrap();
        rewrite_with_metadata(&src, &two, &sample_map()).unwrap();
        assert_eq!(fs::read(&one).unwrap(), fs::read(&two).unwrap());
    }

    #[test]
    fn an_existing_metadata_block_is_replaced_not_merged() {
        let tmp = TempDir::new().unwrap();
        let old: BTreeMap<String, String> = [("format", "pt"), ("stale", "yes"), ("run", "x")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let src = write_src(tmp.path(), Some(old));
        let dst = tmp.path().join("dst.safetensors");
        let digest = rewrite_with_metadata(&src, &dst, &sample_map()).unwrap();
        let header = read_bundle_header(&dst).unwrap().unwrap();
        assert!(!header.contains_key("stale"), "{header:?}");
        assert!(!header.contains_key("run"), "{header:?}");
        let mut expected = sample_map();
        expected.insert(TENSOR_SHA256_KEY.into(), digest);
        assert_eq!(header, expected);
    }

    #[test]
    fn the_header_layout_is_sorted_and_aligned() {
        let tmp = TempDir::new().unwrap();
        let src = write_src(tmp.path(), None);
        let dst = tmp.path().join("dst.safetensors");
        rewrite_with_metadata(&src, &dst, &sample_map()).unwrap();
        let bytes = fs::read(&dst).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(n % 8, 0, "header length is 8-aligned");
        let text = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
        assert!(
            text.starts_with("{\"__metadata__\":{\"alc.card_id\":"),
            "{text}"
        );
        let meta_end = text.find('}').unwrap();
        let keys = &text[..meta_end];
        let pos = |k: &str| keys.find(&format!("\"{k}\"")).unwrap();
        assert!(pos("alc.card_id") < pos("alc.schema"));
        assert!(pos("alc.schema") < pos("alc.tensor_sha256"));
        assert!(pos("alc.tensor_sha256") < pos("format"));
        assert!(text.trim_end().ends_with('}'));
    }

    #[test]
    fn a_rewrite_refuses_to_replace_an_existing_file() {
        let tmp = TempDir::new().unwrap();
        let src = write_src(tmp.path(), None);
        let dst = tmp.path().join("dst.safetensors");
        fs::write(&dst, b"keep me").unwrap();
        let err = rewrite_with_metadata(&src, &dst, &sample_map()).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(fs::read(&dst).unwrap(), b"keep me");
    }

    /// The publish step itself refuses an existing `dst`, so a file that
    /// appears after the up-front check is still not replaced.
    #[test]
    fn publishing_onto_an_existing_file_fails_and_leaves_it_untouched() {
        let tmp = TempDir::new().unwrap();
        let dst = tmp.path().join("dst.bin");
        let staged = temp_path_for(&dst).unwrap();
        fs::write(&staged, b"new bytes").unwrap();
        fs::write(&dst, b"someone else's").unwrap();

        let err = publish_new(&staged, &dst).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(fs::read(&dst).unwrap(), b"someone else's");
        assert!(!staged.exists(), "the temporary is removed on failure");

        // And onto a free name it publishes and cleans up.
        let free = tmp.path().join("free.bin");
        let staged = temp_path_for(&free).unwrap();
        fs::write(&staged, b"new bytes").unwrap();
        publish_new(&staged, &free).unwrap();
        assert_eq!(fs::read(&free).unwrap(), b"new bytes");
        assert!(!staged.exists());
    }

    #[test]
    fn a_supplied_digest_is_refused() {
        let tmp = TempDir::new().unwrap();
        let src = write_src(tmp.path(), None);
        let mut map = sample_map();
        map.insert(TENSOR_SHA256_KEY.into(), "f".repeat(64));
        let err =
            rewrite_with_metadata(&src, &tmp.path().join("dst.safetensors"), &map).unwrap_err();
        assert!(err.contains("computed"), "{err}");
    }

    #[test]
    fn a_truncated_source_is_refused_and_leaves_nothing_behind() {
        let tmp = TempDir::new().unwrap();
        let src = write_src(tmp.path(), None);
        let bytes = fs::read(&src).unwrap();
        let cut = tmp.path().join("cut.safetensors");
        fs::write(&cut, &bytes[..bytes.len() - 4]).unwrap();
        let dst = tmp.path().join("dst.safetensors");
        let err = rewrite_with_metadata(&cut, &dst, &sample_map()).unwrap_err();
        assert!(err.contains("data section"), "{err}");
        assert!(!dst.exists());
        let leftovers: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }
}
