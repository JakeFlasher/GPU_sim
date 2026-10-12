use std::borrow::Cow;
use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path};
use std::process::ExitCode;

use sim_run::{RunError, run_manifest, run_manifest_retained_fds};

const RETAINED_MANIFEST_FDS_PREFIX: &str = "--retained-manifest-fds-v1:";
const RETAINED_MANIFEST_FDS_NAMESPACE: &str = "--retained-manifest-fds";

#[derive(Debug)]
enum FramedWriteError {
    Write(io::Error),
    Flush(io::Error),
}

impl std::fmt::Display for FramedWriteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Write(source) => write!(formatter, "write_all failed: {source}"),
            Self::Flush(source) => write!(formatter, "flush failed: {source}"),
        }
    }
}

fn framed_document(document: &str) -> Vec<u8> {
    let mut framed = Vec::with_capacity(document.len().saturating_add(1));
    framed.extend_from_slice(document.as_bytes());
    framed.push(b'\n');
    framed
}

fn write_framed<W: Write>(writer: &mut W, framed: &[u8]) -> Result<(), FramedWriteError> {
    writer.write_all(framed).map_err(FramedWriteError::Write)?;
    writer.flush().map_err(FramedWriteError::Flush)
}

fn write_success_document_to<W: Write>(document: &str, stdout: &mut W) -> Result<(), RunError> {
    let framed = framed_document(document);
    write_framed(stdout, &framed).map_err(|source| {
        RunError::output_write(format!(
            "could not emit canonical result to stdout: {source}"
        ))
    })
}

fn write_error_to<W: Write>(error: &RunError, stderr: &mut W) -> u8 {
    let (document, exit_code): (Cow<'_, str>, u8) = match error.canonical_json() {
        Ok(document) => (Cow::Owned(document), error.exit_code()),
        Err(_encoding_error) => (
            Cow::Borrowed(RunError::canonical_serialization_fallback()),
            1,
        ),
    };
    let framed = framed_document(&document);
    // This is the terminal output attempt. A stderr write or flush failure must
    // neither recurse through error serialization nor trigger a stdout write.
    let _terminal_write = write_framed(stderr, &framed);
    exit_code
}

fn write_error(error: &RunError) -> ExitCode {
    let mut stderr = std::io::stderr().lock();
    ExitCode::from(write_error_to(error, &mut stderr))
}

#[derive(Debug, PartialEq, Eq)]
enum Invocation {
    Path(OsString),
    Retained {
        manifest_fd: u32,
        evidence_base_fd: u32,
        manifest_name: OsString,
    },
}

fn cli_usage(detail: impl std::fmt::Display, provided: u32) -> RunError {
    RunError::cli_usage(
        format!(
            "{detail}; expected exactly one sim_run_manifest.v2 path token or one canonical \
             {RETAINED_MANIFEST_FDS_PREFIX}MANIFEST_FD:EVIDENCE_BASE_FD:LOWERCASE_HEX_UTF8_MANIFEST_NAME token"
        ),
        provided,
    )
}

fn canonical_fd(argument: &OsStr, role: &'static str, provided: u32) -> Result<u32, RunError> {
    let Some(raw) = argument.to_str() else {
        return Err(cli_usage(
            format!("{role} must be canonical ASCII decimal"),
            provided,
        ));
    };
    let parsed = raw.parse::<i32>().map_err(|_| {
        cli_usage(
            format!("{role} must be a canonical i32 decimal descriptor at least 3"),
            provided,
        )
    })?;
    if parsed < 3 || parsed.to_string() != raw {
        return Err(cli_usage(
            format!("{role} must be a canonical i32 decimal descriptor at least 3"),
            provided,
        ));
    }
    u32::try_from(parsed).map_err(|_| {
        cli_usage(
            format!("{role} must be a canonical i32 decimal descriptor at least 3"),
            provided,
        )
    })
}

fn lowercase_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn decode_manifest_name(raw: &str, provided: u32) -> Result<OsString, RunError> {
    if raw.is_empty() || !raw.len().is_multiple_of(2) {
        return Err(cli_usage(
            "MANIFEST_NAME must be nonempty even-length lowercase hexadecimal UTF-8",
            provided,
        ));
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(raw.len() / 2).map_err(|_| {
        cli_usage(
            "could not reserve the bounded MANIFEST_NAME decode buffer",
            provided,
        )
    })?;
    for pair in raw.as_bytes().chunks_exact(2) {
        let high = lowercase_hex_nibble(pair[0]).ok_or_else(|| {
            cli_usage(
                "MANIFEST_NAME must use lowercase hexadecimal only",
                provided,
            )
        })?;
        let low = lowercase_hex_nibble(pair[1]).ok_or_else(|| {
            cli_usage(
                "MANIFEST_NAME must use lowercase hexadecimal only",
                provided,
            )
        })?;
        bytes.push((high << 4) | low);
    }
    let name = String::from_utf8(bytes)
        .map_err(|_| cli_usage("MANIFEST_NAME hexadecimal must decode as UTF-8", provided))?;
    let path = Path::new(&name);
    let mut components = path.components();
    if name.contains('\0')
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || path.file_name() != Some(OsStr::new(&name))
    {
        return Err(cli_usage(
            "MANIFEST_NAME must decode to exactly one normal file-name component",
            provided,
        ));
    }
    Ok(OsString::from(name))
}

fn parse_retained_token(raw: &str, provided: u32) -> Result<Invocation, RunError> {
    let payload = raw
        .strip_prefix(RETAINED_MANIFEST_FDS_PREFIX)
        .ok_or_else(|| {
            cli_usage(
                "unknown or non-versioned retained-manifest transport token",
                provided,
            )
        })?;
    let mut fields = payload.split(':');
    let manifest_fd = fields.next().unwrap_or_default();
    let evidence_base_fd = fields.next().unwrap_or_default();
    let manifest_name = fields.next().unwrap_or_default();
    if fields.next().is_some() {
        return Err(cli_usage(
            "retained-manifest transport token has an unexpected field",
            provided,
        ));
    }
    let manifest_fd = canonical_fd(OsStr::new(manifest_fd), "MANIFEST_FD", provided)?;
    let evidence_base_fd =
        canonical_fd(OsStr::new(evidence_base_fd), "EVIDENCE_BASE_FD", provided)?;
    if manifest_fd == evidence_base_fd {
        return Err(cli_usage(
            "MANIFEST_FD and EVIDENCE_BASE_FD must be distinct",
            provided,
        ));
    }
    Ok(Invocation::Retained {
        manifest_fd,
        evidence_base_fd,
        manifest_name: decode_manifest_name(manifest_name, provided)?,
    })
}

fn parse_invocation<I>(args: I) -> Result<Invocation, RunError>
where
    I: IntoIterator<Item = OsString>,
{
    let arguments = args.into_iter().collect::<Vec<_>>();
    let provided = u32::try_from(arguments.len()).unwrap_or(u32::MAX);
    match arguments.as_slice() {
        [token]
            if token
                .as_os_str()
                .as_bytes()
                .starts_with(RETAINED_MANIFEST_FDS_NAMESPACE.as_bytes()) =>
        {
            let raw = token.to_str().ok_or_else(|| {
                cli_usage(
                    "retained-manifest transport token must be valid ASCII",
                    provided,
                )
            })?;
            parse_retained_token(raw, provided)
        }
        [token] => Ok(Invocation::Path(token.clone())),
        _ => Err(cli_usage(
            format!("received {provided} post-argv0 tokens"),
            provided,
        )),
    }
}

fn run() -> Result<(), RunError> {
    let output = match parse_invocation(std::env::args_os().skip(1))? {
        Invocation::Path(manifest) => {
            // An empty path is still one positional argument. Manifest admission,
            // not CLI arity handling, owns rejection of that path spelling.
            run_manifest(manifest)?
        }
        Invocation::Retained {
            manifest_fd,
            evidence_base_fd,
            manifest_name,
        } => run_manifest_retained_fds(manifest_fd, evidence_base_fd, manifest_name)?,
    };
    let mut stdout = std::io::stdout().lock();
    write_success_document_to(output.canonical_json(), &mut stdout)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => write_error(&error),
    }
}
