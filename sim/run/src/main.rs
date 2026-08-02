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

#[cfg(test)]
mod tests {
    use std::os::unix::ffi::OsStringExt;

    use super::*;

    #[derive(Default)]
    struct ScriptedWriter {
        bytes: Vec<u8>,
        fail_write: bool,
        fail_flush: bool,
        max_write_size: Option<usize>,
        write_calls: usize,
        flush_calls: usize,
    }

    impl Write for ScriptedWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.write_calls += 1;
            if self.fail_write {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "scripted write"));
            }
            let accepted = self
                .max_write_size
                .unwrap_or(buffer.len())
                .min(buffer.len());
            self.bytes.extend_from_slice(&buffer[..accepted]);
            Ok(accepted)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flush_calls += 1;
            if self.fail_flush {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "scripted flush"));
            }
            Ok(())
        }
    }

    fn retained_token(manifest_fd: &str, base_fd: &str, name: &str) -> OsString {
        let name_hex = name
            .as_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        OsString::from(format!(
            "{RETAINED_MANIFEST_FDS_PREFIX}{manifest_fd}:{base_fd}:{name_hex}"
        ))
    }

    #[test]
    fn invocation_parser_preserves_one_token_path_and_retained_modes() {
        assert_eq!(
            parse_invocation([OsString::from("")]).expect("one empty path argument"),
            Invocation::Path(OsString::from(""))
        );
        assert_eq!(
            parse_invocation([OsString::from("manifest.json")]).expect("one path argument"),
            Invocation::Path(OsString::from("manifest.json"))
        );
        assert_eq!(
            parse_invocation([retained_token("7", "8", "manifest-é.json")])
                .expect("exact retained token"),
            Invocation::Retained {
                manifest_fd: 7,
                evidence_base_fd: 8,
                manifest_name: OsString::from("manifest-é.json"),
            }
        );

        for invalid in [
            Vec::<OsString>::new(),
            vec![OsString::from(RETAINED_MANIFEST_FDS_NAMESPACE)],
            vec![OsString::from("--retained-manifest-fds-v1")],
            vec![OsString::from("--retained-manifest-fds-v2:7:8:6e616d65")],
            vec![
                OsString::from("manifest.json"),
                OsString::from(RETAINED_MANIFEST_FDS_NAMESPACE),
            ],
            vec![
                OsString::from(RETAINED_MANIFEST_FDS_NAMESPACE),
                OsString::from("7"),
                OsString::from("8"),
                OsString::from("manifest.json"),
            ],
        ] {
            let error =
                parse_invocation(invalid).expect_err("mixed, partial, old, and unknown forms fail");
            assert_eq!(error.failure_type(), "CLI_USAGE_FAILED");
        }
    }

    #[test]
    fn retained_token_rejects_noncanonical_fd_and_name_encodings() {
        for invalid in [
            "",
            " 7",
            "7 ",
            "+7",
            "-1",
            "0",
            "1",
            "2",
            "07",
            "7\n",
            "１２",
            "2147483648",
        ] {
            let error = parse_invocation([retained_token(invalid, "8", "manifest.json")])
                .expect_err("noncanonical manifest fd fails");
            assert_eq!(error.failure_type(), "CLI_USAGE_FAILED");
            assert_eq!(error.exit_code(), 2);
        }

        for invalid_token in [
            format!("{RETAINED_MANIFEST_FDS_PREFIX}7:8:"),
            format!("{RETAINED_MANIFEST_FDS_PREFIX}7:8:0"),
            format!("{RETAINED_MANIFEST_FDS_PREFIX}7:8:GG"),
            format!("{RETAINED_MANIFEST_FDS_PREFIX}7:8:FF"),
            format!("{RETAINED_MANIFEST_FDS_PREFIX}7:8:ff"),
            format!("{RETAINED_MANIFEST_FDS_PREFIX}7:8:2e2e2f6e616d65"),
            format!("{RETAINED_MANIFEST_FDS_PREFIX}7:8:6e616d65:00"),
        ] {
            let error = parse_invocation([OsString::from(invalid_token)])
                .expect_err("noncanonical name encoding fails");
            assert_eq!(error.failure_type(), "CLI_USAGE_FAILED");
            assert_eq!(error.exit_code(), 2);
        }

        let duplicate = parse_invocation([retained_token("7", "7", "manifest.json")])
            .expect_err("descriptor roles must use distinct inherited descriptors");
        assert_eq!(duplicate.failure_type(), "CLI_USAGE_FAILED");

        let mut non_utf8 = RETAINED_MANIFEST_FDS_PREFIX.as_bytes().to_vec();
        non_utf8.extend_from_slice(b"7:8:");
        non_utf8.push(0xff);
        let error = parse_invocation([OsString::from_vec(non_utf8)])
            .expect_err("reserved transport namespace must fail closed on non-UTF-8");
        assert_eq!(error.failure_type(), "CLI_USAGE_FAILED");
    }

    #[test]
    fn usage_errors_keep_the_pinned_one_token_positional_evidence() {
        let zero = parse_invocation(Vec::<OsString>::new()).expect_err("zero arguments");
        assert_eq!(zero.failure_type(), "CLI_USAGE_FAILED");
        let zero_document: serde_json::Value =
            serde_json::from_str(&zero.canonical_json().expect("encoded error")).unwrap();
        assert_eq!(
            zero_document["error"]["evidence"]["provided_positional_argument_count"],
            0
        );
        assert_eq!(
            zero_document["error"]["evidence"]["expected_positional_argument_count"],
            1
        );

        let three = parse_invocation([
            OsString::from("first"),
            OsString::from("second"),
            OsString::from("third"),
        ])
        .expect_err("three arguments");
        let three_document: serde_json::Value =
            serde_json::from_str(&three.canonical_json().expect("encoded error")).unwrap();
        assert_eq!(
            three_document["error"]["evidence"]["provided_positional_argument_count"],
            3
        );
        assert_eq!(
            three_document["error"]["evidence"]["expected_positional_argument_count"],
            1
        );
    }

    #[test]
    fn framed_writer_uses_write_all_then_flush() {
        let frame = framed_document("{\"ok\":true}");
        let mut writer = ScriptedWriter::default();
        write_framed(&mut writer, &frame).expect("write succeeds");
        assert_eq!(writer.bytes, b"{\"ok\":true}\n");
        assert_eq!(writer.write_calls, 1);
        assert_eq!(writer.flush_calls, 1);
    }

    #[test]
    fn framed_writer_retries_short_writes_before_flushing() {
        let frame = framed_document("{\"short\":true}");
        let mut writer = ScriptedWriter {
            max_write_size: Some(3),
            ..ScriptedWriter::default()
        };
        write_framed(&mut writer, &frame).expect("short writes complete");
        assert_eq!(writer.bytes, frame);
        assert!(writer.write_calls > 1);
        assert_eq!(writer.flush_calls, 1);
    }

    #[test]
    fn framed_writer_reports_write_and_flush_failures_separately() {
        let frame = framed_document("{}");
        let mut write_failure = ScriptedWriter {
            fail_write: true,
            ..ScriptedWriter::default()
        };
        assert!(matches!(
            write_framed(&mut write_failure, &frame),
            Err(FramedWriteError::Write(_))
        ));
        assert_eq!(write_failure.write_calls, 1);
        assert_eq!(write_failure.flush_calls, 0);

        let mut flush_failure = ScriptedWriter {
            fail_flush: true,
            ..ScriptedWriter::default()
        };
        assert!(matches!(
            write_framed(&mut flush_failure, &frame),
            Err(FramedWriteError::Flush(_))
        ));
        assert_eq!(flush_failure.write_calls, 1);
        assert_eq!(flush_failure.flush_calls, 1);
    }

    #[test]
    fn success_flush_failure_becomes_one_sealed_output_write_error() {
        let mut stdout = ScriptedWriter {
            fail_flush: true,
            ..ScriptedWriter::default()
        };
        let error =
            write_success_document_to("{\"schema_version\":\"sim_result.v2\"}", &mut stdout)
                .expect_err("success-channel flush must fail");
        assert_eq!(error.failure_type(), "OUTPUT_WRITE_FAILED");
        assert_eq!(error.stage(), "output_write");
        assert_eq!(error.exit_code(), 1);
        assert_eq!(stdout.write_calls, 1);
        assert_eq!(stdout.flush_calls, 1);

        let mut stderr = ScriptedWriter::default();
        assert_eq!(write_error_to(&error, &mut stderr), 1);
        assert_eq!(stderr.write_calls, 1);
        assert_eq!(stderr.flush_calls, 1);
        assert_eq!(stderr.bytes.last(), Some(&b'\n'));
        let document = &stderr.bytes[..stderr.bytes.len() - 1];
        assert!(!document.contains(&b'\n'));
        let mut parsed: serde_json::Value = serde_json::from_slice(document).unwrap();
        assert_eq!(parsed["error"]["failure_type"], "OUTPUT_WRITE_FAILED");
        assert_eq!(parsed["error"]["stage"], "output_write");
        assert_eq!(parsed["exit_status"]["exit_code"], 1);
        assert_eq!(
            sim_trace::canonical_json_utf8(&parsed).unwrap().as_bytes(),
            document
        );
        let seal = parsed
            .as_object_mut()
            .unwrap()
            .remove("error_sha256")
            .unwrap();
        let unsigned = sim_trace::canonical_json_utf8(&parsed).unwrap();
        assert_eq!(
            seal,
            sim_trace::Sha256Digest::of(unsigned.as_bytes()).to_string()
        );
    }

    #[test]
    fn terminal_error_write_does_not_recurse_after_channel_failure() {
        let error = RunError::output_write("stdout closed");
        let mut writer = ScriptedWriter {
            fail_write: true,
            ..ScriptedWriter::default()
        };
        assert_eq!(write_error_to(&error, &mut writer), error.exit_code());
        assert_eq!(writer.write_calls, 1);
        assert_eq!(writer.flush_calls, 0);

        let mut flush_failure = ScriptedWriter {
            fail_flush: true,
            ..ScriptedWriter::default()
        };
        assert_eq!(
            write_error_to(&error, &mut flush_failure),
            error.exit_code()
        );
        assert_eq!(flush_failure.write_calls, 1);
        assert_eq!(flush_failure.flush_calls, 1);
    }
}
