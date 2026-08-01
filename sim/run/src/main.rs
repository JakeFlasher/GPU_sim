use std::borrow::Cow;
use std::ffi::OsString;
use std::io::{self, Write};
use std::process::ExitCode;

use sim_run::{RunError, run_manifest};

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

fn one_manifest_argument<I>(args: I) -> Result<OsString, RunError>
where
    I: IntoIterator<Item = OsString>,
{
    let mut provided = 0_u32;
    let mut manifest = None;
    for argument in args {
        provided = provided
            .checked_add(1)
            .expect("the operating-system argument vector fits in u32");
        if provided == 1 {
            manifest = Some(argument);
        }
    }
    if provided != 1 {
        return Err(RunError::cli_usage(
            format!("expected exactly one sim_run_manifest.v2 path argument; received {provided}"),
            provided,
        ));
    }
    Ok(manifest.expect("one counted positional argument is retained"))
}

fn run() -> Result<(), RunError> {
    let manifest = one_manifest_argument(std::env::args_os().skip(1))?;
    // An empty path is still one positional argument. Manifest admission, not
    // CLI arity handling, owns rejection of that path spelling.
    let output = run_manifest(manifest)?;
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

    #[test]
    fn positional_count_consumes_all_arguments_and_keeps_empty_as_one() {
        assert_eq!(
            one_manifest_argument([OsString::from("")]).expect("one empty argument"),
            OsString::from("")
        );

        let zero = one_manifest_argument(Vec::<OsString>::new()).expect_err("zero arguments");
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

        let three = one_manifest_argument([
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
