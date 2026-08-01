//! SHA-256 and Python-compatible canonical JSON used by sealed simulator inputs.
//!
//! The canonical spelling matches `json.dumps(value, sort_keys=True,
//! separators=(",", ":"))` for the finite JSON domain admitted by the
//! simulator contracts.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

const SHA256_INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];
const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// A raw SHA-256 digest with one canonical lowercase-hex wire spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    pub fn parse(value: &str) -> Result<Self, Sha256DigestError> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(Sha256DigestError {
                value: value.to_owned(),
            });
        }
        let mut bytes = [0_u8; 32];
        for (index, slot) in bytes.iter_mut().enumerate() {
            let at = index * 2;
            *slot = u8::from_str_radix(&value[at..at + 2], 16).map_err(|_| Sha256DigestError {
                value: value.to_owned(),
            })?;
        }
        Ok(Self(bytes))
    }

    pub fn of(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.finish_digest()
    }

    pub fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(de::Error::custom)
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[error("not a canonical lowercase SHA-256 digest: {value:?}")]
pub struct Sha256DigestError {
    value: String,
}

fn sha256_compress(state: &mut [u32; 8], chunk: &[u8]) {
    debug_assert_eq!(chunk.len(), 64);
    let mut schedule = [0u32; 64];
    for (index, word) in schedule[..16].iter_mut().enumerate() {
        let offset = index * 4;
        *word = u32::from_be_bytes([
            chunk[offset],
            chunk[offset + 1],
            chunk[offset + 2],
            chunk[offset + 3],
        ]);
    }
    for index in 16..64 {
        let s0 = schedule[index - 15].rotate_right(7)
            ^ schedule[index - 15].rotate_right(18)
            ^ (schedule[index - 15] >> 3);
        let s1 = schedule[index - 2].rotate_right(17)
            ^ schedule[index - 2].rotate_right(19)
            ^ (schedule[index - 2] >> 10);
        schedule[index] = schedule[index - 16]
            .wrapping_add(s0)
            .wrapping_add(schedule[index - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for index in 0..64 {
        let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let choose = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(sum1)
            .wrapping_add(choose)
            .wrapping_add(SHA256_K[index])
            .wrapping_add(schedule[index]);
        let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = sum0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h].into_iter()) {
        *slot = slot.wrapping_add(value);
    }
}

pub(crate) struct Sha256 {
    state: [u32; 8],
    tail: [u8; 64],
    tail_len: usize,
    total_bytes: u64,
}

impl Sha256 {
    pub(crate) fn new() -> Self {
        Self {
            state: SHA256_INITIAL,
            tail: [0; 64],
            tail_len: 0,
            total_bytes: 0,
        }
    }

    pub(crate) fn update(&mut self, mut input: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(input.len() as u64);
        if self.tail_len > 0 {
            let copied = (64 - self.tail_len).min(input.len());
            self.tail[self.tail_len..self.tail_len + copied].copy_from_slice(&input[..copied]);
            self.tail_len += copied;
            input = &input[copied..];
            if self.tail_len < 64 {
                return;
            }
            sha256_compress(&mut self.state, &self.tail);
            self.tail_len = 0;
        }
        let mut chunks = input.chunks_exact(64);
        for chunk in &mut chunks {
            sha256_compress(&mut self.state, chunk);
        }
        let remainder = chunks.remainder();
        self.tail[..remainder.len()].copy_from_slice(remainder);
        self.tail_len = remainder.len();
    }

    pub(crate) fn finish_digest(mut self) -> Sha256Digest {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.tail[self.tail_len] = 0x80;
        self.tail_len += 1;
        if self.tail_len > 56 {
            self.tail[self.tail_len..].fill(0);
            sha256_compress(&mut self.state, &self.tail);
            self.tail = [0; 64];
        } else {
            self.tail[self.tail_len..56].fill(0);
        }
        self.tail[56..].copy_from_slice(&bit_len.to_be_bytes());
        sha256_compress(&mut self.state, &self.tail);
        let mut bytes = [0_u8; 32];
        for (word, chunk) in self.state.iter().zip(bytes.chunks_exact_mut(4)) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        Sha256Digest(bytes)
    }
}

fn write_python_json_number(value: &serde_json::Number, out: &mut String) -> Result<(), String> {
    let original = value.to_string();
    if value.is_i64()
        || value.is_u64()
        || !original
            .bytes()
            .any(|byte| matches!(byte, b'.' | b'e' | b'E'))
    {
        out.push_str(if original == "-0" { "0" } else { &original });
        return Ok(());
    }

    // `arbitrary_precision` preserves the source token so integers stay exact,
    // but Python parses non-integer JSON numbers through binary64 before dumping
    // them. Normalize through the same semantic domain rather than sealing an
    // alternate spelling such as `0.10000000000000001`.
    let finite = value
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("non-finite or out-of-range JSON number: {original}"))?;
    let rendered = serde_json::Number::from_f64(finite)
        .ok_or_else(|| format!("could not render finite JSON number: {original}"))?
        .to_string();
    let (negative, unsigned) = rendered
        .strip_prefix('-')
        .map_or((false, rendered.as_str()), |rest| (true, rest));
    let (mantissa, explicit_exponent) = match unsigned.find(['e', 'E']) {
        Some(at) => (
            &unsigned[..at],
            unsigned[at + 1..].parse::<i32>().unwrap_or(0),
        ),
        None => (unsigned, 0),
    };
    let decimal_at = mantissa.find('.').unwrap_or(mantissa.len());
    let digits: String = mantissa.chars().filter(|ch| *ch != '.').collect();
    let Some(first) = digits.bytes().position(|byte| byte != b'0') else {
        out.push_str(if negative { "-0.0" } else { "0.0" });
        return Ok(());
    };
    let last = digits
        .bytes()
        .rposition(|byte| byte != b'0')
        .expect("a first nonzero digit exists");
    let significant = &digits[first..=last];
    let exponent = explicit_exponent + decimal_at as i32 - first as i32 - 1;

    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent < 0 {
            out.push_str("0.");
            for _ in 0..(-exponent - 1) {
                out.push('0');
            }
            out.push_str(significant);
        } else {
            let integer_digits = exponent as usize + 1;
            if significant.len() <= integer_digits {
                out.push_str(significant);
                for _ in significant.len()..integer_digits {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&significant[..integer_digits]);
                out.push('.');
                out.push_str(&significant[integer_digits..]);
            }
        }
    } else {
        out.push(significant.as_bytes()[0] as char);
        if significant.len() > 1 {
            out.push('.');
            out.push_str(&significant[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let absolute = exponent.unsigned_abs();
        if absolute < 10 {
            out.push('0');
        }
        out.push_str(&absolute.to_string());
    }
    Ok(())
}

fn canonical_python_json_with_ascii_policy(
    value: &serde_json::Value,
    out: &mut String,
    ensure_ascii: bool,
) -> Result<(), String> {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => write_python_json_number(value, out)?,
        serde_json::Value::String(value) => {
            out.push('"');
            for ch in value.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\u{08}' => out.push_str("\\b"),
                    '\u{09}' => out.push_str("\\t"),
                    '\n' => out.push_str("\\n"),
                    '\u{0c}' => out.push_str("\\f"),
                    '\r' => out.push_str("\\r"),
                    '\u{20}'..='\u{7e}' => out.push(ch),
                    ch if !ensure_ascii && u32::from(ch) >= 0x20 => out.push(ch),
                    ch if u32::from(ch) <= 0xffff => {
                        out.push_str(&format!("\\u{:04x}", u32::from(ch)));
                    }
                    ch => {
                        let code = u32::from(ch) - 0x1_0000;
                        let high = 0xd800 + (code >> 10);
                        let low = 0xdc00 + (code & 0x3ff);
                        out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                    }
                }
            }
            out.push('"');
        }
        serde_json::Value::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                canonical_python_json_with_ascii_policy(value, out, ensure_ascii)?;
            }
            out.push(']');
        }
        serde_json::Value::Object(values) => {
            out.push('{');
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                canonical_python_json_with_ascii_policy(
                    &serde_json::Value::String(key.clone()),
                    out,
                    ensure_ascii,
                )?;
                out.push(':');
                canonical_python_json_with_ascii_policy(value, out, ensure_ascii)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Render one JSON value with the historical Python `ensure_ascii=True`
/// canonical spelling used by existing producer seals.
pub fn canonical_json(value: &serde_json::Value) -> Result<String, CanonicalJsonError> {
    let mut rendered = String::new();
    canonical_python_json_with_ascii_policy(value, &mut rendered, true)
        .map_err(CanonicalJsonError)?;
    Ok(rendered)
}

/// Render one JSON value with Python `ensure_ascii=False` canonical spelling.
///
/// This is a separate surface so v2 UTF-8 document seals cannot alter historical
/// trace and evidence identities that use [`canonical_json`].
pub fn canonical_json_utf8(value: &serde_json::Value) -> Result<String, CanonicalJsonError> {
    let mut rendered = String::new();
    canonical_python_json_with_ascii_policy(value, &mut rendered, false)
        .map_err(CanonicalJsonError)?;
    Ok(rendered)
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
#[error("canonical JSON rendering failed: {0}")]
pub struct CanonicalJsonError(String);

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn sha256_vectors_and_strict_spelling() {
        assert_eq!(
            Sha256Digest::of(b"").to_string(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            Sha256Digest::of(b"abc").to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let parsed =
            Sha256Digest::parse("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
                .unwrap();
        assert_eq!(parsed, Sha256Digest::of(b"abc"));
        assert!(Sha256Digest::parse(&"A".repeat(64)).is_err());
        assert!(Sha256Digest::parse("abc").is_err());
    }

    #[test]
    fn python_canonical_json_preserves_arbitrary_precision_integers() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"negative":-18446744073709551617,"positive":18446744073709551616}"#,
        )
        .unwrap();
        assert_eq!(
            canonical_json(&value).unwrap(),
            r#"{"negative":-18446744073709551617,"positive":18446744073709551616}"#
        );
    }

    #[test]
    fn python_canonical_json_normalizes_noncanonical_number_tokens() {
        for (raw, expected) in [
            ("-0", "0"),
            ("0.10000000000000001", "0.1"),
            ("1.234567890123456789", "1.2345678901234567"),
        ] {
            let value: serde_json::Value = serde_json::from_str(raw).unwrap();
            assert_eq!(canonical_json(&value).unwrap(), expected, "token {raw}");
        }
    }

    #[test]
    fn python_canonical_json_sorts_keys_and_escapes_strings() {
        let value = serde_json::json!({"z": "line\n", "a": [1, true, null]});
        assert_eq!(
            canonical_json(&value).unwrap(),
            r#"{"a":[1,true,null],"z":"line\n"}"#
        );
    }

    #[test]
    fn utf8_canonical_json_matches_python_ensure_ascii_false() {
        let value = serde_json::json!({
            "z": "é😀\u{7f}",
            "λ": "line\u{1f}",
        });
        assert_eq!(
            canonical_json_utf8(&value).unwrap(),
            "{\"z\":\"é😀\u{7f}\",\"λ\":\"line\\u001f\"}"
        );
        assert!(canonical_json(&value).unwrap().is_ascii());
    }
}
