//! Strict JSON decoding helpers shared by simulator input cards and traces.

use std::collections::BTreeSet;

use serde::Deserializer;
use serde::de::{self, DeserializeOwned, DeserializeSeed, MapAccess, SeqAccess, Visitor};

const MAX_JSON_DEPTH: usize = 256;

#[derive(Clone, Copy)]
struct RejectDuplicateMembers {
    depth: usize,
}

impl RejectDuplicateMembers {
    fn child(self) -> Result<Self, serde_json::Error> {
        self.depth
            .checked_add(1)
            .filter(|depth| *depth <= MAX_JSON_DEPTH)
            .map(|depth| Self { depth })
            .ok_or_else(|| {
                serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("JSON nesting depth exceeds {MAX_JSON_DEPTH}"),
                ))
            })
    }
}

impl<'de> DeserializeSeed<'de> for RejectDuplicateMembers {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        if self.depth > MAX_JSON_DEPTH {
            return Err(de::Error::custom(format!(
                "JSON nesting depth exceeds {MAX_JSON_DEPTH}"
            )));
        }
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for RejectDuplicateMembers {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object members")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i128<E>(self, _value: i128) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u128<E>(self, _value: u128) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_char<E>(self, _value: char) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(Self {
            depth: self.depth.saturating_add(1),
        })
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_newtype_struct<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(Self {
            depth: self.depth.saturating_add(1),
        })
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let child = self
            .child()
            .map_err(|error| de::Error::custom(error.to_string()))?;
        while sequence.next_element_seed(child)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let child = self
            .child()
            .map_err(|error| de::Error::custom(error.to_string()))?;
        let mut members = BTreeSet::new();
        while let Some(member) = object.next_key::<String>()? {
            if !members.insert(member.clone()) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON member {member:?}"
                )));
            }
            object.next_value_seed(child)?;
        }
        Ok(())
    }
}

/// Deserialize JSON only after recursively rejecting duplicate object
/// members. Standard serde decoding is last-wins for duplicates, which is not
/// safe for simulator identities, paths, or timing metadata.
pub fn strict_json_from_slice<T: DeserializeOwned>(raw: &[u8]) -> Result<T, serde_json::Error> {
    let mut duplicate_check = serde_json::Deserializer::from_slice(raw);
    RejectDuplicateMembers { depth: 0 }.deserialize(&mut duplicate_check)?;
    duplicate_check.end()?;
    serde_json::from_slice(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_members_and_excessive_nesting_are_rejected() {
        assert!(strict_json_from_slice::<serde_json::Value>(br#"{"x": 1, "x": 2}"#).is_err());
        let mut deeply_nested = "[".repeat(MAX_JSON_DEPTH + 1);
        deeply_nested.push('0');
        deeply_nested.push_str(&"]".repeat(MAX_JSON_DEPTH + 1));
        assert!(strict_json_from_slice::<serde_json::Value>(deeply_nested.as_bytes()).is_err());
    }
}
