//! Descriptor-anchored, no-follow reads for sealed trace bundles.
//!
//! Every relative component is opened beneath one retained directory descriptor.
//! Regular-file metadata and bytes come from the same descriptor, so path swaps,
//! symlinks, special files, and hash-then-reopen substitutions fail closed.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::os::fd::OwnedFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use rustix::fs::{AtFlags, Dir, FileType, Mode, OFlags, open, openat, statat};

const DIRECTORY_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::CLOEXEC)
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW);
const FILE_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::CLOEXEC)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK);
const MAX_TREE_DEPTH: usize = 16;
const MIN_TREE_ENTRY_BUDGET: u64 = 1_024;
const TREE_ENTRIES_PER_PAYLOAD: u64 = 8;

/// Stable filesystem identity used only to reject hard-link aliases during admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileIdentity {
    device: u64,
    inode: u64,
}

/// Bytes and metadata read from one retained regular-file descriptor.
pub struct AnchoredFile {
    raw: Vec<u8>,
    identity: FileIdentity,
}

impl AnchoredFile {
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    pub fn into_raw(self) -> Vec<u8> {
        self.raw
    }

    pub fn identity(&self) -> FileIdentity {
        self.identity
    }

    pub fn size_bytes(&self) -> u64 {
        self.raw.len() as u64
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AnchoredReadError {
    #[error("descriptor-anchored path is not a nonempty canonical relative path: {path:?}")]
    PathInvalid { path: PathBuf },
    #[error("could not access descriptor-anchored artifact {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("descriptor-anchored artifact is not a regular file: {path}")]
    NotRegular { path: PathBuf },
    #[error("descriptor-anchored artifact {path} has size {actual}, expected {expected}")]
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("descriptor-anchored artifact {path} has size {actual}, exceeding limit {limit}")]
    SizeLimitExceeded {
        path: PathBuf,
        limit: u64,
        actual: u64,
    },
    #[error("descriptor-anchored artifact {path} changed while its bytes were read")]
    ChangedWhileReading { path: PathBuf },
    #[error("could not reserve {bytes} bytes for descriptor-anchored artifact {path}")]
    AllocationFailed { path: PathBuf, bytes: u64 },
    #[error("threadblock payload traversal exceeds depth {limit} at {path}")]
    TreeDepthExceeded { path: PathBuf, limit: usize },
    #[error("threadblock payload traversal exceeds {limit} entries at {path}")]
    TreeEntryLimitExceeded { path: PathBuf, limit: u64 },
    #[error("threadblock payload contains more than {limit} .pb files")]
    TreeFileLimitExceeded { limit: u64 },
    #[error("threadblock payload bytes exceed declared limit {limit} at {path}")]
    TreeByteLimitExceeded { path: PathBuf, limit: u64 },
    #[error("descriptor-anchored files {first} and {second} alias the same inode")]
    HardlinkAlias { first: PathBuf, second: PathBuf },
}

/// One bundle root retained as a directory descriptor for the whole admission.
pub struct AnchoredDir {
    root: PathBuf,
    fd: OwnedFd,
}

impl AnchoredDir {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, AnchoredReadError> {
        let root = root.into();
        let fd = open(&root, DIRECTORY_FLAGS, Mode::empty()).map_err(|source| {
            AnchoredReadError::Io {
                path: root.clone(),
                source: source.into(),
            }
        })?;
        Ok(Self { root, fd })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn read_exact_regular(
        &self,
        relative: &Path,
        expected_size: u64,
    ) -> Result<AnchoredFile, AnchoredReadError> {
        let Some(fd) = self.open_regular(relative, false)? else {
            unreachable!("required opens never return absence")
        };
        read_opened_regular(
            fd,
            self.root.join(relative),
            Some(expected_size),
            expected_size,
        )
    }

    pub fn read_bounded_regular(
        &self,
        relative: &Path,
        maximum_size: u64,
    ) -> Result<AnchoredFile, AnchoredReadError> {
        let Some(fd) = self.open_regular(relative, false)? else {
            unreachable!("required opens never return absence")
        };
        read_opened_regular(fd, self.root.join(relative), None, maximum_size)
    }

    pub fn read_optional_exact_regular(
        &self,
        relative: &Path,
        expected_size: u64,
    ) -> Result<Option<AnchoredFile>, AnchoredReadError> {
        self.open_regular(relative, true)?
            .map(|fd| {
                read_opened_regular(
                    fd,
                    self.root.join(relative),
                    Some(expected_size),
                    expected_size,
                )
            })
            .transpose()
    }

    pub fn read_optional_bounded_regular(
        &self,
        relative: &Path,
        maximum_size: u64,
    ) -> Result<Option<AnchoredFile>, AnchoredReadError> {
        self.open_regular(relative, true)?
            .map(|fd| read_opened_regular(fd, self.root.join(relative), None, maximum_size))
            .transpose()
    }

    pub fn probe_optional_regular(
        &self,
        relative: &Path,
    ) -> Result<Option<FileIdentity>, AnchoredReadError> {
        self.open_regular(relative, true)?
            .map(|fd| regular_identity(fd, self.root.join(relative)))
            .transpose()
    }

    pub(crate) fn read_pb_tree(
        &self,
        relative: &Path,
        maximum_files: u64,
        maximum_bytes: u64,
    ) -> Result<AnchoredTree, AnchoredReadError> {
        let directory =
            self.open_directory(relative, false)?
                .ok_or_else(|| AnchoredReadError::Io {
                    path: self.root.join(relative),
                    source: std::io::Error::from(std::io::ErrorKind::NotFound),
                })?;
        let entry_limit = maximum_files
            .checked_mul(TREE_ENTRIES_PER_PAYLOAD)
            .and_then(|value| value.checked_add(MIN_TREE_ENTRY_BUDGET))
            .unwrap_or(u64::MAX);
        let mut state = TreeState {
            files: Vec::new(),
            identities: BTreeMap::new(),
            entries_seen: 0,
            entry_limit,
            maximum_files,
            maximum_bytes,
            bytes_seen: 0,
        };
        self.read_pb_tree_from(directory, relative, 0, &mut state)?;
        state
            .files
            .sort_by(|left, right| left.relative.cmp(&right.relative));
        Ok(AnchoredTree {
            files: state.files,
            bytes: state.bytes_seen,
        })
    }

    fn read_pb_tree_from(
        &self,
        directory: OwnedFd,
        relative: &Path,
        depth: usize,
        state: &mut TreeState,
    ) -> Result<(), AnchoredReadError> {
        if depth > MAX_TREE_DEPTH {
            return Err(AnchoredReadError::TreeDepthExceeded {
                path: self.root.join(relative),
                limit: MAX_TREE_DEPTH,
            });
        }
        let mut entries = Dir::read_from(&directory).map_err(|source| AnchoredReadError::Io {
            path: self.root.join(relative),
            source: source.into(),
        })?;
        while let Some(entry) = entries.next() {
            let entry = entry.map_err(|source| AnchoredReadError::Io {
                path: self.root.join(relative),
                source: source.into(),
            })?;
            let name_bytes = entry.file_name().to_bytes();
            if matches!(name_bytes, b"." | b"..") {
                continue;
            }
            state.entries_seen = state.entries_seen.checked_add(1).ok_or(
                AnchoredReadError::TreeEntryLimitExceeded {
                    path: self.root.join(relative),
                    limit: state.entry_limit,
                },
            )?;
            if state.entries_seen > state.entry_limit {
                return Err(AnchoredReadError::TreeEntryLimitExceeded {
                    path: self.root.join(relative),
                    limit: state.entry_limit,
                });
            }
            let name = OsStr::from_bytes(name_bytes);
            let child_relative = relative.join(name);
            let child_path = self.root.join(&child_relative);
            let stat = statat(&directory, name, AtFlags::SYMLINK_NOFOLLOW).map_err(|source| {
                AnchoredReadError::Io {
                    path: child_path.clone(),
                    source: source.into(),
                }
            })?;
            let file_type = FileType::from_raw_mode(stat.st_mode);
            let pb_named = Path::new(name)
                .extension()
                .is_some_and(|extension| extension == "pb")
                || name == ".pb";
            match file_type {
                FileType::Directory => {
                    if pb_named {
                        return Err(AnchoredReadError::NotRegular { path: child_path });
                    }
                    let child = openat(&directory, name, DIRECTORY_FLAGS, Mode::empty()).map_err(
                        |source| AnchoredReadError::Io {
                            path: child_path,
                            source: source.into(),
                        },
                    )?;
                    self.read_pb_tree_from(child, &child_relative, depth + 1, state)?;
                }
                FileType::RegularFile if pb_named => {
                    let next_file_count = u64::try_from(state.files.len())
                        .unwrap_or(u64::MAX)
                        .saturating_add(1);
                    if next_file_count > state.maximum_files {
                        return Err(AnchoredReadError::TreeFileLimitExceeded {
                            limit: state.maximum_files,
                        });
                    }
                    let fd =
                        openat(&directory, name, FILE_FLAGS, Mode::empty()).map_err(|source| {
                            AnchoredReadError::Io {
                                path: child_path.clone(),
                                source: source.into(),
                            }
                        })?;
                    let remaining = state
                        .maximum_bytes
                        .checked_sub(state.bytes_seen)
                        .ok_or_else(|| AnchoredReadError::TreeByteLimitExceeded {
                            path: child_path.clone(),
                            limit: state.maximum_bytes,
                        })?;
                    let file = read_opened_regular(fd, child_path.clone(), None, remaining)
                        .map_err(|error| match error {
                            AnchoredReadError::SizeLimitExceeded { .. } => {
                                AnchoredReadError::TreeByteLimitExceeded {
                                    path: child_path.clone(),
                                    limit: state.maximum_bytes,
                                }
                            }
                            other => other,
                        })?;
                    let next_bytes =
                        state
                            .bytes_seen
                            .checked_add(file.size_bytes())
                            .ok_or_else(|| AnchoredReadError::TreeByteLimitExceeded {
                                path: child_path,
                                limit: state.maximum_bytes,
                            })?;
                    if let Some(first) = state
                        .identities
                        .insert(file.identity(), child_relative.clone())
                    {
                        return Err(AnchoredReadError::HardlinkAlias {
                            first: self.root.join(first),
                            second: self.root.join(child_relative),
                        });
                    }
                    state.bytes_seen = next_bytes;
                    state.files.push(AnchoredTreeFile {
                        relative: child_relative,
                        file,
                    });
                }
                FileType::RegularFile => {}
                _ => return Err(AnchoredReadError::NotRegular { path: child_path }),
            }
        }
        Ok(())
    }

    fn open_directory(
        &self,
        relative: &Path,
        optional: bool,
    ) -> Result<Option<OwnedFd>, AnchoredReadError> {
        let components = relative_components(relative, true)?;
        let mut current = rustix::io::dup(&self.fd).map_err(|source| AnchoredReadError::Io {
            path: self.root.clone(),
            source: source.into(),
        })?;
        let mut walked = PathBuf::new();
        for component in components {
            walked.push(&component);
            current = match openat(&current, &component, DIRECTORY_FLAGS, Mode::empty()) {
                Ok(fd) => fd,
                Err(source) if optional && source == rustix::io::Errno::NOENT => return Ok(None),
                Err(source) => {
                    return Err(AnchoredReadError::Io {
                        path: self.root.join(walked),
                        source: source.into(),
                    });
                }
            };
        }
        Ok(Some(current))
    }

    fn open_regular(
        &self,
        relative: &Path,
        optional: bool,
    ) -> Result<Option<OwnedFd>, AnchoredReadError> {
        let mut components = relative_components(relative, false)?;
        let leaf = components.pop().expect("nonempty path checked");
        let parent = components.iter().collect::<PathBuf>();
        let Some(directory) = self.open_directory(&parent, optional)? else {
            return Ok(None);
        };
        match openat(&directory, &leaf, FILE_FLAGS, Mode::empty()) {
            Ok(fd) => Ok(Some(fd)),
            Err(source) if optional && source == rustix::io::Errno::NOENT => Ok(None),
            Err(source) => Err(AnchoredReadError::Io {
                path: self.root.join(relative),
                source: source.into(),
            }),
        }
    }
}

pub(crate) struct AnchoredTreeFile {
    pub(crate) relative: PathBuf,
    pub(crate) file: AnchoredFile,
}

pub(crate) struct AnchoredTree {
    pub(crate) files: Vec<AnchoredTreeFile>,
    pub(crate) bytes: u64,
}

struct TreeState {
    files: Vec<AnchoredTreeFile>,
    identities: BTreeMap<FileIdentity, PathBuf>,
    entries_seen: u64,
    entry_limit: u64,
    maximum_files: u64,
    maximum_bytes: u64,
    bytes_seen: u64,
}

fn relative_components(path: &Path, allow_empty: bool) -> Result<Vec<PathBuf>, AnchoredReadError> {
    let components = path
        .components()
        .map(|component| match component {
            Component::Normal(value) => Ok(PathBuf::from(value)),
            _ => Err(AnchoredReadError::PathInvalid {
                path: path.to_path_buf(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !allow_empty && components.is_empty() {
        return Err(AnchoredReadError::PathInvalid {
            path: path.to_path_buf(),
        });
    }
    Ok(components)
}

fn regular_identity(fd: OwnedFd, path: PathBuf) -> Result<FileIdentity, AnchoredReadError> {
    let file = File::from(fd);
    let metadata = file.metadata().map_err(|source| AnchoredReadError::Io {
        path: path.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(AnchoredReadError::NotRegular { path });
    }
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

fn read_opened_regular(
    fd: OwnedFd,
    path: PathBuf,
    expected_size: Option<u64>,
    maximum_size: u64,
) -> Result<AnchoredFile, AnchoredReadError> {
    let file = File::from(fd);
    let before = file.metadata().map_err(|source| AnchoredReadError::Io {
        path: path.clone(),
        source,
    })?;
    if !before.is_file() {
        return Err(AnchoredReadError::NotRegular { path });
    }
    let size = before.len();
    if let Some(expected) = expected_size
        && size != expected
    {
        return Err(AnchoredReadError::SizeMismatch {
            path,
            expected,
            actual: size,
        });
    }
    if size > maximum_size {
        return Err(AnchoredReadError::SizeLimitExceeded {
            path,
            limit: maximum_size,
            actual: size,
        });
    }
    let capacity = usize::try_from(size).map_err(|_| AnchoredReadError::AllocationFailed {
        path: path.clone(),
        bytes: size,
    })?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(capacity)
        .map_err(|_| AnchoredReadError::AllocationFailed {
            path: path.clone(),
            bytes: size,
        })?;
    let read_limit = size.checked_add(1).unwrap_or(size);
    (&file)
        .take(read_limit)
        .read_to_end(&mut raw)
        .map_err(|source| AnchoredReadError::Io {
            path: path.clone(),
            source,
        })?;
    let after = file.metadata().map_err(|source| AnchoredReadError::Io {
        path: path.clone(),
        source,
    })?;
    let after_observation = FileObservation::from_metadata(&after);
    let before_observation = FileObservation::from_metadata(&before);
    if u64::try_from(raw.len()).ok() != Some(size) || after_observation != before_observation {
        return Err(AnchoredReadError::ChangedWhileReading { path });
    }
    Ok(AnchoredFile {
        raw,
        identity: before_observation.identity,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileObservation {
    identity: FileIdentity,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileObservation {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            identity: FileIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            },
            size: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn anchored_reads_reject_symlinks_and_hardlink_aliases() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("file.pb"), b"payload").unwrap();
        symlink("file.pb", temp.path().join("link.pb")).unwrap();
        let reader = AnchoredDir::open(temp.path()).unwrap();
        assert!(matches!(
            reader.read_exact_regular(Path::new("link.pb"), 7),
            Err(AnchoredReadError::Io { .. })
        ));

        std::fs::create_dir(temp.path().join("tree")).unwrap();
        std::fs::hard_link(temp.path().join("file.pb"), temp.path().join("tree/a.pb")).unwrap();
        std::fs::hard_link(temp.path().join("file.pb"), temp.path().join("tree/b.pb")).unwrap();
        assert!(matches!(
            reader.read_pb_tree(Path::new("tree"), 2, 14),
            Err(AnchoredReadError::HardlinkAlias { .. })
        ));
    }
}
