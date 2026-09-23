//! Descriptor-anchored, no-follow reads for trace bundles.
//!
//! Every relative component is opened beneath one retained directory descriptor.
//! Regular-file metadata and bytes come from the same descriptor, so path swaps,
//! symlinks, special files, and hash-then-reopen substitutions fail closed.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::File;
use std::mem::MaybeUninit;
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Component, Path, PathBuf};

use rustix::fs::{AtFlags, Dir, FileType, Mode, OFlags, fcntl_getfl, fstat, open, openat, statat};

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
const MAX_TREE_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Stable filesystem identity used to reject hard-link aliases during loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileIdentity {
    device: u64,
    inode: u64,
}

/// One regular file generation observed through a descriptor-anchored name.
///
/// Device/inode alone does not distinguish an in-place overwrite. Size plus the
/// modification/change timestamps make persistent replacement, deletion, and
/// same-inode substitution observable while a bundle is being read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileGenerationIdentity {
    identity: FileIdentity,
    size_bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileGenerationIdentity {
    fn from_stat(stat: &rustix::fs::Stat) -> Option<Self> {
        Some(Self {
            identity: FileIdentity {
                device: stat.st_dev,
                inode: stat.st_ino,
            },
            size_bytes: u64::try_from(stat.st_size).ok()?,
            modified_seconds: stat.st_mtime,
            modified_nanoseconds: i64::try_from(stat.st_mtime_nsec).ok()?,
            changed_seconds: stat.st_ctime,
            changed_nanoseconds: i64::try_from(stat.st_ctime_nsec).ok()?,
        })
    }

    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            identity: FileIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
            },
            size_bytes: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }

    pub fn file_identity(self) -> FileIdentity {
        self.identity
    }

    pub fn size_bytes(self) -> u64 {
        self.size_bytes
    }
}

/// Bytes and metadata read from one retained regular-file descriptor.
pub struct AnchoredFile {
    raw: Vec<u8>,
    generation: FileGenerationIdentity,
}

impl AnchoredFile {
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }

    pub fn into_raw(self) -> Vec<u8> {
        self.raw
    }

    pub fn identity(&self) -> FileIdentity {
        self.generation.file_identity()
    }

    pub fn generation_identity(&self) -> FileGenerationIdentity {
        self.generation
    }

    pub fn size_bytes(&self) -> u64 {
        self.raw.len() as u64
    }
}

/// One duplicated inherited regular-file descriptor.
///
/// Duplication preserves the inherited open file description and its access
/// mode. Positional reads keep the shared open-file-description offset unchanged.
pub struct RetainedFile {
    path: PathBuf,
    fd: OwnedFd,
}

impl RetainedFile {
    /// Validate an inherited regular-file descriptor without allocating another slot.
    pub fn validate_inherited(raw_fd: u32) -> Result<(), AnchoredReadError> {
        validate_retained_descriptor(raw_fd, RetainedDescriptorKind::Regular)
    }

    pub fn open(raw_fd: u32) -> Result<Self, AnchoredReadError> {
        Self::validate_inherited(raw_fd)?;
        let path = retained_descriptor_label(raw_fd);
        let fd = duplicate_readonly_descriptor(raw_fd, &path)?;
        Ok(Self { path, fd })
    }

    pub fn read_bounded_regular(
        &self,
        maximum_size: u64,
    ) -> Result<AnchoredFile, AnchoredReadError> {
        let fd = rustix::io::dup(&self.fd).map_err(|source| AnchoredReadError::Io {
            path: self.path.clone(),
            source: source.into(),
        })?;
        read_opened_regular(fd, self.path.clone(), None, maximum_size)
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
    #[error("retained descriptor is unavailable: {path}: {source}")]
    RetainedDescriptorUnavailable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("retained descriptor is not an ordinary read-only descriptor: {path}")]
    RetainedDescriptorModeInvalid { path: PathBuf },
    #[error("descriptor-anchored artifact is not a regular file: {path}")]
    NotRegular { path: PathBuf },
    #[error("threadblock payload tree contains a non-protobuf regular file: {path}")]
    NonProtobuf { path: PathBuf },
    #[error("descriptor-anchored artifact generation metadata is invalid: {path}")]
    GenerationIdentityInvalid { path: PathBuf },
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
    #[error("descriptor-anchored artifact {path} changed after its roster was captured")]
    ChangedSinceRoster { path: PathBuf },
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

/// One bundle root retained as a directory descriptor for a complete read.
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

    /// Validate an inherited directory descriptor without allocating another slot.
    pub fn validate_inherited(raw_fd: u32) -> Result<(), AnchoredReadError> {
        validate_retained_descriptor(raw_fd, RetainedDescriptorKind::Directory)
    }

    /// Duplicate one inherited read-only directory descriptor as an owned anchor.
    pub fn open_retained(raw_fd: u32) -> Result<Self, AnchoredReadError> {
        Self::validate_inherited(raw_fd)?;
        let root = retained_descriptor_label(raw_fd);
        let fd = duplicate_readonly_descriptor(raw_fd, &root)?;
        let stat = fstat(&fd).map_err(|source| AnchoredReadError::Io {
            path: root.clone(),
            source: source.into(),
        })?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
            return Err(AnchoredReadError::Io {
                path: root,
                source: std::io::Error::from(std::io::ErrorKind::NotADirectory),
            });
        }
        Ok(Self { root, fd })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Derive a child anchor solely through this retained directory descriptor.
    pub fn descendant(&self, relative: &Path) -> Result<Self, AnchoredReadError> {
        let fd = self
            .open_directory(relative, false)?
            .expect("required directory opens never return absence");
        Ok(Self {
            root: self.root.join(relative),
            fd,
        })
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

    /// Inspect a no-follow regular-file generation without reopening its contents.
    ///
    /// This binds a retained directory capability to the named generation even
    /// after read permission on the leaf has been removed.
    pub fn inspect_regular_generation(
        &self,
        relative: &Path,
    ) -> Result<FileGenerationIdentity, AnchoredReadError> {
        let mut components = relative_components(relative, false)?;
        let leaf = components.pop().expect("nonempty path checked");
        let parent = components.iter().collect::<PathBuf>();
        let directory = self
            .open_directory(&parent, false)?
            .expect("required directory opens never return absence");
        let path = self.root.join(relative);
        let stat = statat(&directory, &leaf, AtFlags::SYMLINK_NOFOLLOW).map_err(|source| {
            AnchoredReadError::Io {
                path: path.clone(),
                source: source.into(),
            }
        })?;
        if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
            return Err(AnchoredReadError::NotRegular { path });
        }
        FileGenerationIdentity::from_stat(&stat)
            .ok_or(AnchoredReadError::GenerationIdentityInvalid { path })
    }

    /// Inspect the device/inode portion of a no-follow regular-file name.
    pub fn inspect_regular_identity(
        &self,
        relative: &Path,
    ) -> Result<FileIdentity, AnchoredReadError> {
        self.inspect_regular_generation(relative)
            .map(FileGenerationIdentity::file_identity)
    }

    pub(crate) fn read_pb_tree(
        &self,
        relative: &Path,
        maximum_files: u64,
        maximum_bytes: u64,
    ) -> Result<AnchoredTree, AnchoredReadError> {
        self.read_pb_tree_with_file_limit(
            relative,
            maximum_files,
            maximum_bytes,
            MAX_TREE_FILE_BYTES,
        )
    }

    pub(crate) fn read_pb_tree_with_file_limit(
        &self,
        relative: &Path,
        maximum_files: u64,
        maximum_bytes: u64,
        maximum_file_bytes: u64,
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
            maximum_file_bytes,
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

    /// Capture a complete bounded `.pb` roster without retaining payload
    /// bytes. Every leaf is opened through the retained root, checked as a
    /// regular readable file, and bound to its exact generation. Callers may
    /// later read one member at a time with [`Self::read_rostered_regular`].
    pub(crate) fn roster_pb_tree(
        &self,
        relative: &Path,
        maximum_files: u64,
        maximum_bytes: u64,
    ) -> Result<AnchoredTreeRoster, AnchoredReadError> {
        self.roster_pb_tree_with_file_limit(
            relative,
            maximum_files,
            maximum_bytes,
            MAX_TREE_FILE_BYTES,
        )
    }

    pub(crate) fn roster_pb_tree_with_file_limit(
        &self,
        relative: &Path,
        maximum_files: u64,
        maximum_bytes: u64,
        maximum_file_bytes: u64,
    ) -> Result<AnchoredTreeRoster, AnchoredReadError> {
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
        let mut state = RosterState {
            files: Vec::new(),
            identities: BTreeMap::new(),
            entries_seen: 0,
            entry_limit,
            maximum_files,
            maximum_bytes,
            maximum_file_bytes,
            bytes_seen: 0,
        };
        self.roster_pb_tree_from(directory, relative, 0, &mut state)?;
        state
            .files
            .sort_by(|left, right| left.relative.cmp(&right.relative));
        Ok(AnchoredTreeRoster {
            files: state.files,
            bytes: state.bytes_seen,
        })
    }

    fn roster_pb_tree_from(
        &self,
        directory: OwnedFd,
        relative: &Path,
        depth: usize,
        state: &mut RosterState,
    ) -> Result<(), AnchoredReadError> {
        if depth > MAX_TREE_DEPTH {
            return Err(AnchoredReadError::TreeDepthExceeded {
                path: self.root.join(relative),
                limit: MAX_TREE_DEPTH,
            });
        }
        let entries = Dir::read_from(&directory).map_err(|source| AnchoredReadError::Io {
            path: self.root.join(relative),
            source: source.into(),
        })?;
        for entry in entries {
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
                    self.roster_pb_tree_from(child, &child_relative, depth + 1, state)?;
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
                    let remaining = state.maximum_bytes.checked_sub(state.bytes_seen).ok_or(
                        AnchoredReadError::TreeByteLimitExceeded {
                            path: child_path.clone(),
                            limit: state.maximum_bytes,
                        },
                    )?;
                    let file_limit = remaining.min(state.maximum_file_bytes);
                    let generation = inspect_opened_regular(fd, child_path.clone(), file_limit)
                        .map_err(|error| match error {
                            AnchoredReadError::SizeLimitExceeded { .. } => {
                                AnchoredReadError::TreeByteLimitExceeded {
                                    path: child_path.clone(),
                                    limit: file_limit,
                                }
                            }
                            other => other,
                        })?;
                    let next_bytes = state
                        .bytes_seen
                        .checked_add(generation.size_bytes())
                        .ok_or(AnchoredReadError::TreeByteLimitExceeded {
                            path: child_path,
                            limit: state.maximum_bytes,
                        })?;
                    if let Some(first) = state
                        .identities
                        .insert(generation.file_identity(), child_relative.clone())
                    {
                        return Err(AnchoredReadError::HardlinkAlias {
                            first: self.root.join(first),
                            second: self.root.join(child_relative),
                        });
                    }
                    state.bytes_seen = next_bytes;
                    state.files.push(AnchoredTreeRosterFile {
                        relative: child_relative,
                        generation,
                    });
                }
                FileType::RegularFile => {
                    return Err(AnchoredReadError::NonProtobuf { path: child_path });
                }
                _ => return Err(AnchoredReadError::NotRegular { path: child_path }),
            }
        }
        Ok(())
    }

    /// Read one previously rostered file through the retained root and reject
    /// replacement or in-place mutation before exposing its bytes.
    pub(crate) fn read_rostered_regular(
        &self,
        relative: &Path,
        expected: FileGenerationIdentity,
    ) -> Result<AnchoredFile, AnchoredReadError> {
        let file = self.read_exact_regular(relative, expected.size_bytes())?;
        if file.generation_identity() != expected {
            return Err(AnchoredReadError::ChangedSinceRoster {
                path: self.root.join(relative),
            });
        }
        Ok(file)
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
        let entries = Dir::read_from(&directory).map_err(|source| AnchoredReadError::Io {
            path: self.root.join(relative),
            source: source.into(),
        })?;
        for entry in entries {
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
                    let remaining = state.maximum_bytes.checked_sub(state.bytes_seen).ok_or(
                        AnchoredReadError::TreeByteLimitExceeded {
                            path: child_path.clone(),
                            limit: state.maximum_bytes,
                        },
                    )?;
                    let file_limit = remaining.min(state.maximum_file_bytes);
                    let file = read_opened_regular(fd, child_path.clone(), None, file_limit)
                        .map_err(|error| match error {
                            AnchoredReadError::SizeLimitExceeded { .. } => {
                                AnchoredReadError::TreeByteLimitExceeded {
                                    path: child_path.clone(),
                                    limit: file_limit,
                                }
                            }
                            other => other,
                        })?;
                    let next_bytes = state.bytes_seen.checked_add(file.size_bytes()).ok_or(
                        AnchoredReadError::TreeByteLimitExceeded {
                            path: child_path,
                            limit: state.maximum_bytes,
                        },
                    )?;
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
                FileType::RegularFile => {
                    return Err(AnchoredReadError::NonProtobuf { path: child_path });
                }
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
    #[cfg_attr(not(feature = "campaign"), allow(dead_code))]
    pub(crate) bytes: u64,
}

pub(crate) struct AnchoredTreeRosterFile {
    pub(crate) relative: PathBuf,
    pub(crate) generation: FileGenerationIdentity,
}

pub(crate) struct AnchoredTreeRoster {
    pub(crate) files: Vec<AnchoredTreeRosterFile>,
    #[allow(dead_code)]
    pub(crate) bytes: u64,
}

struct TreeState {
    files: Vec<AnchoredTreeFile>,
    identities: BTreeMap<FileIdentity, PathBuf>,
    entries_seen: u64,
    entry_limit: u64,
    maximum_files: u64,
    maximum_bytes: u64,
    maximum_file_bytes: u64,
    bytes_seen: u64,
}

struct RosterState {
    files: Vec<AnchoredTreeRosterFile>,
    identities: BTreeMap<FileIdentity, PathBuf>,
    entries_seen: u64,
    entry_limit: u64,
    maximum_files: u64,
    maximum_bytes: u64,
    maximum_file_bytes: u64,
    bytes_seen: u64,
}

fn retained_descriptor_label(raw_fd: u32) -> PathBuf {
    PathBuf::from(format!("inherited-fd:{raw_fd}"))
}

#[derive(Clone, Copy)]
enum RetainedDescriptorKind {
    Regular,
    Directory,
}

fn retained_descriptor_error(path: &Path, source: std::io::Error) -> AnchoredReadError {
    if source.raw_os_error() == Some(libc::EBADF) {
        AnchoredReadError::RetainedDescriptorUnavailable {
            path: path.to_path_buf(),
            source,
        }
    } else {
        AnchoredReadError::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

#[allow(unsafe_code)]
fn validate_retained_descriptor(
    raw_fd: u32,
    kind: RetainedDescriptorKind,
) -> Result<(), AnchoredReadError> {
    let path = retained_descriptor_label(raw_fd);
    let raw_fd = libc::c_int::try_from(raw_fd).map_err(|_| AnchoredReadError::Io {
        path: path.clone(),
        source: std::io::Error::from(std::io::ErrorKind::InvalidInput),
    })?;
    let flags = loop {
        // SAFETY: F_GETFL accepts an untrusted integer descriptor and reports
        // EBADF without taking ownership of it.
        let flags = unsafe { libc::fcntl(raw_fd, libc::F_GETFL) };
        if flags >= 0 {
            break flags;
        }
        let source = std::io::Error::last_os_error();
        if source.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(retained_descriptor_error(&path, source));
    };
    if flags & libc::O_ACCMODE != libc::O_RDONLY || flags & libc::O_PATH != 0 {
        return Err(AnchoredReadError::RetainedDescriptorModeInvalid { path });
    }

    let metadata = loop {
        let mut metadata = MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `metadata` points to writable storage for one libc::stat and is
        // only assumed initialized after fstat succeeds. The raw descriptor is
        // borrowed, not owned or closed.
        if unsafe { libc::fstat(raw_fd, metadata.as_mut_ptr()) } == 0 {
            // SAFETY: the successful fstat initialized the complete structure.
            break unsafe { metadata.assume_init() };
        }
        let source = std::io::Error::last_os_error();
        if source.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(retained_descriptor_error(&path, source));
    };
    let file_type = metadata.st_mode & libc::S_IFMT;
    match kind {
        RetainedDescriptorKind::Regular if file_type != libc::S_IFREG => {
            Err(AnchoredReadError::NotRegular { path })
        }
        RetainedDescriptorKind::Directory if file_type != libc::S_IFDIR => {
            Err(AnchoredReadError::Io {
                path,
                source: std::io::Error::from(std::io::ErrorKind::NotADirectory),
            })
        }
        _ => Ok(()),
    }
}

#[allow(unsafe_code)]
fn duplicate_readonly_descriptor(raw_fd: u32, path: &Path) -> Result<OwnedFd, AnchoredReadError> {
    let raw_fd = libc::c_int::try_from(raw_fd).map_err(|_| AnchoredReadError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::from(std::io::ErrorKind::InvalidInput),
    })?;
    // SAFETY: `fcntl(F_DUPFD_CLOEXEC)` accepts an untrusted integer descriptor,
    // reports EBADF for a closed value, and returns a fresh owned descriptor on
    // success. It neither takes ownership of nor closes the supplied descriptor.
    let duplicated = loop {
        // SAFETY: see the immediately preceding safety rationale.
        let duplicated = unsafe { libc::fcntl(raw_fd, libc::F_DUPFD_CLOEXEC, 3) };
        if duplicated >= 0 {
            break duplicated;
        }
        let source = std::io::Error::last_os_error();
        if source.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(retained_descriptor_error(path, source));
    };
    // SAFETY: a successful F_DUPFD_CLOEXEC call returns a new descriptor owned by
    // this process, and no other Rust owner has been constructed for that value.
    let fd = unsafe { OwnedFd::from_raw_fd(duplicated) };
    let flags = fcntl_getfl(&fd).map_err(|source| AnchoredReadError::Io {
        path: path.to_path_buf(),
        source: source.into(),
    })?;
    if flags.intersects(OFlags::WRONLY | OFlags::RDWR | OFlags::PATH) {
        return Err(AnchoredReadError::RetainedDescriptorModeInvalid {
            path: path.to_path_buf(),
        });
    }
    Ok(fd)
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

fn inspect_opened_regular(
    fd: OwnedFd,
    path: PathBuf,
    maximum_size: u64,
) -> Result<FileGenerationIdentity, AnchoredReadError> {
    let file = File::from(fd);
    let metadata = file.metadata().map_err(|source| AnchoredReadError::Io {
        path: path.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(AnchoredReadError::NotRegular { path });
    }
    let size = metadata.len();
    if size > maximum_size {
        return Err(AnchoredReadError::SizeLimitExceeded {
            path,
            limit: maximum_size,
            actual: size,
        });
    }
    Ok(FileGenerationIdentity::from_metadata(&metadata))
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
    let read_limit = size.checked_add(1).unwrap_or(size);
    let capacity =
        usize::try_from(read_limit).map_err(|_| AnchoredReadError::AllocationFailed {
            path: path.clone(),
            bytes: read_limit,
        })?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(capacity)
        .map_err(|_| AnchoredReadError::AllocationFailed {
            path: path.clone(),
            bytes: read_limit,
        })?;
    let mut offset = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    while offset < read_limit {
        let remaining = read_limit - offset;
        let requested = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("bounded read chunk fits usize");
        let count = loop {
            match file.read_at(&mut buffer[..requested], offset) {
                Ok(count) => break count,
                Err(source) if source.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(source) => {
                    return Err(AnchoredReadError::Io {
                        path: path.clone(),
                        source,
                    });
                }
            }
        };
        if count == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..count]);
        offset = offset
            .checked_add(u64::try_from(count).expect("read count fits u64"))
            .expect("read offset is bounded by file size plus one");
    }
    let after = file.metadata().map_err(|source| AnchoredReadError::Io {
        path: path.clone(),
        source,
    })?;
    let after_generation = FileGenerationIdentity::from_metadata(&after);
    let before_generation = FileGenerationIdentity::from_metadata(&before);
    if u64::try_from(raw.len()).ok() != Some(size) || after_generation != before_generation {
        return Err(AnchoredReadError::ChangedWhileReading { path });
    }
    Ok(AnchoredFile {
        raw,
        generation: before_generation,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{PermissionsExt, symlink};

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
        assert!(matches!(
            reader.roster_pb_tree(Path::new("tree"), 2, 14),
            Err(AnchoredReadError::HardlinkAlias { .. })
        ));
    }

    #[test]
    fn roster_reads_exact_generations_one_at_a_time() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("tree")).unwrap();
        std::fs::write(temp.path().join("tree/b.pb"), b"second").unwrap();
        std::fs::write(temp.path().join("tree/a.pb"), b"first").unwrap();
        let reader = AnchoredDir::open(temp.path()).unwrap();
        let roster = reader.roster_pb_tree(Path::new("tree"), 2, 11).unwrap();
        assert_eq!(roster.bytes, 11);
        assert_eq!(
            roster
                .files
                .iter()
                .map(|file| file.relative.as_path())
                .collect::<Vec<_>>(),
            vec![Path::new("tree/a.pb"), Path::new("tree/b.pb")]
        );
        assert_eq!(
            reader
                .read_rostered_regular(&roster.files[1].relative, roster.files[1].generation,)
                .unwrap()
                .raw(),
            b"second"
        );

        std::fs::write(temp.path().join("replacement.pb"), b"other").unwrap();
        std::fs::rename(
            temp.path().join("replacement.pb"),
            temp.path().join("tree/a.pb"),
        )
        .unwrap();
        assert!(matches!(
            reader.read_rostered_regular(&roster.files[0].relative, roster.files[0].generation,),
            Err(AnchoredReadError::ChangedSinceRoster { .. })
        ));
    }

    #[test]
    fn tree_loader_rejects_one_oversized_payload_before_allocating_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let tree = temp.path().join("tree");
        std::fs::create_dir(&tree).unwrap();
        let payload = tree.join("payload.pb");
        let file = File::create(&payload).unwrap();
        file.set_len(MAX_TREE_FILE_BYTES + 1).unwrap();
        let reader = AnchoredDir::open(temp.path()).unwrap();
        assert!(matches!(
            reader.read_pb_tree(Path::new("tree"), 1, u64::MAX),
            Err(AnchoredReadError::TreeByteLimitExceeded { path, limit })
                if path == payload && limit == MAX_TREE_FILE_BYTES
        ));
        assert!(matches!(
            reader.roster_pb_tree(Path::new("tree"), 1, u64::MAX),
            Err(AnchoredReadError::TreeByteLimitExceeded { path, limit })
                if path == payload && limit == MAX_TREE_FILE_BYTES
        ));
    }

    #[test]
    fn strict_tree_loader_rejects_non_protobuf_regular_files() {
        let temp = tempfile::tempdir().unwrap();
        let tree = temp.path().join("tree");
        std::fs::create_dir(&tree).unwrap();
        std::fs::write(tree.join("payload.pb"), b"payload").unwrap();
        std::fs::write(tree.join("junk.txt"), b"unexpected").unwrap();
        let reader = AnchoredDir::open(temp.path()).unwrap();
        assert!(matches!(
            reader.read_pb_tree(Path::new("tree"), 2, 64),
            Err(AnchoredReadError::NonProtobuf { path })
                if path == tree.join("junk.txt")
        ));
        assert!(matches!(
            reader.roster_pb_tree(Path::new("tree"), 2, 64),
            Err(AnchoredReadError::NonProtobuf { path })
                if path == tree.join("junk.txt")
        ));
    }

    #[test]
    fn retained_file_duplicates_exact_read_authority_after_permission_removal() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        std::fs::write(&path, b"retained bytes").unwrap();
        let original = File::open(&path).unwrap();
        let original_fd = u32::try_from(original.as_raw_fd()).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

        let observed = RetainedFile::open(original_fd)
            .and_then(|retained| retained.read_bounded_regular(64))
            .map(AnchoredFile::into_raw);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        assert_eq!(observed.unwrap(), b"retained bytes");
    }

    #[test]
    fn anchored_name_identity_survives_leaf_permission_removal() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        std::fs::write(&path, b"retained bytes").unwrap();
        let base = AnchoredDir::open(temp.path()).unwrap();
        let original = File::open(&path).unwrap();
        let retained = RetainedFile::open(u32::try_from(original.as_raw_fd()).unwrap()).unwrap();
        let expected = retained.read_bounded_regular(64).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

        let observed = base.inspect_regular_identity(Path::new("manifest.json"));
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        assert_eq!(observed.unwrap(), expected.identity());
    }

    #[test]
    fn retained_descriptors_reject_write_and_path_only_modes() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("manifest.json");
        std::fs::write(&path, b"payload").unwrap();

        let write_only = OpenOptions::new().write(true).open(&path).unwrap();
        assert!(matches!(
            RetainedFile::open(u32::try_from(write_only.as_raw_fd()).unwrap()),
            Err(AnchoredReadError::RetainedDescriptorModeInvalid { .. })
        ));

        let read_write = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert!(matches!(
            RetainedFile::open(u32::try_from(read_write.as_raw_fd()).unwrap()),
            Err(AnchoredReadError::RetainedDescriptorModeInvalid { .. })
        ));

        let path_only = open(&path, OFlags::PATH | OFlags::CLOEXEC, Mode::empty()).unwrap();
        assert!(matches!(
            RetainedFile::open(u32::try_from(path_only.as_raw_fd()).unwrap()),
            Err(AnchoredReadError::RetainedDescriptorModeInvalid { .. })
        ));

        let directory_path_only = open(
            temp.path(),
            OFlags::PATH | OFlags::CLOEXEC | OFlags::DIRECTORY,
            Mode::empty(),
        )
        .unwrap();
        assert!(matches!(
            AnchoredDir::open_retained(u32::try_from(directory_path_only.as_raw_fd()).unwrap()),
            Err(AnchoredReadError::RetainedDescriptorModeInvalid { .. })
        ));
    }

    #[test]
    fn invalid_retained_descriptor_is_a_typed_io_error() {
        assert!(matches!(
            RetainedFile::open(i32::MAX as u32),
            Err(AnchoredReadError::RetainedDescriptorUnavailable { source, .. })
                if source.raw_os_error() == Some(libc::EBADF)
        ));
    }

    #[test]
    fn retained_descriptor_implementation_has_no_procfs_path_dependency() {
        assert!(
            !include_str!("anchored.rs").contains(concat!("/proc", "/self/fd")),
            "retained descriptors must be duplicated rather than reopened through procfs"
        );
    }
}
