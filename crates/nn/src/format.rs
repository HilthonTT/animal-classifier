use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::layers::{self, LayerKind};

/// Magic bytes at the start of every model file.
const MAGIC: &[u8; 8] = b"ANIMLNN\0";
/// Model file format version written by this crate.
const VERSION: u32 = 2;
/// Oldest version this crate can still read.
const MIN_VERSION: u32 = 1;
