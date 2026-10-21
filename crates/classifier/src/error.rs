use std::fmt;
use std::io;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Format(String),
}

impl Error {
    pub(crate) fn format(message: impl Into<String>) -> Self {
        Error::Format(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Format(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Format(_) => None,
        }
    }
}

impl From<pawnet_nn::Error> for Error {
    fn from(e: pawnet_nn::Error) -> Self {
        match e {
            pawnet_nn::Error::Io(e) => Error::Io(e),
            pawnet_nn::Error::Format(message) => Error::Format(message),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
