pub mod error;
mod format;
pub mod init;
pub mod layers;
pub mod loss;
mod math;
mod network;
pub mod optimizer;
pub mod rng;
pub mod schedule;

pub use error::{Error, Result};
pub use init::Init;
pub use layers::{Activation, ActivationFn, Conv2d, Dense, Dropout, Layer, LayerKind, MaxPool2d};
pub use loss::Loss;
pub use network::Network;
pub use optimizer::{Adam, Optimizer, Sgd};
pub use rng::Rng;
pub use schedule::LrSchedule;
