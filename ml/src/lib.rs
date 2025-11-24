#![recursion_limit = "256"]

pub mod dataset;
pub mod inference;
pub mod model;
pub mod training;

pub use dataset::{PageBatch, PageBatcher, PageDataset, PageDatasetConfig, PageSample};
pub use inference::{ModelConfig as PageModelConfig, PageClassifier, PagePrediction};
pub use training::{TrainConfig, TrainingArtifacts, train};
