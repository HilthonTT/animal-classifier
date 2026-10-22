use pawnet::data::cifar::{DatasetKind, Split};
use pawnet::train;

use super::{AppResult, dataset, load_predictor};
use crate::args::Options;

pub fn run(opts: &Options) -> AppResult {
    opts.allow(&["data", "dataset", "model", "threads", "tta"])?;
    let model = load_predictor(opts)?;
    let (kind, data_dir) = dataset(opts, DatasetKind::detect(model.labels()))?;
    let test_set = kind.load(&data_dir, Split::Test)?;
    if model.labels() != test_set.class_names.as_slice() {
        return Err(format!(
            "the model's classes ({}) differ from the dataset's ({})",
            model.labels().join(", "),
            test_set.class_names.join(", ")
        )
        .into());
    }
    let threads = opts.get("threads", train::default_threads())?;
    let tta = opts.flag("tta");
    println!(
        "Evaluating on {} test images{}…\n",
        test_set.len(),
        if tta {
            " with test-time augmentation"
        } else {
            ""
        }
    );
    let matrix = train::evaluate(model.as_ref(), &test_set, threads, tta);
    print!("{}", matrix.render(model.labels()));
    Ok(())
}
