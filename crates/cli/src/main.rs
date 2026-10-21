mod args;
mod commands;

use std::process::ExitCode;

use args::Options;

const USAGE: &str = "\
pawnet — a from-scratch CNN that recognises animals (pure Rust, no dependencies)

USAGE:
    pawnet <COMMAND> [OPTIONS]

COMMANDS:
    train             Train a new model on the CIFAR-10 animal images
    evaluate          Measure accuracy on the test set and print a confusion matrix
    predict <IMAGE>…  Classify your own .png, .ppm or .bmp images
    expand            Write a bigger training set of flipped, shifted and rotated copies
    export-samples    Save some test images as .ppm files to try `predict` on
    info              Show the network architecture

COMMON OPTIONS:
    --data <DIR>      CIFAR-10 binary folder   [default: data/cifar-10-batches-bin]
    --model <PATHS>   Model file, folder of models, or comma-separated list. Several
                      models form an ensemble that averages their predictions.
                      [default: models/pawnet.bin]
    --threads <N>     Worker threads           [default: all cores]

TRAIN OPTIONS:
    --epochs <N>            [default: 20]
    --batch-size <N>        [default: 64]
    --lr <RATE>             Initial learning rate               [default: 0.001]
    --schedule <SPEC>       constant | exp:<decay> | step:<factor>:<every> | time:<decay>
                                                                [default: exp:0.9]
    --optimizer <NAME>      adam | sgd                          [default: adam]
    --loss <NAME>           cross-entropy | mse                 [default: cross-entropy]
    --l2 <LAMBDA>           L2 regularisation (weight decay)    [default: 0]
    --clip <NORM>           Gradient clipping by global norm    [default: off]
    --activation <NAME>     relu | leaky-relu | elu | sigmoid | tanh | softplus [default: relu]
    --init <NAME>           he | xavier | uniform | zero        [default: he]
    --dropout <RATE>        Dropout before the output layer     [default: 0.25]
    --width <N>             Filters in the first conv layer     [default: 16]
    --seed <N>              Random seed for reproducible runs
    --limit <N>             Only use the first N training images (quick experiments)
    --no-augment            Disable random flips / shifts during training
    --resume                Continue training the model at --model
    --val <FRACTION>        Hold out this share of the training images to choose
                            the best epoch (0 = choose by test accuracy) [default: 0.1]
    --log <FILE>            Write per-epoch loss and accuracy as CSV

EVALUATE / PREDICT OPTIONS:
    --tta                   Test-time augmentation: also classify the mirror image
    --fit <MODE>            predict only: crop | pad | stretch  [default: crop]
    --auto-contrast         predict only: stretch the brightness range first

EXPAND OPTIONS:
    --out <DIR>             [default: data/cifar-10-expanded]
    --shift <PIXELS>        Shifted copy (0 = none)             [default: 2]
    --rotate <DEGREES>      Rotated copy (0 = none)             [default: 12]
    --no-flip               No mirrored copy

EXPORT-SAMPLES OPTIONS:
    --out <DIR>             [default: samples]
    --count <N>             [default: 12]
    --scale <N>             Upscale factor for easier viewing   [default: 4]

Get the dataset first with scripts/download_data.sh (Linux/macOS) or
scripts\\download_data.ps1 (Windows).
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> commands::AppResult {
    let Some((command, rest)) = args.split_first() else {
        print!("{USAGE}");
        return Ok(());
    };
    const COMMANDS: &[&str] = &[
        "train",
        "evaluate",
        "eval",
        "predict",
        "expand",
        "export-samples",
        "info",
        "help",
        "-h",
        "--help",
    ];
    if !COMMANDS.contains(&command.as_str()) {
        return Err(format!("unknown command `{command}` (run without arguments for help)").into());
    }
    let opts = Options::parse(rest)?;
    if opts.flag("help") {
        print!("{USAGE}");
        return Ok(());
    }
    match command.as_str() {
        "train" => commands::train::run(&opts),
        "evaluate" | "eval" => commands::evaluate::run(&opts),
        "predict" => commands::predict::run(&opts),
        "expand" => commands::expand::run(&opts),
        "export-samples" => commands::export_samples::run(&opts),
        "info" => commands::info::run(&opts),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command `{other}` (run without arguments for help)").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_command_is_reported_before_option_errors() {
        let args: Vec<String> = ["trian", "--epochs"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let err = run(&args).unwrap_err().to_string();
        assert!(err.contains("unknown command `trian`"), "{err}");
    }
}
