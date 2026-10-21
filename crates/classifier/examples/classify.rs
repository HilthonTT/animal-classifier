use pawnet::image::{Preprocessor, RgbImage};
use pawnet::nn::Network;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, image] = args.as_slice() else {
        return Err("usage: classify <model.bin> <image.ppm|image.bmp>".into());
    };

    let net = Network::load(model)?;
    let image = RgbImage::load(image)?;
    let preprocessor = Preprocessor {
        test_time_augmentation: true,
        ..Default::default()
    };
    let probabilities = preprocessor.predict(&net, &image);

    for (label, p) in net.labels().iter().zip(&probabilities) {
        println!("{label:<8} {:5.1}%", p * 100.0);
    }
    Ok(())
}
