use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

pub struct Options {
    pub positional: Vec<String>,
    named: HashMap<String, Option<String>>,
}

impl Options {
    const FLAGS: &[&str] = &[
        "no-augment",
        "no-flip",
        "resume",
        "tta",
        "auto-contrast",
        "help",
    ];

    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut opts = Options {
            positional: Vec::new(),
            named: HashMap::new(),
        };
        let mut iter = args.iter();
        while let Some(arg) = iter.next() {
            if arg == "-h" {
                opts.named.insert("help".into(), None);
            } else if let Some(name) = arg.strip_prefix("--") {
                let (key, value) = match name.split_once('=') {
                    Some((key, value)) => (key, Some(value.to_string())),
                    None if Self::FLAGS.contains(&name) => (name, None),
                    None => {
                        let value = iter
                            .next()
                            .ok_or_else(|| format!("--{name} needs a value"))?;
                        (name, Some(value.clone()))
                    }
                };
                opts.named.insert(key.to_string(), value);
            } else {
                opts.positional.push(arg.clone());
            }
        }
        Ok(opts)
    }

    pub fn allow(&self, known: &[&str]) -> Result<(), String> {
        match self
            .named
            .keys()
            .find(|k| *k != "help" && !known.contains(&k.as_str()))
        {
            Some(key) => Err(format!("unknown option --{key} for this command")),
            None => Ok(()),
        }
    }

    pub fn flag(&self, key: &str) -> bool {
        self.named.contains_key(key)
    }

    pub fn string(&self, key: &str, default: &str) -> String {
        self.named
            .get(key)
            .cloned()
            .flatten()
            .unwrap_or_else(|| default.to_string())
    }

    pub fn path(&self, key: &str, default: &str) -> PathBuf {
        PathBuf::from(self.string(key, default))
    }

    pub fn get_opt<T: FromStr>(&self, key: &str) -> Result<Option<T>, String>
    where
        T::Err: std::fmt::Display,
    {
        match self.named.get(key) {
            None => Ok(None),
            Some(None) => Err(format!("--{key} needs a value")),
            Some(Some(v)) => v
                .parse()
                .map(Some)
                .map_err(|e| format!("invalid value `{v}` for --{key}: {e}")),
        }
    }

    pub fn get<T: FromStr>(&self, key: &str, default: T) -> Result<T, String>
    where
        T::Err: std::fmt::Display,
    {
        Ok(self.get_opt(key)?.unwrap_or(default))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Options {
        Options::parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn parses_values_flags_and_positionals() {
        let opts = parse(&["a.ppm", "--epochs", "5", "--lr=0.01", "--tta", "b.ppm"]);
        assert_eq!(opts.positional, ["a.ppm", "b.ppm"]);
        assert_eq!(opts.get::<usize>("epochs", 1), Ok(5));
        assert_eq!(opts.get::<f32>("lr", 1.0), Ok(0.01));
        assert!(opts.flag("tta"));
        assert_eq!(opts.get::<usize>("missing", 7), Ok(7));
    }

    #[test]
    fn reports_bad_input() {
        assert!(Options::parse(&["--epochs".to_string()]).is_err());
        let opts = parse(&["--epochs", "many"]);
        assert!(opts.get::<usize>("epochs", 1).is_err());
        assert!(opts.allow(&["lr"]).is_err());
    }
}
