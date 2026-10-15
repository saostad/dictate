mod model;

fn main() -> anyhow::Result<()> {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
    Ok(())
}

fn run() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        eprintln!("{}", usage());
        std::process::exit(2);
    };
    match command.as_str() {
        "download" => model::download(),
        "transcribe" => anyhow::bail!("the transcribe command is not implemented yet"),
        "help" | "--help" => {
            println!("{}", usage());
            Ok(())
        }
        unknown => {
            eprintln!("Unknown command \"{unknown}\". Run \"dictate help\" for usage.");
            std::process::exit(2);
        }
    }
}

fn usage() -> &'static str {
    "Usage: dictate <command> [args]\n\nCommands:\n  download               Download the speech model (about 660 MB)\n  transcribe <file.wav>  Transcribe a WAV file and print the text\n  help                   Show this help"
}
