#![feature(collections)]
#![feature(core)]
#![feature(io)]
extern crate getopts;
use std::error::Error;
use std::fs;
use std::io;

mod process_jpeg;

fn print_usage(program: &str, opts: getopts::Options) {
	let brief = format!("Usage: {} -i FILE", program);
	print!("{}", opts.usage(&brief));
}

fn main() {
	let args: Vec<_> = std::env::args().collect();
	let mut opts = getopts::Options::new();
	opts.optopt("i", "input", "Input file name", "FILE")
			.optopt("o", "output", "Output file name", "FILE")
			.optflag("h", "help", "Print this help text");
	let matches = match opts.parse(args.tail()) {
		Ok(m) => m,
		Err(e) => panic!(e.to_string()),
	};
	if matches.opt_present("h") {
		print_usage(&args[0], opts);
		return;
	} else if let (Some(input), Some(output))
			= (matches.opt_str("i"), matches.opt_str("o")) {
		match process_jpeg::process_jpeg(
				&mut io::BufReader::new(fs::File::open(&input).unwrap()),
				&mut io::BufWriter::new(fs::File::create(&output).unwrap())) {
			Ok(()) => println!("Success!"),
			Err(err) => println!("Error: {}", err.description()),
		}
	}
}
