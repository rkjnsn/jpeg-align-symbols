#![feature(collections)]
#![feature(fs)]
#![feature(io)]
#![feature(path)]
extern crate getopts;
use std::fs;
use std::io;
use std::io::ReadExt;
use std::path;
use std::path::AsPath;

fn print_usage(program: &str, opts: getopts::Options) {
	let brief = format!("Usage: {} -i FILE", program);
	print!("{}", opts.usage(&brief));
}

fn skip_variable<I, E>(input: &mut I)
		where I: Iterator<Item = Result<u8, E>>, E: std::fmt::Debug {
	let size = ((input.next().expect("Unexpected EOF").unwrap() as u16) << 8)
			+ input.next().expect("Unexpected EOF").unwrap() as u16;
	for byte in input.take((size - 2) as usize) {
		byte.unwrap();
	}
}

fn process_jpeg(infile: &path::Path) {
	let mut input = io::BufReader::new(fs::File::open(infile).unwrap())
			.bytes().peekable();


	'outer: loop {
		assert!(input.next().expect("Unexpected EOF").unwrap() == 0xFF,
				"JPEG invalid or corrupt");
		let mut byte = input.next().expect("Unexpect EOF").unwrap();
		loop{
			match byte {
				0xD8 => {
					println!("Start of Image");
				}
				0xC0 => {
					println!("Start of Frame (Baseline DCT)");
					skip_variable(&mut input);
				}
				0xC2 => {
					println!("Start of Frame (Progressive DCT)");
					panic!("Progessive images not supported");
				}
				0xC4 => {
					println!("Define Huffman Table(s)");
					skip_variable(&mut input);
				}
				0xDB => {
					println!("Define Quantization Table(s)");
					skip_variable(&mut input);
				}
				0xDD => {
					println!("Define Restart Interval");
					panic!("Restarts not supported");
				}
				0xDA => {
					println!("Start of Scan");
					skip_variable(&mut input);
					loop {
						while input.next().expect("Unexpect EOF").unwrap()
							!= 0xFF {}
						byte = input.next().expect("Unexpect EOF").unwrap();
						match byte {
							0x00 => continue,
							0xD0...0xD7 => panic!("Unexpected restart"),
							_ => break
						}
					}
					continue;
				}
				0xE0...0xEF => {
					println!("App Marker: {}", byte & 0x0F);
					skip_variable(&mut input);
				}
				0xFE => {
					println!("Comment");
					skip_variable(&mut input);
				}
				0xD9 => {
					println!("End of Image");
					if let Some(_) = input.next() {
						println!("Warning: Trailing data present");
					}
					break 'outer;
				}
				0xFF => { // Padding
					byte = input.next().expect("Unexpect EOF").unwrap();
					continue;
				}
				_ => panic!("Unexpected segment: {}", byte)
			}
			break;
		}
	}
}

fn main() {
	let args: Vec<_> = std::env::args().collect();
	let mut opts = getopts::Options::new();
	opts.optopt("i", "input", "Input file name", "FILE")
			.optflag("h", "help", "Print this help text");
	let matches = match opts.parse(args.tail()) {
		Ok(m) => m,
		Err(e) => panic!(e.to_string()),
	};
	if matches.opt_present("h") {
		print_usage(&args[0], opts);
		return;
	} else if let Some(input) = matches.opt_str("i") {
		process_jpeg(input.as_path());
	}
}
