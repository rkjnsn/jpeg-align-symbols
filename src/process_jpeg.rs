extern crate byteorder;
use std::cmp;
use std::error;
use std::fmt;
use std::io;
use std::io::Read;
use std::io::Write;
use std::iter::AdditiveIterator;
use std::num::ToPrimitive;
use self::byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

#[derive(Debug)]
enum Error {
	Io(io::Error),
	Parse(ParseError),
	Unsupported(UnsupportedError),
}

#[derive(Debug)]
enum ParseError {
	UnexpectedEof,
	Syntax,
}

impl ParseError {
	fn err<V>(self) -> Result<V, Error> {
		Err(Error::Parse(self))
	}
}

#[derive(Debug)]
enum UnsupportedError {
	Progressive,
	Restart,
	SpectralSelection,
}

impl UnsupportedError {
	fn err<V>(self) -> Result<V, Error> {
		Err(Error::Unsupported(self))
	}
}

impl fmt::Display for Error {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		fmt.write_str(error::Error::description(self))
	}
}

impl error::Error for Error {
	fn description(&self) -> &str {
		match *self {
			Error::Io(ref err) => error::Error::description(err),
			Error::Parse(ref err) => match *err {
				ParseError::UnexpectedEof => "Unexpected End of File",
				ParseError::Syntax => "Invalid or Corrupt JPEG",
			},
			Error::Unsupported(ref err) => match *err {
				UnsupportedError::Progressive
						=> "Progressive JPEGs are not supported",
				UnsupportedError::Restart
						=> "JPEGs with restarts are not currently supported",
				UnsupportedError::SpectralSelection
						=> "Non-standard spectral selection not supported",
			}
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			Error::Io(ref err) => Some(err),
			_ => None,
		}
	}
}

impl error::FromError<io::Error> for Error {
	fn from_error(err: io::Error) -> Error {
		Error::Io(err)
	}
}

impl error::FromError<byteorder::Error> for Error {
	fn from_error(err: byteorder::Error) -> Error {
		match err {
			byteorder::Error::UnexpectedEOF
					=> Error::Parse(ParseError::UnexpectedEof),
			byteorder::Error::Io(err) => Error::Io(err),
		}
	}
}

struct SplitIter<'a, T: 'a, I: 'a + ToPrimitive> {
	remaining: Option<&'a [T]>,
	indexes: &'a [I],
	offset: usize,
}

impl<'a, T: 'a, I: 'a + ToPrimitive> SplitIter<'a, T, I> {
	fn new(slice: &'a [T], indexes: &'a [I]) -> Self {
		SplitIter { remaining: Some(slice), indexes: indexes, offset: 0 }
	}
}

impl<'a, T: 'a, I: 'a + ToPrimitive> Iterator for SplitIter<'a, T, I> {
	type Item = &'a [T];

	fn next(&mut self) -> Option<&'a [T]> {
		match self.remaining.take() {
			Some(remaining) => {
				match self.indexes.first() {
					Some(index) => {
						let mid = index.to_usize().unwrap() - self.offset;
						let (left, right) = remaining.split_at(mid);
						self.remaining = Some(right);
						self.indexes = self.indexes.tail();
						self.offset += mid;
						Some(left)
					}
					None => Some(remaining),
				}
			}
			None => None,
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let size = if self.remaining.is_some() {
				self.indexes.len() + 1 } else { 0 };
		(size, Some(size))
	}
}

// JPEG specific bit reader handling bit packing
struct BitReader<R: Read> {
	inner: R,
	byte: u8,
	offset: u8,
}

impl<R: Read> BitReader<R> {
	fn new(reader: R) -> Self {
		BitReader { inner: reader, byte: 0, offset: 8 }
	}

	fn read_bit(&mut self) -> Result<bool, Error> {
		if self.offset >= 8 {
			self.offset = 0;
			self.byte = try!(self.inner.read_u8());
			
			// 0xFF bytes in the bit stream must be followed by a 0x00, which
			// we ignore.
			if self.byte == 0xFF && try!(self.inner.read_u8()) != 0x00 {
				return ParseError::Syntax.err();
			}
		}
		let bit = self.byte & 0x80 >> self.offset != 0;
		self.offset += 1;
		Ok(bit)
	}
}

// JPEG specific bit writer handling bit packing. Note: you must call `finish`
// to write any partial final byte.
struct BitWriter<W: Write> {
	inner: W,
	byte: u8,
	offset: u8,
}

impl<W: Write> BitWriter<W> {
	fn new(writer: W) -> Self {
		BitWriter { inner: writer, byte: 0, offset: 0 }
	}

	fn write_bit(&mut self, bit: bool) -> Result<(), Error> {
		if bit { self.byte |= 0x80 >> self.offset }
		self.offset += 1;
		if self.offset >= 8 {
			try!(self.inner.write_u8(self.byte));
			// 0xFF byte in the bit stream must be followed by 0x00
			if self.byte == 0xFF { try!(self.inner.write_u8(0x00)); }
			self.byte = 0;
			self.offset = 0;
		}
		Ok(())
	}

	fn finish(mut self) -> Result<W, Error> {
		if self.offset != 0 {
			try!(self.inner.write_u8(self.byte));
		}
		Ok(self.inner)
	}
}

const MAX_HUFF_CODE_LEN: u8 = 16;
const MAX_HUFF_SYMBOLS: u16 = 256;

struct Symbol {
	symbol: u8,
	data: u16,
}

struct Component {
	id: u8,
	sample_factors: [u8; 2], // Horizontal, Vertical
	quant_table_id: u8,
	dc_huff_table_id: u8,
	ac_huff_table_id: u8,
	block_offsets: Vec<u32>,
	symbols: Vec<Symbol>,
}

#[derive(Eq)]
#[derive(PartialEq)]
enum CoefficientType {
	DC,
	AC,
}

struct HuffmanTable {
	id: u8,
	coef_type: CoefficientType,
	counts: [u8; MAX_HUFF_CODE_LEN as usize],
	symbols: [u8; MAX_HUFF_SYMBOLS as usize],
}

// Encode and decode symbols. Do we want to use a more efficent data structure
// for this?
impl HuffmanTable {
	fn decode_symbol<R: Read>(&self, input: &mut BitReader<R>)
			-> Result<u8, Error> {
		let mut offset = 0;
		let mut symbols: &[u8] = &self.symbols;
		for &count in &self.counts {
			let (left, right) = symbols.split_at(count as usize);
			let val = try!(input.read_bit()) as usize + offset;
			if let Some(&symbol) = left.get(val) { return Ok(symbol); }
			offset = (val - count as usize) * 2;
			symbols = right;
		}
		ParseError::Syntax.err() // Not a valid code
	}

	/*fn encode_symbol<W: Write>(&self, symbol: u8, output: &mut BitWriter<W>)
			-> Result<(), Error> {
		let mut offset = 0;
		let mut symbols: &[u8] = &self.symbols;
		for (level, &count) in (0..).zip(self.counts.iter()) {
			let (left, right) = symbols.split_at(count as usize);
			if let Some(index) = left.position_elem(&symbol) {
				return JpegData::write_bits(offset + index as u16,
						level + 1, output);
			}
			offset = (offset + count as u16) * 2;
			symbols = right;
		}
		ParseError::Syntax.err() // Not a valid symbol
	}*/
	fn encode_symbol<W: Write>(&self, symbol: u8, output: &mut BitWriter<W>)
			-> Result<(), Error> {
		if let Some(mut index) = self.symbols.position_elem(&symbol) {
			let mut offset = 0;
			for (level, &count) in (0..).zip(self.counts.iter()) {
				if index < count as usize {
					return JpegData::write_bits(offset + index as u16,
							level + 1, output);
				}
				offset = (offset + count as u16) * 2;
				index -= count as usize;
			}
		}
		ParseError::Syntax.err() // Not a valid symbol
	}
}

struct QuantTable {
	id: u8,
	precision: u8,
	data: Vec<u8>,
}

struct AppSegment {
	id: u8,
	data: Vec<u8>,
}

struct JpegData {
	width: u16,
	height: u16,
	precision: u8,
	components: Vec<Component>,
	huff_tables: Vec<HuffmanTable>,
	quant_tables: Vec<QuantTable>,
	app_segments: Vec<AppSegment>,
}

impl JpegData {
	fn new() -> JpegData {
		JpegData { width: 0, height: 0, precision: 0, components: vec![],
				huff_tables: vec![], quant_tables: vec![], app_segments: vec![] }
	}

	// TODO: Make sure there's only one of these
	fn parse_sof<R: io::Read + ?Sized>(&mut self, input: &mut R)
			-> Result<(), Error> {
		println!("Start of Frame (Baseline DCT)");

		self.precision = try!(input.read_u8());
		self.height = try!(input.read_u16::<BigEndian>());
		self.width = try!(input.read_u16::<BigEndian>());

		let num_components = try!(input.read_u8());
		self.components.reserve_exact(num_components as usize);
		for _ in 0..num_components {
			self.components.push(Component {
					id: try!(input.read_u8()),
					sample_factors: { let f = try!(input.read_u8());
						[f >> 4, f & 0xF] },
					quant_table_id: try!(input.read_u8()),
					// The rest is filled in by scans
					dc_huff_table_id: 0, ac_huff_table_id: 0,
					block_offsets: vec![], symbols: vec![] } );
		}

		Ok(())
	}

	// TODO: Make sure there are no duplicate table definitions
	fn parse_huff<R: io::Read + ?Sized>(&mut self, input: &mut R)
			-> Result<(), Error> {
		println!("Define Huffman Table(s)");

		loop {
			let info;
			match input.read_u8() {
				Ok(byte) => info = byte,
				Err(byteorder::Error::UnexpectedEOF) => break,
				Err(byteorder::Error::Io(err)) => return Err(Error::Io(err)),
			}
			let id = info & 0xF;
			let coef_type = if info & 0x10 == 0 {CoefficientType::DC}
					else {CoefficientType::AC};

			let mut counts = [0; MAX_HUFF_CODE_LEN as usize];
			try!(fill_buffer(input, &mut counts));

			let num_symbols = counts.iter().map(|&x| x as u16).sum();
			if num_symbols > MAX_HUFF_SYMBOLS {
					return ParseError::Syntax.err(); }

			let mut symbols = [0; MAX_HUFF_SYMBOLS as usize];
			try!(fill_buffer(input, &mut symbols[..num_symbols as usize]));

			self.huff_tables.push(HuffmanTable { id: id, coef_type: coef_type,
					counts: counts, symbols: symbols } );
		}

		Ok(())
	}

	fn parse_quant<R: io::Read + ?Sized>(&mut self, input: &mut R)
			-> Result<(), Error> {
		println!("Define Quantization Table(s)");

		loop {
			let info;
			match input.read_u8() {
				Ok(byte) => info = byte,
				Err(byteorder::Error::UnexpectedEOF) => break,
				Err(byteorder::Error::Io(err)) => return Err(Error::Io(err)),
			}
			let id = info & 0xF;
			let precision: u8 = if info >> 4 == 0 {8} else {16};
			let len = 64 + (precision as u16 + 1);

			let mut data = Vec::with_capacity(len as usize);
			try!(input.take(len as u64).read_to_end(&mut data));

			self.quant_tables.push(QuantTable { id: id, precision: precision,
					data: data } );
		}

		Ok(())
	}

	fn parse_app<R: io::Read + ?Sized>(&mut self, app_id: u8, input: &mut R)
			-> Result<(), Error> {
		println!("App Marker: {}", app_id);
		let mut data = vec![];
		try!(input.read_to_end(&mut data));
		self.app_segments.push(AppSegment { id: app_id, data: data });
		Ok(())
	}

	// Don't take self so whole struct isn't borrowed
	fn get_component_mut(components: &mut Vec<Component>, component_id: u8)
			-> Result<&mut Component, Error> {
		match components.iter_mut().find(|c| c.id == component_id) {
			Some(c) => Ok(c),
			None => ParseError::Syntax.err(),
		}
	}

	fn parse_scan<R: io::Read + ?Sized>(&mut self, input: &mut R)
			-> Result<Vec<u8>, Error> {
		println!("Start of Scan");

		let num_components = try!(input.read_u8());
		let mut component_ids = Vec::with_capacity(num_components as usize);

		for _ in 0..num_components {
			let component_id = try!(input.read_u8());
			let component = try!(JpegData::get_component_mut(
					&mut self.components, component_id));
			let tables = try!(input.read_u8());
			component.dc_huff_table_id = tables >> 4;
			component.ac_huff_table_id = tables & 0xF;
			component_ids.push(component_id);
		}

		// The following seem to be fixed and don't need to be stored
		if try!(input.read_u8()) != 0 || try!(input.read_u8()) != 63
				|| try!(input.read_u8()) != 0 {
			return UnsupportedError::SpectralSelection.err();
		}

		Ok(component_ids)
	}

	fn get_mcu_size(&self) -> [u16; 2] {
		let max_sample_factors = self.components.iter().fold([0, 0],
				|m, c| [cmp::max(m[0], c.sample_factors[0]),
					cmp::max(m[1], c.sample_factors[1])]);
		[max_sample_factors[0] as u16 * 8, max_sample_factors[1] as u16 * 8]
	}

	fn get_mcu_dimensions(&self) -> [u16; 2] {
		let mcu_size = self.get_mcu_size();
		[(self.width - 1) / mcu_size[0] + 1,
				(self.height - 1) / mcu_size[1] + 1]
	}

	fn get_huff_table(huff_tables: &Vec<HuffmanTable>, table_id: u8,
			coef_type: CoefficientType) -> Result<&HuffmanTable, Error> {
		match huff_tables.iter().find(
				|d| d.id == table_id && d.coef_type == coef_type) {
			Some(c) => Ok(c),
			None => ParseError::Syntax.err(),
		}
	}

	fn read_bits<R: Read>(count: u8, input: &mut BitReader<R>)
			-> Result<u16, Error> {
		let mut val = 0;
		for _ in 0..count {
			val <<= 1;
			val |= try!(input.read_bit()) as u16
		}
		Ok(val)
	}

	// Only used for debugging
	#[allow(dead_code)]
	fn decode_bits(num: u8, bits: u16) -> i16 {
		if num == 0 { return 0; }
		let negative;
		let pos_bits;

		if bits >> num - 1 == 0 {
			negative = true;
			pos_bits = bits ^ (0x1 << num) - 1;
		} else {
			negative = false;
			pos_bits = bits;
		}

		if negative {
			-(pos_bits as i16)
		} else {
			pos_bits as i16
		}
	}

	fn parse_data<R: io::Read + ?Sized>(&mut self, component_ids: Vec<u8>,
			input: &mut R) -> Result<(), Error> {
		println!("Scan Data");

		let mut bit_reader = BitReader::new(input);
		let mcu_dimensions = self.get_mcu_dimensions();

		struct ComponentData<'a> {
			component: &'a mut Component,
			dc_table: &'a HuffmanTable,
			ac_table: &'a HuffmanTable,
			sample_factor: u8,
			order: u8,
		}
		let mut components = vec![];

		// Gets a reference to each needed component in the needed order (along
		// with looking up the relevant huffman trees).
		// TODO: Is there a better way to do this?
		for component in &mut self.components {
			if let Some(position) = component_ids.position_elem(&component.id) {
				let dc_id = component.dc_huff_table_id;
				let ac_id = component.ac_huff_table_id;
				let sample_factor = component.sample_factors[0]
						* component.sample_factors[1];
				components.push(ComponentData { component: component,
						dc_table: try!(JpegData::get_huff_table(
							&self.huff_tables, dc_id, CoefficientType::DC)),
						ac_table: try!(JpegData::get_huff_table(
							&self.huff_tables, ac_id, CoefficientType::AC)),
						sample_factor: sample_factor, order: position as u8 } );
			}
		}
		if components.len() != component_ids.len() {
			// Some weren't found or there were duplicates
			return ParseError::Syntax.err();
		}
		components.sort_by(|a, b| a.order.cmp(&b.order));

		let num_mcus = mcu_dimensions[0] as u32 * mcu_dimensions[1] as u32;

		for _ in 0..num_mcus {
			for component in &mut components {
				for _ in 0..component.sample_factor {
					component.component.block_offsets.push(
							component.component.symbols.len() as u32);

					let dc_symbol = try!(component.dc_table.decode_symbol(
							&mut bit_reader));
					let dc_bits = try!(JpegData::read_bits(dc_symbol & 0xF,
							&mut bit_reader));
					component.component.symbols.push(
							Symbol { symbol: dc_symbol, data: dc_bits });

					//println!("\nDC Symbol: {:X}, Bits {:X}, Coef: {}", dc_symbol, dc_bits,
					//		JpegData::decode_bits(dc_symbol & 0xF, dc_bits));

					let mut read_acs = 0;
					while read_acs < 63 {
						let ac_symbol = try!(component.ac_table.decode_symbol(
								&mut bit_reader));
						let ac_bits = try!(JpegData::read_bits(ac_symbol & 0xF,
								&mut bit_reader));
						//println!("AC Symbol: {:X}, Bits {:X}, Coef: {}", ac_symbol, ac_bits,
						//		JpegData::decode_bits(ac_symbol & 0xF, ac_bits));
						component.component.symbols.push(
								Symbol { symbol: ac_symbol, data: ac_bits });
						if ac_symbol == 0x00 { break; }
						read_acs += (ac_symbol >> 4) + 1;
					}
				}
			}
		}

		Ok(())
	}

	fn parse_segment<R: io::Read + ?Sized>(&mut self, segment_id: u8,
			input: &mut R) -> Result<(), Error> {
		let length = try!(input.read_u16::<BigEndian>());
		if length < 2 { return ParseError::Syntax.err(); }

		// This can probably be improved with futer finer-grained lifetimes
		let mut scan_components = vec![];
		{
			let mut segment_data = input.take((length - 2) as u64);

			// TODO: Make a too-short segment a syntax error, not an eof error
			match segment_id {
				0xC0 => try!(self.parse_sof(&mut segment_data)),
				0xC4 => try!(self.parse_huff(&mut segment_data)),
				0xDA => scan_components = try!(
						self.parse_scan(&mut segment_data)),
				0xDB => try!(self.parse_quant(&mut segment_data)),
				0xE0...0xEF => try!(self.parse_app(segment_id & 0xF,
						&mut segment_data)),
				0xFE => println!("Comment"),
				_ => unreachable!(),
			}
			if segment_data.limit() != 0 { // Unread segment data
				return ParseError::Syntax.err();
			}
		}
		if segment_id == 0xDA { // Read scan data
			try!(self.parse_data(scan_components, input));
		}

		Ok(())
	}

	fn read_jpeg<R: io::Read + ?Sized>(input: &mut R) -> Result<Self, Error> {
		println!("Reading...");
		let mut jpeg_data = JpegData::new();

		'outer: loop {
			if try!(input.read_u8()) != 0xFF {
				return ParseError::Syntax.err();
			}

			loop {
				let segment_id = try!(input.read_u8());
			
				match segment_id {
					0xD8 => println!("Start of Image"),
					0xC2 => return UnsupportedError::Progressive.err(),
					0xDD => return UnsupportedError::Restart.err(),
					0xC0 | 0xC4 | 0xDA | 0xDB | 0xE0...0xEF | 0xFE
							=> try!(jpeg_data.parse_segment(segment_id, input)),
					0xD9 => { // End of Image
						println!("End of Image");
						break 'outer;
					}
					0xFF => continue, // Padding byte
					_ => return ParseError::Syntax.err(),
				}
				break;
			}
		}
		Ok(jpeg_data)
	}

	fn write_apps<W: io::Write + ?Sized>(&self, output: &mut W)
			-> Result<(), Error> {
		println!("App Segments");
		for app in &self.app_segments {
			try!(output.write_u16::<BigEndian>(0xFFE0 + app.id as u16));
			try!(output.write_u16::<BigEndian>(2 + app.data.len() as u16));
			try!(output.write_all(&app.data));
		}
		Ok(())
	}

	fn write_sof<W: io::Write + ?Sized>(&self, output: &mut W)
			-> Result<(), Error> {
		println!("Start of Frame");
		try!(output.write_u16::<BigEndian>(0xFFC0));
		try!(output.write_u16::<BigEndian>(
				8 + 3 * self.components.len() as u16));
		try!(output.write_u8(self.precision));
		try!(output.write_u16::<BigEndian>(self.height));
		try!(output.write_u16::<BigEndian>(self.width));
		try!(output.write_u8(self.components.len() as u8));
		for component in &self.components {
			try!(output.write_u8(component.id));
			try!(output.write_u8(component.sample_factors[0] << 4
					| component.sample_factors[1]));
			try!(output.write_u8(component.quant_table_id));
		}

		Ok(())
	}

	fn write_quant<W: io::Write + ?Sized>(&self, output: &mut W)
			-> Result<(), Error> {
		println!("Quantization Tables");
		try!(output.write_u16::<BigEndian>(0xFFDB));
		try!(output.write_u16::<BigEndian>(2 + self.quant_tables.iter().map(
				|q| 1 + q.data.len() as u16).sum()));
		for quant in &self.quant_tables {
			try!(output.write_u8(((quant.precision == 16) as u8) << 4
					| quant.id));
			try!(output.write_all(&quant.data));
		}

		Ok(())
	}

	fn write_huff<W: io::Write + ?Sized>(&self, output: &mut W)
			-> Result<(), Error> {
		println!("Huffman Tables");
		try!(output.write_u16::<BigEndian>(0xFFC4));
		try!(output.write_u16::<BigEndian>(2 + self.huff_tables.iter().map(
				|h| 1 + 16 + h.counts.iter().map(|&c| c as u16).sum()).sum()));
		for huff in &self.huff_tables {
			try!(output.write_u8(match huff.coef_type { CoefficientType::DC
					=> 0x00, CoefficientType::AC => 0x10 } | huff.id));
			try!(output.write_all(&huff.counts));
			try!(output.write_all(&huff.symbols[0..huff.counts.iter()
					.map(|&c| c as usize).sum()]));
		}

		Ok(())
	}

	fn write_scan<W: io::Write + ?Sized>(&self, output: &mut W)
			-> Result<(), Error> {
		println!("Start of Scan");
		try!(output.write_u16::<BigEndian>(0xFFDA));
		try!(output.write_u16::<BigEndian>(
				6 + 2 * self.components.len() as u16));
		try!(output.write_u8(self.components.len() as u8));
		for component in &self.components {
			try!(output.write_u8(component.id));
			try!(output.write_u8(component.dc_huff_table_id << 4
					| component.ac_huff_table_id));
		}
		try!(output.write_all(&[0x00u8, 0x3F, 0x00]));

		Ok(())
	}

	fn write_bits<W: Write>(bits: u16, mut count: u8, output: &mut BitWriter<W>)
			-> Result<(), Error> {
		while count > 0 {
			count -= 1;
			try!(output.write_bit(bits & 0x1 << count != 0));
		}
		Ok(())
	}

	fn write_data<W: io::Write + ?Sized>(&self, output: &mut W)
			-> Result<(), Error> {
		println!("Scan Data");

		let mut bit_writer = BitWriter::new(output);

		let mcu_dimensions = self.get_mcu_dimensions();
		let num_mcus = mcu_dimensions[0] as u32 * mcu_dimensions[1] as u32;

		struct ComponentData<'a> {
			slice_iter: SplitIter<'a, Symbol, u32>,
			dc_table: &'a HuffmanTable,
			ac_table: &'a HuffmanTable,
			sample_factor: u8,
		}

		let mut components = vec![];

		for component in &self.components {
			components.push(ComponentData {
					slice_iter: SplitIter::new(&component.symbols,
						&component.block_offsets[1..]),
					dc_table: try!(JpegData::get_huff_table(&self.huff_tables,
						component.dc_huff_table_id, CoefficientType::DC)),
					ac_table: try!(JpegData::get_huff_table(&self.huff_tables,
						component.ac_huff_table_id, CoefficientType::AC)),
					sample_factor: component.sample_factors[0]
						* component.sample_factors[1] } );
		}

		for _ in 0..num_mcus {
			for component in &mut components {
				for _ in 0..component.sample_factor {
					let block = component.slice_iter.next().unwrap();

					let dc = block.first().unwrap();
					try!(component.dc_table.encode_symbol(dc.symbol,
							&mut bit_writer));
					try!(JpegData::write_bits(dc.data, dc.symbol & 0xF,
							&mut bit_writer));
					//println!("\nDC Symbol: {:X}, Bits {:X}, Coef: {}", dc.symbol, dc.data,
					//		JpegData::decode_bits(dc.symbol & 0xF, dc.data));

					for ac in block.tail() {
						try!(component.ac_table.encode_symbol(ac.symbol,
								&mut bit_writer));
						try!(JpegData::write_bits(ac.data, ac.symbol & 0xF,
								&mut bit_writer));
						//println!("AC Symbol: {:X}, Bits {:X}, Coef: {}", ac.symbol, ac.data,
						//		JpegData::decode_bits(ac.symbol & 0xF, ac.data));
					}
				}
			}
		}

		try!(bit_writer.finish());

		Ok(())
	}

	fn write_jpeg<W: io::Write + ?Sized>(&self, output: &mut W)
			-> Result<(), Error> {
		println!("Writing...");
		try!(output.write_u16::<BigEndian>(0xFFD8));
		try!(self.write_apps(output));
		try!(self.write_sof(output));
		try!(self.write_quant(output));
		try!(self.write_huff(output));
		try!(self.write_scan(output));
		try!(self.write_data(output));
		try!(output.write_u16::<BigEndian>(0xFFD9));

		Ok(())
	}

	// Calculate an aligned huffman table as compact as possible
	fn calc_huffman_tables<'a, I: Iterator<Item = &'a [Symbol]>>(
			iter: &'a mut I) -> (HuffmanTable, HuffmanTable) {
		let mut dc_counts = [0u64; 256];
		let mut ac_counts = [0u64; 256];

		for slice in iter {
			dc_counts[slice.first().unwrap().symbol as usize] += 1;
			for symbol in slice.tail() {
				ac_counts[symbol.symbol as usize] +=1;
			}
		}

		struct SymbolCount {
			symbol: u8,
			length: u8,
			count: u64,
		}

		let mut dc_table = HuffmanTable { id: 0, coef_type: CoefficientType::DC,
				counts: [0; 16], symbols: [0; 256] };
		let mut ac_table = HuffmanTable { id: 0, coef_type: CoefficientType::AC,
				counts: [0; 16], symbols: [0; 256] };

		fn encoding_space_cost(length: u8) -> u16 {
			0x1 << 16 - length as u16
		}

		for &mut(counts, &mut ref mut table) in [(&dc_counts, &mut dc_table),
				(&ac_counts, &mut ac_table)].iter_mut() {
			let mut vec = vec![];

			// Maximum pontential space left in the huffman tree
			let mut encoding_space = 0x1u32 << 16;

			// Convert counts into a vec using largest aligned encoding and
			// update available encoding space
			for (symbol, &count) in counts.iter().enumerate()
					.filter(|&(_, &c)| c != 0) {
				let length = 16 - symbol as u8 % 8;
				encoding_space -= encoding_space_cost(length) as u32;
				vec.push(SymbolCount { symbol: symbol as u8,
						length: length, count: count });
			}

			// Knapsack Problem
			// Table is over 40 MiB. Maybe try meet-in-the-middle?
			let mut max_counts = vec![0; (encoding_space as usize + 1)
					* (vec.len() + 1)];
			{
				let mut row_iter = max_counts.chunks_mut(
						encoding_space as usize + 1);
				let mut prev_row = row_iter.next().unwrap();
				for (mut row, sym) in row_iter.zip(vec.iter()) {
					let cost = encoding_space_cost(sym.length - 8)
							- encoding_space_cost(sym.length);
					for j in 0..encoding_space as usize + 1 {
						row[j] = if cost as usize <= j {
							cmp::max(prev_row[j],
									prev_row[j - cost as usize] + sym.count)
						} else {
							prev_row[j]
						}
					}
					prev_row = row;
				}
			}

			{
				let mut row_iter = max_counts.chunks_mut(
						encoding_space as usize + 1).rev();
				let mut row = row_iter.next().unwrap();
				let mut j = encoding_space as usize;
				for (mut prev_row, sym) in row_iter.zip(vec.iter_mut().rev()) {
					if row[j] != prev_row[j] {
						j -= (encoding_space_cost(sym.length - 8)
							- encoding_space_cost(sym.length)) as usize;
						sym.length -= 8;
					}
					row = prev_row;
				}
			}

			// Resort by ascending length
			vec.sort_by(|a, b| a.length.cmp(&b.length));

			// And finally write to output structure
			for (table_v, vec_v) in table.symbols.iter_mut().zip(vec.iter()) {
				*table_v = vec_v.symbol;
				table.counts[vec_v.length as usize - 1] += 1;
			}
			// println!("{:?}", table.counts.as_slice());
			// println!("{:?}", table.symbols.as_slice());
		}
		(dc_table, ac_table)
	}

	fn replace_huffman_tables(&mut self) {
		let (dc_table, ac_table) = JpegData::calc_huffman_tables(
				&mut self.components.iter().flat_map(
					|c| SplitIter::new(&c.symbols, &c.block_offsets[1..])));
		self.huff_tables.clear();
		self.huff_tables.push(dc_table);
		self.huff_tables.push(ac_table);
		for component in &mut self.components {
			component.dc_huff_table_id = 0;
			component.ac_huff_table_id = 0;
		}
	}

	// Only used for debugging
	#[allow(dead_code)]
	fn print_symbol_stats(&self) {
		for component in &self.components {
			let mut dc_symbol_counts = [0u32; 256];
			let mut ac_symbol_counts = [0u32; 256];

			let slice_iter = SplitIter::new(&component.symbols,
					&component.block_offsets[1..]);

			for slice in slice_iter {
				dc_symbol_counts[slice.first().unwrap().symbol as usize] += 1;
				for symbol in slice.tail() {
					ac_symbol_counts[symbol.symbol as usize] +=1;
				}
			}

			println!("Component: {}", component.id);
			for &(coef, syms) in [("DC", &dc_symbol_counts),
					("AC", &ac_symbol_counts)].iter() {
				println!("{}", coef);
				for (i, &sym) in syms.iter().enumerate() {
					if sym != 0 {
						println!("{:02X}: {}", i, sym);
					}
				}
			}
		}
	}
}

fn fill_buffer<R: io::Read + ?Sized>(input: &mut R, buffer: &mut [u8])
		-> Result<(), Error> {
	let mut bytes_read = 0;
	while bytes_read < buffer.len() {
		match input.read(&mut buffer[bytes_read..]) {
			Ok(0) => return ParseError::UnexpectedEof.err(),
			Ok(n) => bytes_read += n,
			Err(ref err) if err.kind() == io::ErrorKind::Interrupted => {},
			Err(err) => return Err(Error::Io(err)),
		}
	}
	Ok(())
}

pub fn process_jpeg<R: io::Read + ?Sized, W: io::Write + ?Sized>(
		input: &mut R, output: &mut W) -> Result<(), Error> {
	let mut jpeg_data = try!(JpegData::read_jpeg(input));
	jpeg_data.replace_huffman_tables();
	try!(jpeg_data.write_jpeg(output));
	Ok(())
}
