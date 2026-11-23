//! Zero-dependency JSON value AST, parser, and serializer + ZIP/OOXML (`.xlsx`) reader.
//!
//! Designed so `plo_desktop_commands` compiles 100% offline with zero crates.io network fetches
//! while providing exact RFC 8259 JSON round-tripping and native `.xlsx` OOXML archive reading
//! (`stored` method `0` and `deflate` method `8` via pure-Rust RFC 1951 inflate).

use std::collections::BTreeMap;
use std::fmt;

// ============================================================================
// Part 1: Minimal RFC 8259 JSON Value, Serializer, and Parser
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn obj(pairs: impl IntoIterator<Item = (&'static str, Json)>) -> Self {
        let mut m = BTreeMap::new();
        for (k, v) in pairs {
            m.insert(k.to_string(), v);
        }
        Json::Obj(m)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|x| x as i64)
    }

    pub fn as_u32(&self) -> Option<u32> {
        self.as_f64().and_then(|x| {
            if x >= 0.0 && x <= u32::MAX as f64 {
                Some(x as u32)
            } else {
                None
            }
        })
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().and_then(|x| {
            if x >= 0.0 {
                Some(x as usize)
            } else {
                None
            }
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&BTreeMap<String, Json>> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    pub fn to_json_pretty(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                if n.is_finite() {
                    if n.fract() == 0.0 && n.abs() < 1e15 {
                        out.push_str(&(*n as i64).to_string());
                    } else {
                        out.push_str(&n.to_string());
                    }
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(s) => write_escaped_str(out, s),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_compact(out);
                }
                out.push(']');
            }
            Json::Obj(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped_str(out, k);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, indent: usize) {
        let pad = "  ".repeat(indent);
        let pad_inner = "  ".repeat(indent + 1);
        match self {
            Json::Arr(items) if !items.is_empty() => {
                out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    out.push_str(&pad_inner);
                    item.write_pretty(out, indent + 1);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push(']');
            }
            Json::Obj(map) if !map.is_empty() => {
                out.push_str("{\n");
                let len = map.len();
                for (i, (k, v)) in map.iter().enumerate() {
                    out.push_str(&pad_inner);
                    write_escaped_str(out, k);
                    out.push_str(": ");
                    v.write_pretty(out, indent + 1);
                    if i + 1 < len {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push('}');
            }
            _ => self.write_compact(out),
        }
    }

    pub fn parse(input: &str) -> Result<Self, String> {
        let mut p = JsonParser {
            bytes: input.as_bytes(),
            pos: 0,
        };
        let val = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(format!("trailing characters at byte {}", p.pos));
        }
        Ok(val)
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_json_string())
    }
}

fn write_escaped_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        let b = self
            .peek()
            .ok_or_else(|| "unexpected EOF in JSON".to_string())?;
        match b {
            b'n' => {
                self.expect_bytes(b"null")?;
                Ok(Json::Null)
            }
            b't' => {
                self.expect_bytes(b"true")?;
                Ok(Json::Bool(true))
            }
            b'f' => {
                self.expect_bytes(b"false")?;
                Ok(Json::Bool(false))
            }
            b'"' => Ok(Json::Str(self.parse_string()?)),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            _ => Err(format!("unexpected JSON byte {:?} at {}", b as char, self.pos)),
        }
    }

    fn expect_bytes(&mut self, lit: &[u8]) -> Result<(), String> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(format!("invalid literal at {}", self.pos))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect_bytes(b"\"")?;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = *self
                    .bytes
                    .get(self.pos)
                    .ok_or_else(|| "unterminated string escape".to_string())?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{08}'),
                    b'f' => out.push('\u{0C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hex = self
                            .bytes
                            .get(self.pos..self.pos + 4)
                            .ok_or_else(|| "short \\u escape".to_string())?;
                        let s = core::str::from_utf8(hex).map_err(|_| "bad \\u hex")?;
                        let cp = u32::from_str_radix(s, 16).map_err(|_| "bad \\u hex")?;
                        self.pos += 4;
                        if let Some(ch) = char::from_u32(cp) {
                            out.push(ch);
                        }
                    }
                    _ => return Err(format!("unknown escape \\{}", esc as char)),
                }
            } else {
                let rest = core::str::from_utf8(&self.bytes[self.pos..])
                    .map_err(|_| "invalid UTF-8 in JSON string")?;
                let ch = rest.chars().next().unwrap();
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err("unterminated JSON string".to_string())
    }

    fn parse_number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        let s = core::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number UTF-8")?;
        let val: f64 = s
            .parse()
            .map_err(|_| format!("invalid JSON number {s:?}"))?;
        Ok(Json::Num(val))
    }

    fn parse_array(&mut self) -> Result<Json, String> {
        self.expect_bytes(b"[")?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at {}", self.pos)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Json, String> {
        self.expect_bytes(b"{")?;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect_bytes(b":")?;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(map));
                }
                _ => return Err(format!("expected ',' or '}}' at {}", self.pos)),
            }
        }
    }
}

// ============================================================================
// Part 2: Pure-Rust ZIP Central Directory + RFC 1951 DEFLATE Reader (for .xlsx)
// ============================================================================

/// Reads all entries of a `.zip` / `.xlsx` archive into a map of `entry_name -> decompressed_bytes`.
pub fn read_zip_entries(archive_bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
    if archive_bytes.len() < 22 {
        return Err("file is too small to be a valid .xlsx/.zip archive".to_string());
    }

    // Locate End of Central Directory (EOCD) signature `0x06054b50` scanning backward from EOF
    let min_eocd = archive_bytes.len().saturating_sub(65_557);
    let mut eocd_pos = None;
    let mut i = archive_bytes.len() - 22;
    loop {
        if read_u32_le(archive_bytes, i) == 0x0605_4b50 {
            eocd_pos = Some(i);
            break;
        }
        if i == min_eocd {
            break;
        }
        i -= 1;
    }
    let eocd = eocd_pos.ok_or_else(|| "missing ZIP End-of-Central-Directory signature".to_string())?;
    let total_entries = read_u16_le(archive_bytes, eocd + 10) as usize;
    let cd_offset = read_u32_le(archive_bytes, eocd + 16) as usize;

    let mut out = BTreeMap::new();
    let mut pos = cd_offset;
    for _ in 0..total_entries {
        if pos + 46 > archive_bytes.len() || read_u32_le(archive_bytes, pos) != 0x0201_4b50 {
            return Err("corrupt ZIP Central Directory header".to_string());
        }
        let method = read_u16_le(archive_bytes, pos + 10);
        let comp_size = read_u32_le(archive_bytes, pos + 20) as usize;
        let uncomp_size = read_u32_le(archive_bytes, pos + 24) as usize;
        let name_len = read_u16_le(archive_bytes, pos + 28) as usize;
        let extra_len = read_u16_le(archive_bytes, pos + 30) as usize;
        let comment_len = read_u16_le(archive_bytes, pos + 32) as usize;
        let local_offset = read_u32_le(archive_bytes, pos + 42) as usize;

        let name_bytes = archive_bytes
            .get(pos + 46..pos + 46 + name_len)
            .ok_or_else(|| "truncated ZIP entry name".to_string())?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        pos += 46 + name_len + extra_len + comment_len;

        if name.ends_with('/') {
            continue;
        }

        if local_offset + 30 > archive_bytes.len()
            || read_u32_le(archive_bytes, local_offset) != 0x0403_4b50
        {
            return Err(format!("corrupt local file header for {name}"));
        }
        let lf_name_len = read_u16_le(archive_bytes, local_offset + 26) as usize;
        let lf_extra_len = read_u16_le(archive_bytes, local_offset + 28) as usize;
        let data_start = local_offset + 30 + lf_name_len + lf_extra_len;
        let comp_slice = archive_bytes
            .get(data_start..data_start + comp_size)
            .ok_or_else(|| format!("truncated compressed payload for {name}"))?;

        let data = match method {
            0 => comp_slice.to_vec(),
            8 => inflate_raw(comp_slice, uncomp_size)?,
            m => return Err(format!("unsupported ZIP compression method {m} in {name}")),
        };
        out.insert(name, data);
    }

    Ok(out)
}

#[inline]
fn read_u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

// ============================================================================
// Pure-Rust RFC 1951 DEFLATE Decompressor
// ============================================================================

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u32,
    bit_cnt: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    fn ensure(&mut self, n: u8) {
        while self.bit_cnt < n && self.byte_pos < self.data.len() {
            self.bit_buf |= (self.data[self.byte_pos] as u32) << self.bit_cnt;
            self.byte_pos += 1;
            self.bit_cnt += 8;
        }
    }

    fn read_bits(&mut self, n: u8) -> Result<u32, String> {
        if n == 0 {
            return Ok(0);
        }
        self.ensure(n);
        if self.bit_cnt < n {
            return Err("unexpected EOF in DEFLATE bitstream".to_string());
        }
        let val = self.bit_buf & ((1u32 << n) - 1);
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop = self.bit_cnt & 7;
        self.bit_buf >>= drop;
        self.bit_cnt -= drop;
    }

    fn read_u16_aligned(&mut self) -> Result<u16, String> {
        self.align_byte();
        let lo = self.read_bits(8)? as u16;
        let hi = self.read_bits(8)? as u16;
        Ok(lo | (hi << 8))
    }
}

struct HuffmanTree {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl HuffmanTree {
    fn from_lengths(lengths: &[u8]) -> Result<Self, String> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l as usize >= 16 {
                return Err("Huffman code length > 15".to_string());
            }
            if l > 0 {
                counts[l as usize] += 1;
            }
        }
        let mut offsets = [0u16; 16];
        let mut total = 0u16;
        for bits in 1..16 {
            offsets[bits] = total;
            total += counts[bits];
        }
        let mut symbols = vec![0u16; total as usize];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                let idx = offsets[l as usize] as usize;
                symbols[idx] = sym as u16;
                offsets[l as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, String> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=15 {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err("invalid Huffman code in DEFLATE stream".to_string())
    }
}

fn fixed_trees() -> (HuffmanTree, HuffmanTree) {
    let mut lit = vec![0u8; 288];
    lit[0..=143].fill(8);
    lit[144..=255].fill(9);
    lit[256..=279].fill(7);
    lit[280..=287].fill(8);
    let dist = vec![5u8; 32];
    (
        HuffmanTree::from_lengths(&lit).unwrap(),
        HuffmanTree::from_lengths(&dist).unwrap(),
    )
}

fn decode_dynamic_trees(br: &mut BitReader<'_>) -> Result<(HuffmanTree, HuffmanTree), String> {
    let hlit = br.read_bits(5)? as usize + 257;
    let hdist = br.read_bits(5)? as usize + 1;
    let hclen = br.read_bits(4)? as usize + 4;

    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut cl_lens = [0u8; 19];
    for &idx in ORDER.iter().take(hclen) {
        cl_lens[idx] = br.read_bits(3)? as u8;
    }
    let cl_tree = HuffmanTree::from_lengths(&cl_lens)?;

    let mut all_lens = Vec::with_capacity(hlit + hdist);
    while all_lens.len() < hlit + hdist {
        let sym = cl_tree.decode(br)?;
        match sym {
            0..=15 => all_lens.push(sym as u8),
            16 => {
                let prev = *all_lens
                    .last()
                    .ok_or_else(|| "repeat code 16 with no previous length".to_string())?;
                let rep = br.read_bits(2)? as usize + 3;
                for _ in 0..rep {
                    all_lens.push(prev);
                }
            }
            17 => {
                let rep = br.read_bits(3)? as usize + 3;
                all_lens.resize(all_lens.len() + rep, 0);
            }
            18 => {
                let rep = br.read_bits(7)? as usize + 11;
                all_lens.resize(all_lens.len() + rep, 0);
            }
            _ => return Err("invalid code-length symbol".to_string()),
        }
    }
    if all_lens.len() != hlit + hdist {
        return Err("DEFLATE code length overflow".to_string());
    }
    let lit_tree = HuffmanTree::from_lengths(&all_lens[..hlit])?;
    let dist_tree = HuffmanTree::from_lengths(&all_lens[hlit..])?;
    Ok((lit_tree, dist_tree))
}

fn inflate_raw(input: &[u8], expected_size: usize) -> Result<Vec<u8>, String> {
    const LEN_BASE: [usize; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [usize; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    let mut br = BitReader::new(input);
    let mut out = Vec::with_capacity(expected_size);

    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        match btype {
            0 => {
                let len = br.read_u16_aligned()? as usize;
                let nlen = br.read_u16_aligned()?;
                if (len as u16) != !nlen {
                    return Err("stored DEFLATE block length mismatch".to_string());
                }
                for _ in 0..len {
                    out.push(br.read_bits(8)? as u8);
                }
            }
            1 | 2 => {
                let (lit_tree, dist_tree) = if btype == 1 {
                    fixed_trees()
                } else {
                    decode_dynamic_trees(&mut br)?
                };
                loop {
                    let sym = lit_tree.decode(&mut br)? as usize;
                    if sym < 256 {
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else if sym <= 285 {
                        let l_idx = sym - 257;
                        let length =
                            LEN_BASE[l_idx] + br.read_bits(LEN_EXTRA[l_idx])? as usize;
                        let d_sym = dist_tree.decode(&mut br)? as usize;
                        if d_sym >= 30 {
                            return Err("invalid distance symbol".to_string());
                        }
                        let dist =
                            DIST_BASE[d_sym] + br.read_bits(DIST_EXTRA[d_sym])? as usize;
                        if dist == 0 || dist > out.len() {
                            return Err("DEFLATE back-reference out of bounds".to_string());
                        }
                        for _ in 0..length {
                            let b = out[out.len() - dist];
                            out.push(b);
                        }
                    } else {
                        return Err("invalid literal/length symbol".to_string());
                    }
                }
            }
            _ => return Err("reserved DEFLATE block type 3".to_string()),
        }
        if bfinal == 1 {
            break;
        }
    }

    Ok(out)
}

// ============================================================================
// Part 3: Minimal OOXML Worksheet / SharedStrings Extractor
// ============================================================================

#[derive(Clone, Debug)]
pub struct XlsxSheet {
    pub name: String,
    /// Map `row_number_1_based -> Map<column_letters, cell_text_value>`
    pub rows: BTreeMap<u32, BTreeMap<String, String>>,
}

/// Parses an `.xlsx` file buffer into ordered [`XlsxSheet`] tables (evaluating cached cell values
/// and resolving `sharedStrings.xml` indices + inline strings).
pub fn read_xlsx_workbook(bytes: &[u8]) -> Result<Vec<XlsxSheet>, String> {
    let entries = read_zip_entries(bytes)?;

    let sst = if let Some(sst_bytes) = entries.get("xl/sharedStrings.xml") {
        let xml = String::from_utf8_lossy(sst_bytes);
        parse_shared_strings(&xml)
    } else {
        Vec::new()
    };

    let wb_xml = String::from_utf8_lossy(
        entries
            .get("xl/workbook.xml")
            .ok_or_else(|| "missing xl/workbook.xml in .xlsx".to_string())?,
    );
    let rels_xml = String::from_utf8_lossy(
        entries
            .get("xl/_rels/workbook.xml.rels")
            .ok_or_else(|| "missing xl/_rels/workbook.xml.rels in .xlsx".to_string())?,
    );

    let rel_map = parse_workbook_rels(&rels_xml);
    let sheet_defs = parse_workbook_sheets(&wb_xml);

    let mut sheets = Vec::new();
    for (sheet_name, r_id) in sheet_defs {
        let Some(target) = rel_map.get(&r_id) else {
            continue;
        };
        let norm_target = if let Some(stripped) = target.strip_prefix('/') {
            stripped.to_string()
        } else if target.starts_with("xl/") {
            target.clone()
        } else {
            format!("xl/{target}")
        };
        if let Some(sheet_bytes) = entries.get(&norm_target) {
            let xml = String::from_utf8_lossy(sheet_bytes);
            let rows = parse_worksheet_rows(&xml, &sst);
            sheets.push(XlsxSheet {
                name: sheet_name,
                rows,
            });
        }
    }

    Ok(sheets)
}

fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

fn extract_attr<'a>(tag_header: &'a str, attr_name: &str) -> Option<&'a str> {
    let pat = format!("{attr_name}=\"");
    let idx = tag_header.find(&pat)?;
    let rest = &tag_header[idx + pat.len()..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn parse_shared_strings(xml: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel_si) = xml[cursor..].find("<si") {
        let si_start = cursor + rel_si;
        let Some(after_open) = xml[si_start..].find('>') else {
            break;
        };
        let body_start = si_start + after_open + 1;
        let Some(rel_end) = xml[body_start..].find("</si>") else {
            break;
        };
        let si_body = &xml[body_start..body_start + rel_end];
        let mut combined = String::new();
        let mut tc = 0usize;
        while let Some(rt) = si_body[tc..].find("<t") {
            let ts = tc + rt;
            let after_t_name = si_body.as_bytes().get(ts + 2).copied().unwrap_or(b' ');
            if after_t_name != b'>' && after_t_name != b' ' {
                tc = ts + 2;
                continue;
            }
            let Some(gt) = si_body[ts..].find('>') else {
                break;
            };
            if si_body[ts..ts + gt].ends_with('/') {
                tc = ts + gt + 1;
                continue;
            }
            let txt_start = ts + gt + 1;
            let Some(end_t) = si_body[txt_start..].find("</t>") else {
                break;
            };
            combined.push_str(&xml_unescape(&si_body[txt_start..txt_start + end_t]));
            tc = txt_start + end_t + 4;
        }
        out.push(combined);
        cursor = body_start + rel_end + 5;
    }
    out
}

fn parse_workbook_rels(xml: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut cursor = 0usize;
    while let Some(rel_pos) = xml[cursor..].find("<Relationship ") {
        let start = cursor + rel_pos;
        let Some(end) = xml[start..].find('>') else {
            break;
        };
        let tag = &xml[start..start + end];
        if let (Some(id), Some(target)) = (extract_attr(tag, "Id"), extract_attr(tag, "Target")) {
            map.insert(id.to_string(), target.to_string());
        }
        cursor = start + end + 1;
    }
    map
}

fn parse_workbook_sheets(xml: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(pos) = xml[cursor..].find("<sheet ") {
        let start = cursor + pos;
        let Some(end) = xml[start..].find('>') else {
            break;
        };
        let tag = &xml[start..start + end];
        let name = extract_attr(tag, "name").map(xml_unescape);
        let rid = extract_attr(tag, "r:id").or_else(|| extract_attr(tag, "id"));
        if let (Some(n), Some(r)) = (name, rid) {
            out.push((n, r.to_string()));
        }
        cursor = start + end + 1;
    }
    out
}

fn parse_worksheet_rows(
    xml: &str,
    sst: &[String],
) -> BTreeMap<u32, BTreeMap<String, String>> {
    let mut rows = BTreeMap::new();
    let mut cursor = 0usize;
    while let Some(rpos) = xml[cursor..].find("<row ") {
        let r_start = cursor + rpos;
        let Some(r_gt) = xml[r_start..].find('>') else {
            break;
        };
        let r_tag = &xml[r_start..r_start + r_gt];
        if r_tag.ends_with('/') {
            cursor = r_start + r_gt + 1;
            continue;
        }
        let row_num: u32 = extract_attr(r_tag, "r")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let body_start = r_start + r_gt + 1;
        let Some(r_end) = xml[body_start..].find("</row>") else {
            break;
        };
        let row_body = &xml[body_start..body_start + r_end];

        let mut cells = BTreeMap::new();
        let mut cc = 0usize;
        while let Some(cpos) = row_body[cc..].find("<c ") {
            let cs = cc + cpos;
            let Some(cgt) = row_body[cs..].find('>') else {
                break;
            };
            let c_tag = &row_body[cs..cs + cgt];
            if c_tag.ends_with('/') {
                cc = cs + cgt + 1;
                continue;
            }
            let cell_ref = extract_attr(c_tag, "r").unwrap_or("");
            let col_letters: String = cell_ref
                .chars()
                .take_while(|ch| ch.is_ascii_alphabetic())
                .collect();
            let cell_type = extract_attr(c_tag, "t").unwrap_or("");

            let c_body_start = cs + cgt + 1;
            let Some(c_end) = row_body[c_body_start..].find("</c>") else {
                break;
            };
            let c_body = &row_body[c_body_start..c_body_start + c_end];

            let val_opt = if cell_type == "inlineStr" {
                extract_tag_text(c_body, "t").map(xml_unescape)
            } else if let Some(raw_v) = extract_tag_text(c_body, "v") {
                if cell_type == "s" {
                    raw_v
                        .trim()
                        .parse::<usize>()
                        .ok()
                        .and_then(|idx| sst.get(idx).cloned())
                } else {
                    Some(xml_unescape(raw_v.trim()))
                }
            } else {
                None
            };

            if let Some(val) = val_opt {
                let trimmed = val.trim().to_string();
                if !trimmed.is_empty() && !col_letters.is_empty() {
                    cells.insert(col_letters, trimmed);
                }
            }

            cc = c_body_start + c_end + 4;
        }

        if row_num > 0 && !cells.is_empty() {
            rows.insert(row_num, cells);
        }
        cursor = body_start + r_end + 6;
    }
    rows
}

fn extract_tag_text<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}");
    let pos = xml.find(&open)?;
    let after = xml.as_bytes().get(pos + open.len()).copied()?;
    if after != b'>' && after != b' ' {
        return None;
    }
    let gt = xml[pos..].find('>')?;
    if xml[pos..pos + gt].ends_with('/') {
        return None;
    }
    let start = pos + gt + 1;
    let close = format!("</{tag}>");
    let end = xml[start..].find(&close)?;
    Some(&xml[start..start + end])
}
