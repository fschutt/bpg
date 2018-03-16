extern crate byteorder;
extern crate smallvec;

use std::io;
use byteorder::{BigEndian, ReadBytesExt};

pub mod exp_golomb;

pub const FILE_MAGIC: u32 = 0x425047fb;

/// Decodes a ue7(n) value.
/// 
/// ue7(n) is an unsigned integer of at most n bits stored on a variable
/// number of bytes. All the bytes except the last one have a '1' as
/// their first bit. The unsigned integer is represented as the
/// concatenation of the remaining 7 bit codewords. Only the shortest
/// encoding for a given unsigned integer shall be accepted by the
/// decoder (i.e. the first byte is never 0x80). Example:
/// 
/// Encoded bytes       Unsigned integer value
/// 0x08                8
/// 0x84 0x1e           542
/// 0xac 0xbe 0x17      728855
pub fn ue7_decode<R: ReadBytesExt>(bytes: &mut R, max_bytes: usize) -> Result<usize, BpgDecodeError> {
    
    use smallvec::SmallVec;

    // the usual size is 4 bytes, which is why it makes sense to use smallvec here
    let mut bit_vec = SmallVec::<[bool; 32]>::new(); // should use 28 items for 4 bytes

    let mut is_last_byte = false;
    let mut read_bytes = 0;

    while !is_last_byte && read_bytes <= max_bytes {
        let b = bytes.read_u8()?;
        bit_vec.push((b & 0b01000000) >> 6 != 0);
        bit_vec.push((b & 0b00100000) >> 5 != 0);
        bit_vec.push((b & 0b00010000) >> 4 != 0);
        bit_vec.push((b & 0b00001000) >> 3 != 0);
        bit_vec.push((b & 0b00000100) >> 2 != 0);
        bit_vec.push((b & 0b00000010) >> 1 != 0);
        bit_vec.push((b & 0b00000001)      != 0);

        is_last_byte = (b & 0b10000000) >> 7 != 1;
        read_bytes += 1;
    }

    // bit_vec now contains a concatenation of 7-bit Bits
    // These bits, when concatenated, form the 
    let bit_len = bit_vec.len();
    let remaining_len = 8 - (bit_len % 8); // probably not correct
    let next_power_of_two = bit_len + remaining_len;

    // how many bytes the usize has
    let is_64_bit = (usize::max_value() as u64) == u64::max_value();
    let max_len_bits = if is_64_bit { 64 } else { 32 };

    if next_power_of_two > max_len_bits {
        return Err(BpgDecodeError::OverflowingUnsignedInteger(next_power_of_two));
    }

    // since the usize cannot hold more than 64 bits without overflowing,
    // we should use the maximum of 64 bits here
    let mut final_vec = SmallVec::<[u8; 64]>::new();
    for _ in 0..next_power_of_two {
        final_vec.push(0);
    }

    assert!(bit_vec.len() <= final_vec.len());

    for (i, b) in bit_vec.into_iter().enumerate() {
        final_vec[remaining_len + i] = b as u8;
    }

    assert!(final_vec.len() % 8 == 0);

    // now we have the bits in the correct order, 0000010101010 etc.

    // pad out to 32 / 64 bits
    assert!(max_len_bits >= final_vec.len());
    let missing_to_usize_bit_len = max_len_bits - final_vec.len(); 

    if missing_to_usize_bit_len != 0 {
        let mut temp_vec = SmallVec::<[u8; 64]>::new();
        // missing_to_usize_bit_len
        for _ in 0..missing_to_usize_bit_len { 
            temp_vec.push(0);
        }
        for e in final_vec.iter() {
            temp_vec.push(*e);
        }
        final_vec = temp_vec;
    }

    assert!(final_vec.len() == max_len_bits);

    let mut final_num: usize = 0;

    for i in 0..max_len_bits {
        final_num |= (final_vec[i] as usize) << (max_len_bits - i - 1);
    }

    Ok(final_num)
}

#[test]
fn ue7_decode_test_1() {
    use std::io::Cursor;
    assert_eq!(ue7_decode(&mut Cursor::new(vec![0x08]), 4).unwrap(), 8);
}

#[test]
fn ue7_decode_test_2() {
    use std::io::Cursor;
    assert_eq!(ue7_decode(&mut Cursor::new(vec![0x84, 0x1e]), 4).unwrap(), 542);
}

#[test]
fn ue7_decode_test_3() {
    use std::io::Cursor;
    assert_eq!(ue7_decode(&mut Cursor::new(vec![0xac, 0xbe, 0x17]), 4).unwrap(), 728855);
}

#[derive(Debug, Clone)]
pub struct BpgFile {
    pub header: BpgFileHeader,
    pub frame: HevcFrame,
}

#[derive(Debug, Clone)]
pub struct BpgFileHeader {
    pub pixel_format: PixelFormat,
    pub alpha: (AlphaPlanePresent, HasPremultipliedAlpha),
    pub bit_depth_minus_8: BitDepthMinus8,
    pub color_space: ColorSpace,
    pub extension_present: ExtensionPresent,
    pub limited_range: HasLimitedRange,
    pub animation: AnimationFlag,
    pub width: usize,
    pub height: usize,
    pub extensions: Vec<Extension>,
    pub picture_data_length: usize,
}

#[derive(Debug)]
pub enum BpgDecodeError {
    InvalidFileMagic(u32),
    InvalidPixelFormat(u8),
    InvalidAlpha1Flag(u8),
    InvalidBitDepthMinus8(u8),
    InvalidPremultipliedFlag(u8),
    /// The pixel_format was Grayscale, but the ColorSpace was not YCbCr
    ColorSpaceMismatch(u8),
    InvalidColorSpace(u8),
    OverflowingUnsignedInteger(usize),
    /// Failed to read more bytes, corrupt file
    Io(io::Error),
}

impl From<io::Error> for BpgDecodeError {
    fn from(e: io::Error) -> Self {
        BpgDecodeError::Io(e)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PixelFormat {
    Grayscale,
    Chroma420JPEG,
    Chroma422JPEG,
    Chroma444,
    Chroma420MPEG2,
    Chroma422MPEG2,
}

impl PixelFormat {
    /// Reads the first 3 bits of a byte
    fn try_from_u8(data: u8) -> Result<Self, BpgDecodeError> {
        use PixelFormat::*;
        // [B1, B2, B3, _, _, _, _, _]
        let data = (data & 0b11100000) >> 5;
        // [0, 0, 0, 0, 0, B1, B2, B3]
        match data {
            0 => Ok(Grayscale),
            1 => Ok(Chroma420JPEG),
            2 => Ok(Chroma422JPEG),
            3 => Ok(Chroma444),
            4 => Ok(Chroma420MPEG2),
            5 => Ok(Chroma422MPEG2),
            _ => Err(BpgDecodeError::InvalidPixelFormat(data)),
        }
    }
}

impl Into<u8> for PixelFormat {
    fn into(self) -> u8 {
        use PixelFormat::*;
        match self {
            Grayscale => 0,
            Chroma420JPEG => 1,
            Chroma422JPEG => 2,
            Chroma444 => 3,
            Chroma420MPEG2 => 4,
            Chroma422MPEG2 => 5,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AlphaPlanePresent {
    NoAlphaPresent, /* 0 */
    AlphaPresent,   /* 1 */
}

impl AlphaPlanePresent {
    /// Reads the 3rd bit from the right (tested with tux image)
    fn from_u8(data: u8) -> Self {
        use AlphaPlanePresent::*;
        // [_, _, _, FLAG, _, _, _, _]
        let data: u8 = (data & 0b0001000) >> 4;
        // [0, 0, 0, 0, 0, 0, 0, FLAG]
        match data {
            0 => NoAlphaPresent,
            1 => AlphaPresent,
            _ => unreachable!(),
        }
    }
}

impl Into<u8> for AlphaPlanePresent {
    fn into(self) -> u8 {
        use AlphaPlanePresent::*;
        match self {
            NoAlphaPresent => 0,
            AlphaPresent => 1,
        }
    }
}

// Bit depth minus 8, guaranteed to be <= 6
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitDepthMinus8(pub u8);

impl BitDepthMinus8 {
    fn try_from_u8(data: u8) -> Result<Self, BpgDecodeError> {
        // [_, _, _, _, B1, B2, B3, B4]
        let data: u8 = data & 0b00001111;
        // [0, 0, 0, 0, B1, B2, B3, B4]
        if data > 6 {
            Err(BpgDecodeError::InvalidBitDepthMinus8(data))
        } else {
            Ok(BitDepthMinus8(data))
        }
    }
}

/// "alpha2_flag"
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum HasPremultipliedAlpha {
    NotPremultiplied,
    Premultiplied,
}

impl HasPremultipliedAlpha {
    fn from_u8(data: u8) -> Self {
        use HasPremultipliedAlpha::*;
        // [_, _, _, _, _, FLAG, _, _]
        let data: u8 = (data & 0b00000100) >> 2;
        // [0, 0, 0, 0, 0, 0, 0, FLAG]
        match data {
            0 => NotPremultiplied,
            1 => Premultiplied,
            _ => unreachable!()
        }
    }
}

impl Into<u8> for HasPremultipliedAlpha {
    fn into(self) -> u8 {
        use HasPremultipliedAlpha::*;
        match self {
            NotPremultiplied => 0,
            Premultiplied => 1,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ColorSpace {
    /// YCbCr (BT 601, same as JPEG and HEVC matrix_coeffs = 5)
    YCbCrB601,
    /// RGB (component order: G B R)
    Gbr,
    /// YCgCo (same as HEVC matrix_coeffs = 8)
    YCgCo,
    /// YCbCr (BT 709, same as HEVC matrix_coeffs = 1)
    YCbCrBT709,
    /// YCbCr (BT 2020 non constant luminance system, same as HEVC 
    /// matrix_coeffs = 9)
    YCbCrBT2020,
    /* other values reserved */
}

impl ColorSpace {
    /// Reads the first 3 bits of a byte
    fn try_from_u8(data: u8, pixel_format: &PixelFormat) -> Result<Self, BpgDecodeError> {
        use ColorSpace::*;
        // [B1, B2, B3, B4, _, _, _, _]
        let data = (data & 0b11110000) >> 4;
        // [0, 0, 0, 0, 0, B1, B2, B3]

        if *pixel_format == PixelFormat::Grayscale && data != 0 {
            return Err(BpgDecodeError::ColorSpaceMismatch(data));
        }

        match data {
            0 => Ok(YCbCrB601),
            1 => Ok(Gbr),
            2 => Ok(YCgCo),
            3 => Ok(YCbCrBT709),
            4 => Ok(YCbCrBT2020),
            // 5 is in the BPG specification, but unsupported even there.
            // So it doesn't make any sense to include it here.
            _ => Err(BpgDecodeError::InvalidColorSpace(data)),
        }
    }
}

impl Into<u8> for ColorSpace {
    fn into(self) -> u8 {
        use ColorSpace::*;
        match self {
            YCbCrB601 => 0,
            Gbr => 1,
            YCgCo => 2,
            YCbCrBT709 => 3,
            YCbCrBT2020 => 4,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ExtensionPresent {
    NotPresent,
    Present,
}

impl ExtensionPresent {
    fn from_u8(data: u8) -> Self {
        use ExtensionPresent::*;
        // [_, _, _, _, FLAG, _, _, _]
        let data: u8 = (data & 0b00001000) >> 3;
        // [0, 0, 0, 0, 0, 0, 0, FLAG]
        match data {
            0 => NotPresent,
            1 => Present,
            _ => unreachable!()
        }
    }
}

impl Into<u8> for ExtensionPresent {
    fn into(self) -> u8 {
        use ExtensionPresent::*;
        match self {
            NotPresent => 0,
            Present => 1,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum HasLimitedRange {
    FullRange,
    LimitedRange,
}

impl HasLimitedRange {
    fn from_u8(data: u8) -> Self {
        use HasLimitedRange::*;
        // [_, _, _, _, FLAG, _, _, _]
        let data: u8 = (data & 0b00000010) >> 1;
        // [0, 0, 0, 0, 0, 0, 0, FLAG]
        match data {
            0 => FullRange,
            1 => LimitedRange,
            _ => unreachable!()
        }
    }
}

impl Into<u8> for HasLimitedRange {
    fn into(self) -> u8 {
        use HasLimitedRange::*;
        match self {
            FullRange => 0,
            LimitedRange => 1,
        }
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AnimationFlag {
    NoAnimation,
    HasAnimation,
}

impl AnimationFlag {
    fn from_u8(data: u8) -> Self {
        use AnimationFlag::*;
        // [_, _, _, _, _, _, _, FLAG]
        let data: u8 = data & 0b00000001;
        // [0, 0, 0, 0, 0, 0, 0, FLAG]
        match data {
            0 => NoAnimation,
            1 => HasAnimation,
            _ => unreachable!()
        }
    }
}

impl Into<u8> for AnimationFlag {
    fn into(self) -> u8 {
        use AnimationFlag::*;
        match self {
            NoAnimation => 0,
            HasAnimation => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Extension {
    pub ext_type: ExtensionType,
    pub ext_data: ExtensionData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtensionType {
    /// EXIF data
    Exif = 1,
    /// ICC profile (see [4])
    IccProfile = 2,
    /// XMP (see [5])
    Xmp = 3,
    /// Thumbnail (the thumbnail shall be a lower resolution version
    /// of the image and stored in BPG format).
    Thumbnail = 4,
    /// Animation control data
    AnimationControl = 5,
}

#[derive(Debug, Clone)]
pub struct ExtensionData(pub Vec<u8>);

/// Decodes a BPG file
pub fn decode<R: ReadBytesExt>(bytes: &mut R) -> Result<BpgFile, BpgDecodeError> {

    {
        let first_byte = bytes.read_u32::<BigEndian>()?;
        if FILE_MAGIC != first_byte {
            return Err(BpgDecodeError::InvalidFileMagic(first_byte));
        }
    }

    let second_byte = bytes.read_u8()?;

    /*
        pixel_format                                                u(3)
        alpha1_flag                                                 u(1)
        bit_depth_minus_8                                           u(4)
    */

    let pixel_format = PixelFormat::try_from_u8(second_byte)?;
    let alpha_plane_present = AlphaPlanePresent::from_u8(second_byte);
    let bit_depth_minus_8 = BitDepthMinus8::try_from_u8(second_byte).unwrap();

    /*
        color_space                                                 u(4)
        extension_present_flag                                      u(1)
        alpha2_flag                                                 u(1)
        limited_range_flag                                          u(1)
        animation_flag                                              u(1)
    */

    let third_byte = bytes.read_u8()?;

    let color_space = ColorSpace::try_from_u8(third_byte, &pixel_format)?;
    let extension_present_flag = ExtensionPresent::from_u8(third_byte);
    let has_premutliplied_alpha = HasPremultipliedAlpha::from_u8(third_byte);
    let limited_range_flag = HasLimitedRange::from_u8(third_byte);
    let animation_flag = AnimationFlag::from_u8(third_byte);
    
    /*
        picture_width                                               ue7(32)
        picture_height                                              ue7(32)
        picture_data_length                                         ue7(32)
    */

    // TODO: Width of 0 not allowed
    let picture_width = ue7_decode(bytes, 4)?;
    // TODO: Height of 0 not allowed
    let picture_height = ue7_decode(bytes, 4)?;

    // TODO: The special value of zero indicates that the picture data 
    // goes up to the end of the file.
    let picture_data_length = ue7_decode(bytes, 4)?;

    /*
        if (extension_present_flag)  
            extension_data_length                                   ue7(32)
            extension_data()
        }
    */

    let mut extensions = Vec::new();
    if extension_present_flag == ExtensionPresent::Present {
        let extension_data_length = ue7_decode(bytes, 4)?;
        extensions = decode_extension_data(bytes, extension_data_length)?;
    }

    let bpg_file_header = BpgFileHeader {
        pixel_format: pixel_format,
        alpha: (alpha_plane_present, has_premutliplied_alpha),
        bit_depth_minus_8: bit_depth_minus_8,
        color_space: color_space,
        extension_present: extension_present_flag,
        limited_range: limited_range_flag,
        animation: animation_flag,
        width: picture_width,
        height: picture_height,
        extensions: extensions,
        picture_data_length: picture_data_length,
    };

    let hevc_frame = decode_hevc_header_and_data(bytes, &bpg_file_header)?;

    Ok(BpgFile {
        header: bpg_file_header,
        frame: hevc_frame,
    })
}

fn decode_extension_data<R: ReadBytesExt>(bytes: &mut R, max_length: usize) -> Result<Vec<Extension>, BpgDecodeError> {

    // at least advance the pointer, even if the result is thrown away
    for _ in 0..max_length {
        // TODO: process the extensions correctly !! 
        let _b = bytes.read_u8()?;
    }

    Ok(Vec::new())
}

#[derive(Debug, Clone)]
pub struct HevcFrame {
    header: HevcHeader,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct HevcHeader {

}

fn decode_hevc_header_and_data<R: ReadBytesExt>(bytes: &mut R, file_header: &BpgFileHeader) -> Result<HevcFrame, BpgDecodeError>
{
    let hevc_header = decode_hevc_header(bytes, file_header)?;
     /*
         if (alpha1_flag || alpha2_flag) {
             hevc_header()
         }
         hevc_header()
         hevc_data()
     */
     let mut data_vec = Vec::new();
     let data = if file_header.picture_data_length == 0 {
        bytes.read_to_end(&mut data_vec)?;
     } else {
        data_vec.resize(file_header.picture_data_length, 0);
        bytes.read_exact(&mut data_vec)?;
     };

     Ok(HevcFrame {
        header: hevc_header,
        data: data_vec,
     })
}

/// - ue(v): unsigned integer 0-th order Exp-Golomb-coded syntax element with the left bit first. The parsing
/// process for this descriptor is specified in clause 9.2.
pub fn ue_decode<R: ReadBytesExt>(bytes: &mut R) {

}

fn decode_hevc_header<R: ReadBytesExt>(bytes: &mut R, file_header: &BpgFileHeader) -> Result<HevcHeader, BpgDecodeError> {
    /*
        hevc_header_length                                          ue7(32)
        log2_min_luma_coding_block_size_minus3                      ue(v)
        log2_diff_max_min_luma_coding_block_size                    ue(v)
        log2_min_transform_block_size_minus2                        ue(v)
        log2_diff_max_min_transform_block_size                      ue(v)
        max_transform_hierarchy_depth_intra                         ue(v)
        sample_adaptive_offset_enabled_flag                         u(1)
        pcm_enabled_flag                                            u(1)
    */

    let hevc_header_length = ue7_decode(bytes, 4);

    Ok(HevcHeader {

    })
}

pub fn recover_non_premultiplied_rgb(r: u8, g: u8, b: u8, a: u8) -> (f32, f32, f32) {
    /*
        alpha1_flag=1 alpha2_flag=1: alpha present. The color is premultiplied. 
        The resulting non-premultiplied R', G', B' shall
               be recovered as:
                  
                 if A != 0 
                   R' = min(R / A, 1), G' = min(G / A, 1), B' = min(B / A, 1)
                 else
                   R' = G' = B' = 1 .
    */

    if a != 0 {
        let r = (r as f32 / a as f32).min(1.0);
        let g = (g as f32 / a as f32).min(1.0);
        let b = (b as f32 / a as f32).min(1.0);
        (r, g, b)
    } else {
        (1.0, 1.0, 1.0)
    }
}

pub fn recover_cmyk(r: f32, g: f32, b: f32, w: f32) -> (f32, f32, f32, f32) {
    
    /*
        alpha1_flag=0 alpha2_flag=1: the alpha plane is present and
        contains the W color component (CMYK color). The resulting CMYK
        data can be recovered as follows:

          C = (1 - R), M = (1 - G), Y = (1 - B), K = (1 - W) .
    */

    let c = 1.0 - r;
    let m = 1.0 - g;
    let y = 1.0 - b;
    let k = 1.0 - w;
    (c, m, y, k)
}

pub fn recover_limited_range_y() {
    /*
             - (16 << (bit_depth - 8) to (235 << (bit_depth - 8)) for Y
        and G, B, R,
             - (16 << (bit_depth - 8) to (240 << (bit_depth - 8)) for Cb and Cr.
    */
}