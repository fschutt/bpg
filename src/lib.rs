extern crate byteorder;

use std::io;
use byteorder::{BigEndian, ReadBytesExt};

pub const FILE_MAGIC: u32 = 0x425047fb;

/*
heic_file() {

     file_magic                                                  u(32)

     pixel_format                                                u(3)
     alpha1_flag                                                 u(1)
     bit_depth_minus_8                                           u(4)

     color_space                                                 u(4)
     extension_present_flag                                      u(1)
     alpha2_flag                                                 u(1)
     limited_range_flag                                          u(1)
     animation_flag                                              u(1)
     
     picture_width                                               ue7(32)
     picture_height                                              ue7(32)
     
     picture_data_length                                         ue7(32)
     if (extension_present_flag)  
         extension_data_length                                   ue7(32)
         extension_data()
     }

     hevc_header_and_data()
}

*/

fn ue7_decode() -> usize {
/*
- ue7(n) is an unsigned integer of at most n bits stored on a variable
  number of bytes. All the bytes except the last one have a '1' as
  their first bit. The unsigned integer is represented as the
  concatenation of the remaining 7 bit codewords. Only the shortest
  encoding for a given unsigned integer shall be accepted by the
  decoder (i.e. the first byte is never 0x80). Example:
*/
/*
0x08                8
0x84 0x1e           542
0xac 0xbe 0x17      728855
*/
    728855
}

#[derive(Debug, Clone)]
pub struct BpgFile {
    pub pixel_format: PixelFormat,
    pub alpha: (AlphaPlanePresent, HasPremultipliedAlpha),
    pub bit_depth_minus_8: BitDepthMinus8,
    pub color_space: ColorSpace,
    pub extension_present: ExtensionPresent,
    pub limited_range: LimitedRange,
    pub animation: AnimationFlag,
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
        let data: u8 = (data & 0b00010000) >> 4;
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
pub enum LimitedRange {
    NotLimited,
    Limited,
}

impl LimitedRange {
    fn from_u8(data: u8) -> Self {
        use LimitedRange::*;
        // [_, _, _, _, FLAG, _, _, _]
        let data: u8 = (data & 0b00000010) >> 1;
        // [0, 0, 0, 0, 0, 0, 0, FLAG]
        match data {
            0 => NotLimited,
            1 => Limited,
            _ => unreachable!()
        }
    }
}

impl Into<u8> for LimitedRange {
    fn into(self) -> u8 {
        use LimitedRange::*;
        match self {
            NotLimited => 0,
            Limited => 1,
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
    let limited_range_flag = LimitedRange::from_u8(third_byte);
    let animation_flag = AnimationFlag::from_u8(third_byte);

    Ok(BpgFile {
        pixel_format: pixel_format,
        alpha: (alpha_plane_present, has_premutliplied_alpha),
        bit_depth_minus_8: bit_depth_minus_8,
        color_space: color_space,
        extension_present: extension_present_flag,
        limited_range: limited_range_flag,
        animation: animation_flag,
    })
}