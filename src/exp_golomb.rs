//! Exp-golomb coding

use byteorder::ReadBytesExt;
use std::io::Cursor;

pub struct ExpGolombDecoder<R: ReadBytesExt> {
    pub data: R,
    /// The bits we've already taken but not completely used
    /// (for incomplete values)
    pub last_bytes: Vec<u8>,
    /// Bit lag into the last byte
    pub bit_lag_current_byte: u8,
    /// Total bit lag
    pub bit_lag_total: isize,
}

impl<R: ReadBytesExt> ExpGolombDecoder<R> {
    pub fn new(data: R) -> Self {
        Self {
            data: data,
            last_bytes: Vec::new(),
            bit_lag_current_byte: 0,
            bit_lag_total: 0,
        }
    }
}

pub struct ExpGolombEncoder {

}

#[derive(Debug, Clone)]
pub enum ExpGolombEncodeError {

}

pub fn exp_golomb_decode<R: ReadBytesExt>(decoder: &mut ExpGolombDecoder<R>, max_bytes: Option<usize>) 
    -> Result<Vec<usize>, ExpGolombEncodeError> 
{    
    // TODO: 32-bit / 64-bit overflow correctness!

    let mut result = Vec::<usize>::new();
    let mut current_bit: isize = 0; 

    #[derive(Debug, PartialEq)]
    enum Status {
        Ok,
        NeedNextByte,
    }

    let mut status = Status::NeedNextByte;

    'outer: loop {

        // if we have a byte stream like this: 
        // [0, 0, 1, 1, 1, | 0, 0, 0, 0] = 6
        // ^^^^^^^^^^^^^^^^  ^^^^^^^^^^
        // actual number     padding (ignored)
        // 
        // current_bit = 0
        // leading_zeros = 2
        // bits_to_read = 3
        // total = 5
        //
        // 7 = 0b111
        // 6 = (7 - 1)
        match status {
            Status::Ok => {

                /*
                let needed_bytes_backlog = if current_bit.is_positive() {
                    1
                } else {
                    (current_bit / 8).abs() as usize
                };
                */
                
                {   
                    let last_byte = match decoder.last_bytes.last() {
                        Some(s) => s,
                        None => break 'outer,
                    };
                    let leading_z = last_byte.leading_zeros() as usize;
                    let bits_to_read = leading_z + 1;
                    
                    if (current_bit + leading_z as isize + bits_to_read as isize) > 8 {
                        status = Status::NeedNextByte;
                        current_bit = 0;
                        continue;
                    } else {

                        // calculate the actual number
                        let mut num: usize = 0;
                        let mut bit_counter = 0;

                        let start = current_bit as usize + leading_z;
                        let end = current_bit as usize + leading_z + bits_to_read;

                        for i in start..end {
                            let last_bit = (last_byte >> (7 - i))  & 0x01;
                            num |= (last_bit as usize) << (bits_to_read as isize -1 - (bit_counter as isize));
                            bit_counter += 1;
                        }

                        num -= 1;
                        result.push(num);
                        status = Status::Ok;
                    }
                }
                decoder.last_bytes = Vec::new();
            },
            Status::NeedNextByte => {
                if let Some(max) = max_bytes {
                    if result.len() == max {
                        break 'outer;
                    }
                }

                if let Ok(next_byte) = decoder.data.read_u8() {
                    decoder.last_bytes.push(next_byte);
                    status = Status::Ok;
                } else {
                    break 'outer;
                }
            }
        }
    }

    Ok(result)
}

pub fn exp_golomb_encode(_data: &Vec<usize>) -> Result<Vec<u8>, ExpGolombEncodeError> {
    Ok(Vec::new())
}

#[test]
fn test_exp_golomb_decode_multiple_values() {
    let mut c = Cursor::new(vec![0b11111111]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_exp_golomb_decode_single_value() {

    let mut c = Cursor::new(vec![0b10000000]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![0]);

    let mut c = Cursor::new(vec![0b01000000]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![1]);

    let mut c = Cursor::new(vec![0b01100000]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![2]);

    let mut c = Cursor::new(vec![0b00100000]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![3]);

    let mut c = Cursor::new(vec![0b00101000]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![4]);

    let mut c = Cursor::new(vec![0b00110000]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![5]);

    let mut c = Cursor::new(vec![0b00111000]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![6]);

    let mut c = Cursor::new(vec![0b00010000]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![7]);

    let mut c = Cursor::new(vec![0b00010010]);
    let mut decoder = ExpGolombDecoder::new(&mut c);
    assert_eq!(exp_golomb_decode(&mut decoder, None).unwrap(), vec![8]);
}

#[test]
fn test_exp_golomb_encode() {
    
}