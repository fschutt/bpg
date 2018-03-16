//! Exp-golomb coding

use byteorder::{BigEndian, ReadBytesExt};

pub struct ExpGolombEncoder {

}

pub struct ExpGolombDecoder {

}

#[derive(Debug, Clone)]
pub enum ExpGolombEncodeError {

}

pub fn exp_golomb_decode<R: ReadBytesExt>(data: &mut R) -> Result<Vec<usize>, ExpGolombEncodeError> {
    
    let mut result = Vec::<usize>::new();
    let mut current_bit: isize = 0; // carry bit for in-between byte-encoding
    let mut last_bytes: Option<Vec<u8>> = None;

    #[derive(Debug, PartialEq)]
    enum Status {
        Ok,
        NeedNextByte,
    }

    let mut status = Status::NeedNextByte;

    let mut debug_loop_counter = 0;

    'outer: loop {

        println!("entering loop {}", debug_loop_counter);

        // if we have a byte stream like this: 
        // [0, 0, 1, 1, 1, | 0, 0, 0, 0] = 6
        // ^^^^^^^^^^^^^^^^  ^^^^^^^^^^
        // actual number     padding (ignored)
        // 
        // current_bit = 0
        // leading_zeros = 2
        // bits_to_read = 3
        // total = 5
        // start_position = 
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
                    let last_b: &Vec<u8> = match last_bytes.as_ref() {
                        Some(s) => s,
                        None => break 'outer,
                    };
                    let last_byte = last_b[last_b.len() - 1];
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
                last_bytes = None;
            },
            Status::NeedNextByte => {
                // push an extra byte
                if let Ok(next_byte) = data.read_u8() {
                    match last_bytes {
                        None => { 
                            last_bytes = Some(vec![next_byte]); 
                        }
                        Some(ref mut l) => { 
                            l.push(next_byte); 
                        } 
                    }
                    status = Status::Ok;
                } else {
                    break 'outer;
                }
            }
        }

        debug_loop_counter += 1;
    }

    Ok(result)
}

pub fn exp_golomb_encode(data: &Vec<usize>) -> Result<Vec<u8>, ExpGolombEncodeError> {
    Ok(Vec::new())
}

#[test]
fn test_exp_golomb_decode() {
    use std::io::Cursor;
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b10000000])).unwrap(), vec![0]);
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b01000000])).unwrap(), vec![1]);
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b01100000])).unwrap(), vec![2]);
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b00100000])).unwrap(), vec![3]);
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b00101000])).unwrap(), vec![4]);
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b00110000])).unwrap(), vec![5]);
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b00111000])).unwrap(), vec![6]);
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b00010000])).unwrap(), vec![7]);
    assert_eq!(exp_golomb_decode(&mut Cursor::new(vec![0b00010010])).unwrap(), vec![8]);
}

#[test]
fn test_exp_golomb_encode() {
    
}