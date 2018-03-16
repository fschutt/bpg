extern crate bpg;

pub const TEST_TUX: &'static [u8] = include_bytes!("../assets/tux_with_alpha.bpg");

fn main() {
    /*
    let decoded = bpg::decode(&mut TEST_TUX).unwrap();
    println!("{:?}", decoded);
    */

    use std::io::Cursor;
    let res = bpg::exp_golomb::exp_golomb_decode(&mut Cursor::new(vec![0b00010010]));
    println!("res: {:?}", res);
}