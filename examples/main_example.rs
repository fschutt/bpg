extern crate bpg;

pub const TEST_LENA: &'static [u8] = include_bytes!("../assets/lena_q36.bpg");

fn main() {
    // let decoded = bpg::decode(&mut TEST_LENA).unwrap();
    // println!("{:#?}", decoded);
    let width = bpg::ue7_decode(&[0x08]);
    println!("{:?}", width);
}