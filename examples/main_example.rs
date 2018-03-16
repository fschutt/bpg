extern crate bpg;

pub const TEST_TUX: &'static [u8] = include_bytes!("../assets/tux_with_alpha.bpg");

fn main() {
    let decoded = bpg::decode(&mut TEST_TUX).unwrap();
    println!("{:?}", decoded);
}