#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Desktop binary entry point — delegates to lib.rs run().
mod lib;
fn main() {
    lib::run();
}
