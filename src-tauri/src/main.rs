// Точка входа desktop-приложения.
//
// ВАЖНО: здесь не должно быть `mod lib;`. Файл lib.rs компилируется как
// отдельный крейт (crate-type = ["staticlib", "cdylib", "rlib"]), и объявление
// его модулем внутри бинарника даёт ошибку компиляции
// «found module declaration for lib.rs».
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    difri_privacy_lib::run();
}
