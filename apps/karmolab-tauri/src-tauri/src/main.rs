// Hide the extra console window when launching the .exe on Windows (debug and release).
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

fn main() {
    karmolab_desktop_lib::run()
}
