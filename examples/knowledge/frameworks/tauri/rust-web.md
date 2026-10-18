---
id: tauri-rust-web
title: Tauri (Rust + Webview)
type: howto
tags: [tauri, rust, desktop]
links: []
---
# Tauri

Tauri ships a Rust backend + a webview-based UI. `tauri::Builder::default().invoke_handler(...)` registers commands callable from the JS side. Smaller binaries than Electron because no Chromium runtime.