---
id: egui-eframe
title: egui + eframe
type: howto
tags: [egui, rust, ui]
links: []
---
# egui + eframe

`eframe` is the official desktop/web framework for `egui`. It runs on `winit` and renders via `wgpu` or `glow`. Use `eframe::run_native` to start the event loop. `App::update` is called every frame; `CentralPanel` / `SidePanel` / `Window` divide the screen.