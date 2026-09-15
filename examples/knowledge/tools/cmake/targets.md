---
id: cmake-targets
title: CMake Targets
type: reference
tags: [cmake, intermediate]
links: []
---
# Modern CMake

Declare targets with `add_executable` / `add_library`. Use `target_link_libraries(my PRIVATE fmt::fmt)` — the `PRIVATE`/`PUBLIC` keywords propagate dependencies. `target_compile_features(my PUBLIC cxx_std_20)`. Avoid global `include_directories` and `link_libraries`.