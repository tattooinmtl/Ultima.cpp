---
id: cpp-raii
title: C++ RAII
type: pattern
tags: [cpp, memory, intermediate]
links: []
---
# RAII

Resource Acquisition Is Initialization. Bind resource lifetime to object lifetime: open in the constructor, close in the destructor. Smart pointers (`unique_ptr`, `shared_ptr`) are RAII wrappers for raw pointers. Use `make_unique` and `make_shared`, not `new`.