---
id: ripgrep-usage
title: ripgrep
type: howto
tags: [ripgrep, cli, intermediate]
links: []
---
# ripgrep

`rg pattern` is 10× faster than `grep -r`. `-i` case-insensitive, `-l` files only, `-c` count, `-n` line numbers, `-C 3` context, `-t rs` filter to file type. `rg --hidden` to include dotfiles. `rg -g '!target/**'` to exclude paths.