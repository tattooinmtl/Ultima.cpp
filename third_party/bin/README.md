# Bundled binaries

Prebuilt native tools shipped with Ultima. See Decision 02 §2.5 in `roadmap01.md`.

Populated by `scripts/refresh_bundled_binaries.*` (added post-scaffold).

## Currently expected (empty until populated)

| Platform      | Binary   | Source                                                |
|---------------|----------|-------------------------------------------------------|
| windows-x64   | rg.exe   | https://github.com/BurntSushi/ripgrep/releases        |
| windows-x64   | fd.exe   | https://github.com/sharkdp/fd/releases                |
| linux-x64     | rg       | same                                                  |
| linux-x64     | fd       | same                                                  |
| linux-arm64   | rg       | same                                                  |
| linux-arm64   | fd       | same                                                  |
| macos-arm64   | rg       | same                                                  |
| macos-arm64   | fd       | same                                                  |

## Version pin

See `VERSIONS.txt`.
