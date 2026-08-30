// Ultima launcher — a small Go binary that discovers GGUF files across the
// user's disk, spawns the Ultima runtime (ultima.exe) with a selected model,
// and opens the runtime testpad in a browser tab.
//
// v0.1 ships as an HTTP-based launcher (bind on 127.0.0.1:11435 by default)
// rather than a Wails-native window: much simpler to build, hits the ship
// deadline, and the browser is the eventual UI surface anyway
// (Decision 03b). Migrating to a native Wails window is a v0.2 concern.
package main

import (
	"embed"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

//go:embed ui/*
var uiFiles embed.FS

var version = "0.1.0-alpha"

func main() {
	var (
		bind        = flag.String("bind", "127.0.0.1", "launcher UI bind host")
		port        = flag.Int("port", 11435, "launcher UI port")
		runtimePath = flag.String("runtime", defaultRuntimePath(), "path to ultima.exe")
		openBrowser = flag.Bool("open", true, "open the launcher UI in the default browser at startup")
		printVer    = flag.Bool("version", false, "print version and exit")
	)
	flag.Parse()

	if *printVer {
		fmt.Printf("ultima-launcher %s\n", version)
		return
	}

	if _, err := os.Stat(*runtimePath); err != nil {
		log.Printf("warning: runtime binary not found at %s (searched via --runtime)", *runtimePath)
		log.Printf("         model launch will fail until this is corrected in the UI")
	}

	cfg, cfgPath := loadOrInitConfig()
	log.Printf("launcher config: %s", cfgPath)

	rt := NewRuntimeManager(*runtimePath)

	addr := net.JoinHostPort(*bind, fmt.Sprintf("%d", *port))
	url := "http://" + addr + "/"
	log.Printf("ultima-launcher %s serving %s", version, url)
	log.Printf("runtime binary: %s", *runtimePath)

	if *openBrowser {
		// Small delay so the browser doesn't beat the server to bind.
		go func() {
			time.Sleep(300 * time.Millisecond)
			_ = tryOpenBrowser(url)
		}()
	}

	if err := serve(addr, cfg, cfgPath, rt); err != nil {
		log.Fatal(err)
	}
}

// defaultRuntimePath returns the expected `ultima` binary location relative
// to the launcher binary — matches Decision 03 §3.7's search order 1
// (co-located portable folder).
func defaultRuntimePath() string {
	exe, err := os.Executable()
	if err != nil {
		return "ultima"
	}
	dir := filepath.Dir(exe)
	name := "ultima"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(dir, name)
}
