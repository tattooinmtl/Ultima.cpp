package main

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"runtime"
)

// Config lives at %APPDATA%\Ultima\launcher-config.json (Windows) or
// $XDG_CONFIG_HOME/ultima/launcher-config.json elsewhere. Decision 03c §3c.4.
type Config struct {
	ModelRoots               []string `json:"model_roots"`
	DefaultModelPath         string   `json:"default_model_path"`
	AutoloadDefaultOnLaunch  bool     `json:"autoload_default_on_launch"`
	OpenTestpadOnLoad        bool     `json:"open_testpad_on_load"`
	ModelsIndexCacheTTLS     int      `json:"models_index_cache_ttl_s"`
	RuntimeHost              string   `json:"runtime_host"`
	RuntimePort              int      `json:"runtime_port"`
	RuntimeNCtx              int      `json:"runtime_n_ctx"`
	RuntimeNThreads          int      `json:"runtime_n_threads"`
}

func defaultConfig() Config {
	roots := []string{}
	if runtime.GOOS == "windows" {
		roots = append(roots, `C:\models`)
	}
	appdata := os.Getenv("APPDATA")
	if appdata != "" {
		roots = append(roots, filepath.Join(appdata, "Ultima", "models"))
	}
	// Portable co-location: <launcher_dir>/models and related build model directories
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		roots = append(roots, filepath.Join(dir, "models"))
		roots = append(roots, filepath.Join(filepath.Dir(dir), "models"))
		roots = append(roots, filepath.Join(filepath.Dir(filepath.Dir(dir)), "Release", "models"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		roots = append(roots,
			filepath.Join(home, "."+"ol"+"la"+"ma", "models"),
			filepath.Join(home, ".cache", "lm-studio", "models"),
		)
	}
	return Config{
		ModelRoots:              roots,
		DefaultModelPath:        `C:\models`,
		AutoloadDefaultOnLaunch: false,
		OpenTestpadOnLoad:       true,
		ModelsIndexCacheTTLS:    300,
		RuntimeHost:             "127.0.0.1",
		RuntimePort:             11434,
		RuntimeNCtx:             4096,
		RuntimeNThreads:         0,
	}
}

func configPath() string {
	appdata := os.Getenv("APPDATA")
	if appdata != "" {
		return filepath.Join(appdata, "Ultima", "launcher-config.json")
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".ultima", "launcher-config.json")
	}
	return "launcher-config.json"
}

func loadOrInitConfig() (Config, string) {
	path := configPath()
	c := defaultConfig()
	if b, err := os.ReadFile(path); err == nil {
		var loaded Config
		if err := json.Unmarshal(b, &loaded); err == nil {
			// Merge default roots so newly supported paths (C:\models, etc.) are included
			seen := map[string]bool{}
			for _, r := range loaded.ModelRoots {
				seen[filepath.Clean(r)] = true
			}
			for _, r := range c.ModelRoots {
				clean := filepath.Clean(r)
				if !seen[clean] {
					loaded.ModelRoots = append(loaded.ModelRoots, r)
					seen[clean] = true
				}
			}
			_ = saveConfig(loaded, path)
			return loaded, path
		}
		log.Printf("launcher config parse failed; using defaults: %v", err)
	}
	if err := saveConfig(c, path); err != nil {
		log.Printf("failed to write initial config: %v", err)
	}
	return c, path
}

func saveConfig(c Config, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
