package main

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
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
	appdata := os.Getenv("APPDATA")
	if appdata != "" {
		roots = append(roots, filepath.Join(appdata, "Ultima", "models"))
	}
	// Portable co-location: <launcher_dir>/models
	if exe, err := os.Executable(); err == nil {
		roots = append(roots, filepath.Join(filepath.Dir(exe), "models"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		roots = append(roots,
			filepath.Join(home, ".ollama", "models"),
			filepath.Join(home, ".cache", "lm-studio", "models"),
		)
	}
	return Config{
		ModelRoots:              roots,
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
	if b, err := os.ReadFile(path); err == nil {
		var c Config
		if err := json.Unmarshal(b, &c); err == nil {
			return c, path
		}
		log.Printf("launcher config parse failed; using defaults: %v", err)
	}
	c := defaultConfig()
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
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
