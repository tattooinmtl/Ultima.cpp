package main

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os/exec"
	"runtime"
	"sync"
)

// launcherState bundles config + runtime manager for the handlers.
type launcherState struct {
	mu       sync.Mutex
	cfg      Config
	cfgPath  string
	rt       *RuntimeManager
}

func serve(addr string, cfg Config, cfgPath string, rt *RuntimeManager) error {
	state := &launcherState{cfg: cfg, cfgPath: cfgPath, rt: rt}
	mux := http.NewServeMux()

	// ---- UI (embedded HTML) ----
	sub, err := fs.Sub(uiFiles, "ui")
	if err != nil {
		return fmt.Errorf("embed ui/: %w", err)
	}
	mux.Handle("/", http.FileServer(http.FS(sub)))

	// ---- API ----
	mux.HandleFunc("/api/config",  state.handleConfig)
	mux.HandleFunc("/api/models",  state.handleModels)
	mux.HandleFunc("/api/rescan",  state.handleRescan)
	mux.HandleFunc("/api/load",    state.handleLoad)
	mux.HandleFunc("/api/unload",  state.handleUnload)
	mux.HandleFunc("/api/status",  state.handleStatus)
	mux.HandleFunc("/api/open",    state.handleOpen)

	srv := &http.Server{Addr: addr, Handler: mux}
	return srv.ListenAndServe()
}

// ---- Handlers --------------------------------------------------------------

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *launcherState) handleConfig(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.mu.Lock()
		cfg := s.cfg
		s.mu.Unlock()
		writeJSON(w, 200, cfg)
	case http.MethodPost:
		var incoming Config
		if err := json.NewDecoder(r.Body).Decode(&incoming); err != nil {
			writeJSON(w, 400, map[string]string{"error": err.Error()})
			return
		}
		s.mu.Lock()
		s.cfg = incoming
		path := s.cfgPath
		s.mu.Unlock()
		if err := saveConfig(incoming, path); err != nil {
			writeJSON(w, 500, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, 200, incoming)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (s *launcherState) handleModels(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	roots := append([]string(nil), s.cfg.ModelRoots...)
	loaded := s.rt.Status()
	s.mu.Unlock()
	models := discoverModels(roots)
	writeJSON(w, 200, map[string]any{
		"models":     models,
		"roots":      roots,
		"loaded":     loaded,
	})
}

func (s *launcherState) handleRescan(w http.ResponseWriter, r *http.Request) {
	// Discovery has no cache in v0.1 — this endpoint exists so the UI can
	// re-issue the walk without a page reload.
	s.handleModels(w, r)
}

func (s *launcherState) handleLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		Path string `json:"path"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, 400, map[string]string{"error": err.Error()})
		return
	}
	if body.Path == "" {
		writeJSON(w, 400, map[string]string{"error": "path required"})
		return
	}
	s.mu.Lock()
	host := s.cfg.RuntimeHost
	port := s.cfg.RuntimePort
	nCtx := s.cfg.RuntimeNCtx
	nT := s.cfg.RuntimeNThreads
	openPad := s.cfg.OpenTestpadOnLoad
	s.mu.Unlock()

	info, err := s.rt.Load(body.Path, host, port, nCtx, nT)
	if err != nil {
		writeJSON(w, 500, map[string]string{"error": err.Error()})
		return
	}
	testpad := fmt.Sprintf("http://%s:%d/?bench=1", host, port)
	if openPad {
		go func() { _ = tryOpenBrowser(testpad) }()
	}
	writeJSON(w, 200, map[string]any{
		"loaded":  info,
		"testpad": testpad,
	})
}

func (s *launcherState) handleUnload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	s.rt.Unload()
	writeJSON(w, 200, map[string]string{"status": "unloaded"})
}

func (s *launcherState) handleStatus(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, s.rt.Status())
}

func (s *launcherState) handleOpen(w http.ResponseWriter, r *http.Request) {
	info := s.rt.Status()
	if info.Port == 0 {
		writeJSON(w, 400, map[string]string{"error": "no model loaded"})
		return
	}
	url := fmt.Sprintf("http://%s:%d/", info.Host, info.Port)
	go func() { _ = tryOpenBrowser(url) }()
	writeJSON(w, 200, map[string]string{"url": url})
}

// tryOpenBrowser opens `url` in the user's default browser. Best-effort.
func tryOpenBrowser(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	if err := cmd.Start(); err != nil {
		log.Printf("failed to open browser at %s: %v", url, err)
		return err
	}
	return nil
}
