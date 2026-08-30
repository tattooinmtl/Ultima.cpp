package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"time"
)

// RuntimeManager spawns and kills the ultima.exe subprocess.
type RuntimeManager struct {
	runtimeBin string
	mu         sync.Mutex
	cmd        *exec.Cmd
	stopFn     context.CancelFunc
	loaded     LoadedInfo
}

type LoadedInfo struct {
	Path string `json:"path"`
	Host string `json:"host"`
	Port int    `json:"port"`
	PID  int    `json:"pid"`
}

func NewRuntimeManager(binPath string) *RuntimeManager {
	return &RuntimeManager{runtimeBin: binPath}
}

// Load spawns `ultima serve <path>` and waits until its /api/health responds
// (or times out). If a previous runtime is running, it's killed first.
func (r *RuntimeManager) Load(path, host string, port, nCtx, nThreads int) (LoadedInfo, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.cmd != nil && r.cmd.Process != nil {
		r.killLocked()
	}

	if _, err := os.Stat(r.runtimeBin); err != nil {
		return LoadedInfo{}, fmt.Errorf("runtime binary not found: %s", r.runtimeBin)
	}
	if _, err := os.Stat(path); err != nil {
		return LoadedInfo{}, fmt.Errorf("model file not found: %s", path)
	}

	args := []string{"serve", path,
		"--host", host,
		"--port", strconv.Itoa(port),
		"--n-ctx", strconv.Itoa(nCtx),
	}
	if nThreads > 0 {
		args = append(args, "--n-threads", strconv.Itoa(nThreads))
	}

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, r.runtimeBin, args...)
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	if err := cmd.Start(); err != nil {
		cancel()
		return LoadedInfo{}, fmt.Errorf("start ultima.exe: %w", err)
	}

	// Copy runtime logs to the launcher's log so the user can see them.
	go pipeLogs("runtime.out", stdout)
	go pipeLogs("runtime.err", stderr)

	// Wait for /api/health to answer.
	healthy := waitForPort(host, port, 15*time.Second)
	if !healthy {
		_ = cmd.Process.Kill()
		cancel()
		return LoadedInfo{}, fmt.Errorf("runtime did not become healthy within 15s")
	}

	r.cmd = cmd
	r.stopFn = cancel
	r.loaded = LoadedInfo{Path: path, Host: host, Port: port, PID: cmd.Process.Pid}
	log.Printf("runtime loaded: pid=%d model=%s at %s:%d", cmd.Process.Pid, path, host, port)

	// Watchdog: reap the process when it exits so we don't leak zombies.
	go func(c *exec.Cmd, info LoadedInfo) {
		err := c.Wait()
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.cmd == c {
			r.cmd = nil
			r.stopFn = nil
			r.loaded = LoadedInfo{}
			log.Printf("runtime exited (pid=%d): %v", info.PID, err)
		}
	}(cmd, r.loaded)

	return r.loaded, nil
}

// Unload kills the running runtime. Safe to call when nothing is loaded.
func (r *RuntimeManager) Unload() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.killLocked()
}

func (r *RuntimeManager) killLocked() {
	if r.cmd == nil {
		return
	}
	if r.stopFn != nil {
		r.stopFn()
	}
	if r.cmd.Process != nil {
		_ = r.cmd.Process.Kill()
	}
	r.cmd = nil
	r.stopFn = nil
	r.loaded = LoadedInfo{}
}

// Status returns the currently-loaded model info (empty if none).
func (r *RuntimeManager) Status() LoadedInfo {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.loaded
}

func pipeLogs(prefix string, rc io.ReadCloser) {
	if rc == nil {
		return
	}
	defer rc.Close()
	buf := make([]byte, 4096)
	for {
		n, err := rc.Read(buf)
		if n > 0 {
			log.Printf("[%s] %s", prefix, trimTrailingNL(string(buf[:n])))
		}
		if err != nil {
			return
		}
	}
}

func trimTrailingNL(s string) string {
	for len(s) > 0 && (s[len(s)-1] == '\n' || s[len(s)-1] == '\r') {
		s = s[:len(s)-1]
	}
	return s
}

func waitForPort(host string, port int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	addr := net.JoinHostPort(host, strconv.Itoa(port))
	for time.Now().Before(deadline) {
		c, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err == nil {
			_ = c.Close()
			return true
		}
		time.Sleep(200 * time.Millisecond)
	}
	return false
}
