package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DiscoveredModel is one .gguf file the launcher found on disk.
type DiscoveredModel struct {
	Name       string    `json:"name"`         // basename without .gguf
	Path       string    `json:"path"`         // absolute path
	Root       string    `json:"root"`         // which discovery root it belongs to
	Size       int64     `json:"size"`         // bytes
	Modified   time.Time `json:"modified"`
	Arch       string    `json:"arch"`         // empty until we've inspected — we defer that
	SizeHuman  string    `json:"size_human"`   // "4.5 GB" etc.
}

// discoverModels walks the configured roots (first-match-wins on basename).
// Missing roots are skipped silently — a fresh install has none of the
// user-added folders and that's fine.
func discoverModels(roots []string) []DiscoveredModel {
	out := []DiscoveredModel{}
	seen := map[string]bool{}
	for _, root := range roots {
		if root == "" {
			continue
		}
		info, err := os.Stat(root)
		if err != nil || !info.IsDir() {
			continue
		}
		_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.IsDir() {
				return nil
			}
			if !strings.EqualFold(filepath.Ext(d.Name()), ".gguf") {
				return nil
			}
			fi, err := d.Info()
			if err != nil {
				return nil
			}
			key := strings.ToLower(d.Name())
			if seen[key] {
				return nil
			}
			seen[key] = true
			out = append(out, DiscoveredModel{
				Name:      strings.TrimSuffix(d.Name(), filepath.Ext(d.Name())),
				Path:      path,
				Root:      root,
				Size:      fi.Size(),
				Modified:  fi.ModTime(),
				SizeHuman: humanBytes(fi.Size()),
			})
			return nil
		})
	}
	return out
}

func humanBytes(n int64) string {
	const (
		kib = 1024
		mib = kib * 1024
		gib = mib * 1024
	)
	switch {
	case n >= gib:
		return formatFloat(float64(n)/float64(gib)) + " GB"
	case n >= mib:
		return formatFloat(float64(n)/float64(mib)) + " MB"
	case n >= kib:
		return formatFloat(float64(n)/float64(kib)) + " KB"
	default:
		return formatFloat(float64(n)) + " B"
	}
}

func formatFloat(f float64) string {
	// Two decimals, drop trailing zeros for readability.
	s := ""
	if f >= 100 {
		s = intString(int(f + 0.5))
	} else if f >= 10 {
		s = trimZeros(oneDecimal(f))
	} else {
		s = trimZeros(twoDecimal(f))
	}
	return s
}

func intString(n int) string      { return itoa(n) }
func twoDecimal(f float64) string { return floatToStr(f, 2) }
func oneDecimal(f float64) string { return floatToStr(f, 1) }

func floatToStr(f float64, dp int) string {
	// Small dp only.
	if dp < 0 {
		dp = 0
	}
	mult := 1
	for i := 0; i < dp; i++ {
		mult *= 10
	}
	scaled := int(f*float64(mult) + 0.5)
	whole := scaled / mult
	frac := scaled % mult
	if dp == 0 {
		return itoa(whole)
	}
	fracStr := itoa(frac)
	for len(fracStr) < dp {
		fracStr = "0" + fracStr
	}
	return itoa(whole) + "." + fracStr
}

func trimZeros(s string) string {
	if !strings.Contains(s, ".") {
		return s
	}
	s = strings.TrimRight(s, "0")
	s = strings.TrimRight(s, ".")
	if s == "" {
		s = "0"
	}
	return s
}

// itoa without importing strconv (we only need decimal ints).
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	buf := [20]byte{}
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
