//go:build oracle

package oracle

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
)

// UV is a persistent "uv run python -c Program" child process; it implements
// Transport. It is only built with the "oracle" tag because it needs uv and
// network-cached NumPy/SciPy wheels.
type UV struct {
	cmd *exec.Cmd
	in  *bufio.Writer
	out *bufio.Reader
	sin interface{ Close() error }
}

// StartUV starts the oracle with numpy==NumPyVersion and scipy==SciPyVersion.
// The child runs on one core (OPENBLAS/OMP/MKL thread counts are set to 1;
// these are not secrets).
func StartUV() (*UV, error) {
	cmd := exec.Command("uv", "run", "--with", "numpy=="+NumPyVersion, "--with", "scipy=="+SciPyVersion,
		"python", "-c", Program)
	cmd.Env = append(os.Environ(), "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1")
	cmd.Stderr = os.Stderr
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &UV{cmd: cmd, in: bufio.NewWriter(in), out: bufio.NewReaderSize(out, 1<<20), sin: in}, nil
}

// Do implements Transport.
func (u *UV) Do(req []byte) ([]byte, error) {
	if _, err := u.in.Write(append(req, '\n')); err != nil {
		return nil, err
	}
	if err := u.in.Flush(); err != nil {
		return nil, err
	}
	line, err := u.out.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("oracle: reading answer: %w", err)
	}
	return line, nil
}

// Close ends the child process.
func (u *UV) Close() error {
	u.sin.Close()
	return u.cmd.Wait()
}
