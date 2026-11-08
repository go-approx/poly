//go:build race

package poly

// raceOn: sync.Pool drops items at random under the race detector, so an
// allocation count is not a property of the code there.
const raceOn = true
