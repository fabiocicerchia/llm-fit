package hfapi

import (
	"net/url"
	"strings"
	"testing"
)

// The id comes off the command line and is interpolated into the request path.
// The cases that matter are the ones that would leave the model-repo namespace.
func TestValidateIDRejectsAnythingThatIsNotOwnerSlashName(t *testing.T) {
	good := []string{
		"Qwen/Qwen3-8B",
		"meta-llama/Llama-3.1-70B-Instruct",
		"mistralai/Mixtral-8x7B-v0.1",
		"deepseek-ai/DeepSeek-V3",
		"google/gemma-2-2b-it",
		"my_org/model.v2",
	}
	for _, id := range good {
		if err := ValidateID(id); err != nil {
			t.Errorf("ValidateID(%q) rejected a real repo id: %v", id, err)
		}
	}

	bad := []string{
		"",                      // nothing
		"Qwen3-8B",              // no owner
		"Qwen/Qwen3/8B",         // not a two-segment id
		"../../etc/passwd",      // climbs out of the repo path
		"Qwen/..",               // ditto, one segment at a time
		"../secrets",            // ditto
		"Qwen/Qwen3-8B?x=1",     // smuggles a query string
		"Qwen/Qwen3-8B#frag",    // smuggles a fragment
		"Qwen/Qwen3 8B",         // space
		"Qwen/Qwen3%2f8B",       // percent-encoded separator
		"evil.com/x",            // '.' is legal in a segment, ':' is not...
		"https://evil.com/x",    // ...so a full URL cannot pass
		"//evil.com/x",          // protocol-relative
		"Qwen/\nHost: evil.com", // header injection shape
	}
	for _, id := range bad {
		if err := ValidateID(id); err == nil {
			// "evil.com/x" is a legal *shape*; it is only ever a path under
			// huggingface.co, which is what the URL check below asserts.
			if id == "evil.com/x" {
				continue
			}
			t.Errorf("ValidateID(%q) accepted an id it should not have", id)
		}
	}
}

// The point of the allowlist is that an accepted id cannot move the request off
// huggingface.co or above the repo path. Assert that directly rather than
// trusting the character rules to imply it.
func FuzzValidateIDKeepsTheURLOnHuggingFace(f *testing.F) {
	for _, s := range []string{"Qwen/Qwen3-8B", "../..", "a/b", "", "%2e%2e/x", "a/../../b"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, id string) {
		if ValidateID(id) != nil {
			return
		}
		raw := base + "/" + id + "/resolve/main/config.json"
		u, err := url.Parse(raw)
		if err != nil {
			t.Fatalf("accepted id %q produced an unparsable URL %q: %v", id, raw, err)
		}
		if u.Scheme != "https" || u.Host != "huggingface.co" {
			t.Fatalf("accepted id %q redirected the request to %s://%s", id, u.Scheme, u.Host)
		}
		if u.RawQuery != "" || u.Fragment != "" {
			t.Fatalf("accepted id %q smuggled a query/fragment: %q", id, raw)
		}
		// url.Parse resolves "." and ".." while cleaning; if the path still has
		// the shape we asked for, nothing climbed out.
		if !strings.HasSuffix(u.Path, "/resolve/main/config.json") {
			t.Fatalf("accepted id %q escaped the repo path: %q", id, u.Path)
		}
		if strings.Count(strings.TrimPrefix(u.Path, "/"), "/") != 4 {
			t.Fatalf("accepted id %q changed the path depth: %q", id, u.Path)
		}
	})
}
