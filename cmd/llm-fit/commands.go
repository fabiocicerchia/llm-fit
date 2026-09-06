package main

// One function per verb. Each gathers what its subcommand needs and hands it to
// render.go; none of them formats a column.

import (
	"context"
	"fmt"
	"os"

	"github.com/fabiocicerchia/llm-fit/internal/advisor"
	"github.com/fabiocicerchia/llm-fit/internal/arch"
	"github.com/fabiocicerchia/llm-fit/internal/catalog"
	"github.com/fabiocicerchia/llm-fit/internal/engine"
	"github.com/fabiocicerchia/llm-fit/internal/gguf"
	"github.com/fabiocicerchia/llm-fit/internal/hfapi"
	"github.com/fabiocicerchia/llm-fit/internal/hw"
	"github.com/fabiocicerchia/llm-fit/internal/quant"
)

func cmdDetect(m hw.Machine, asJSON bool) {
	if asJSON {
		emitJSON(m)
		return
	}
	printMachine(m)
}

func cmdSuggest(m hw.Machine, req advisor.Request, top int, asJSON bool) {
	opts := advisor.Suggest(m, req)
	if asJSON {
		emitJSON(opts)
		return
	}
	if len(opts) == 0 {
		printNothingFits(req)
		return
	}
	if len(opts) > top {
		opts = opts[:top]
	}
	printSuggestions(m, req, opts)
}

func cmdCheck(
	ctx context.Context, m hw.Machine, req advisor.Request, query string, asJSON, useHF bool,
) {
	model, fileFormat := resolveModel(ctx, query, useHF)

	opts := advisor.Inspect(model, m, req)
	if fileFormat != nil {
		opts = onlyFormat(opts, fileFormat.Name)
	}
	if asJSON {
		emitJSON(opts)
		return
	}
	printModelShape(model)
	printKVCache(model, req)
	printPlanTable(opts)
}

// resolveModel turns the query into a model, from whichever of the three
// sources it names: a GGUF file on disk, a Hugging Face repo id, or the
// built-in catalogue. It does not return on failure.
//
// The returned format is non-nil only for a file, where the quantization is a
// fact about the copy on disk rather than one of the options.
func resolveModel(ctx context.Context, query string, useHF bool) (arch.Model, *quant.Format) {
	switch {
	// A path to a file on disk is unambiguous — no catalogue entry or repo id
	// looks like one — so it needs no flag to select it, and it describes the
	// copy about to be loaded rather than the model in the abstract.
	case isFile(query):
		info, err := gguf.Read(query)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(exitDataErr)
		}
		if f, found := info.Format(); found {
			return info.Model(), &f
		}
		fmt.Fprintf(os.Stderr, "note: %s declares file type %d, which is not in the quant table — "+
			"sizing every format instead of the one in the file\n", query, info.FileType)
		return info.Model(), nil
	case useHF:
		fetched, err := hfapi.Fetch(ctx, query)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(exitUnavailable)
		}
		return fetched, nil
	}
	if model, ok := catalog.Find(query); ok {
		return model, nil
	}
	hits := catalog.Matches(query)
	if len(hits) == 0 {
		fmt.Fprintf(os.Stderr, "no model matching %q. Try: llm-fit models\n", query)
		os.Exit(exitUsage)
	}
	fmt.Fprintf(os.Stderr, "%q matches several models:\n", query)
	for _, h := range hits {
		fmt.Fprintf(os.Stderr, "  %s\n", h.ID)
	}
	os.Exit(exitUsage)
	return arch.Model{}, nil
}

// onlyFormat narrows the plans to the quantization the file is already in: the
// other twenty describe a download the caller has not made. Filtered here
// rather than inside Inspect — the ranking is the advisor's business, the
// narrowing is this command's.
func onlyFormat(opts []advisor.Option, name string) []advisor.Option {
	var only []advisor.Option
	for _, o := range opts {
		if o.Format.Name == name {
			only = append(only, o)
		}
	}
	return only
}

func cmdEngines(m hw.Machine) {
	for _, e := range engine.All {
		ok, why := e.RunsOn(m)
		printEngine(e, ok, why)
	}
}

func cmdModels(asJSON bool) {
	all := catalog.All()
	if asJSON {
		emitJSON(all)
		return
	}
	for _, m := range all {
		printCatalogueEntry(m)
	}
}
