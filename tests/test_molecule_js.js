#!/usr/bin/env node
/**
 * Tests for the JavaScript implementation (molecule.js).
 *
 * Mirrors the Python test suite structure — autograd, tokenizer, model,
 * sampling, and the specific fixes for the Copilot/Codex review items.
 *
 * Run:  node tests/test_molecule_js.js
 */

"use strict";

const path = require("path");
const m = require(path.join(__dirname, "..", "molecule.js"));

const {
    CFG, softmaxProbsFloat, topKTopPSample, VectorValue, ScalarValue,
    backward, withNoGrad, MatrixParam, EvolvingTokenizer, GPT,
    extractCandidateSentences, reservoirMixKeep, normalizeText, rng,
} = m;

let _passed = 0;
let _failed = 0;
const _failures = [];

function assert(cond, msg) {
    if (!cond) throw new Error("Assertion failed: " + msg);
}

function assertAlmostEqual(a, b, tol, msg) {
    if (tol === undefined) tol = 1e-6;
    if (Math.abs(a - b) > tol) {
        throw new Error(`${msg || ""}  expected ≈${b}, got ${a} (tol=${tol})`);
    }
}

function test(name, fn) {
    try {
        fn();
        _passed++;
    } catch (e) {
        _failed++;
        _failures.push({ name, error: e.message });
        console.error(`  FAIL  ${name}: ${e.message}`);
    }
}

// ──────────────────────────────────────────────
// Autograd — VectorValue
// ──────────────────────────────────────────────

test("VectorValue: add two vectors", () => {
    const a = new VectorValue([1, 2, 3]);
    const b = new VectorValue([4, 5, 6]);
    const c = a.add(b);
    assert(c.data[0] === 5 && c.data[1] === 7 && c.data[2] === 9, "add result");
});

test("VectorValue: add scalar", () => {
    const a = new VectorValue([1, 2, 3]);
    const c = a.add(10);
    assert(c.data[0] === 11 && c.data[1] === 12 && c.data[2] === 13, "add scalar");
});

test("VectorValue: mul element-wise", () => {
    const a = new VectorValue([2, 3]);
    const b = new VectorValue([4, 5]);
    const c = a.mul(b);
    assert(c.data[0] === 8 && c.data[1] === 15, "mul result");
});

test("VectorValue: backward propagates gradients via dot", () => {
    const a = new VectorValue([1, 2]);
    const b = new VectorValue([3, 4]);
    const loss = a.dot(b);      // loss = 1*3 + 2*4 = 11
    loss.grad = 1;
    backward(loss);
    // dloss/da = b, dloss/db = a
    assertAlmostEqual(a.grad[0], 3, 1e-9, "grad a[0]");
    assertAlmostEqual(a.grad[1], 4, 1e-9, "grad a[1]");
    assertAlmostEqual(b.grad[0], 1, 1e-9, "grad b[0]");
    assertAlmostEqual(b.grad[1], 2, 1e-9, "grad b[1]");
});

// ──────────────────────────────────────────────
// Autograd — ScalarValue
// ──────────────────────────────────────────────

test("ScalarValue: add", () => {
    const a = new ScalarValue(3);
    const b = new ScalarValue(7);
    const c = a.add(b);
    assert(c.data === 10, "scalar add");
});

test("ScalarValue: mul", () => {
    const a = new ScalarValue(4);
    const b = new ScalarValue(5);
    const c = a.mul(b);
    assert(c.data === 20, "scalar mul");
});

test("ScalarValue: backward", () => {
    const a = new ScalarValue(2);
    const b = new ScalarValue(3);
    const c = a.mul(b);
    c.grad = 1;
    backward(c);
    assertAlmostEqual(a.grad, 3, 1e-9, "grad a");
    assertAlmostEqual(b.grad, 2, 1e-9, "grad b");
});

// ──────────────────────────────────────────────
// Tokenizer
// ──────────────────────────────────────────────

test("EvolvingTokenizer: encode/decode round-trip", () => {
    const tok = new EvolvingTokenizer(["hello world"]);
    const ids = tok.encode("hello");
    const text = tok.decode(ids);
    assert(text.includes("hello"), "round-trip should contain original text");
});

test("EvolvingTokenizer: BOS / EOS tokens exist", () => {
    const tok = new EvolvingTokenizer(["abc"]);
    assert(tok.stoi.has(tok.BOS), "BOS in vocab");
    assert(tok.stoi.has(tok.EOS), "EOS in vocab");
    assert(tok.stoi.has(tok.PAD), "PAD in vocab");
});

test("EvolvingTokenizer: vocab covers input chars", () => {
    const tok = new EvolvingTokenizer(["abcxyz"]);
    for (const ch of "abcxyz") {
        assert(tok.stoi.has(ch), `char '${ch}' in vocab`);
    }
});

test("EvolvingTokenizer: vocabSize matches tokens array", () => {
    const tok = new EvolvingTokenizer(["hello"]);
    assert(tok.vocabSize === tok.tokens.length, "vocabSize == tokens.length");
});

// ──────────────────────────────────────────────
// Sampling — softmax
// ──────────────────────────────────────────────

test("softmaxProbsFloat: sums to 1", () => {
    const probs = softmaxProbsFloat([1, 2, 3, 4]);
    const sum = probs.reduce((a, b) => a + b, 0);
    assertAlmostEqual(sum, 1.0, 1e-6, "softmax sum");
});

test("softmaxProbsFloat: all positive", () => {
    const probs = softmaxProbsFloat([-10, 0, 10]);
    assert(probs.every(p => p >= 0), "all positive");
});

test("softmaxProbsFloat: highest logit gets highest prob", () => {
    const probs = softmaxProbsFloat([1, 5, 2]);
    assert(probs[1] > probs[0] && probs[1] > probs[2], "argmax preserved");
});

test("softmaxProbsFloat: numerical stability with large values", () => {
    const probs = softmaxProbsFloat([1000, 1001, 1002]);
    const sum = probs.reduce((a, b) => a + b, 0);
    assertAlmostEqual(sum, 1.0, 1e-6, "large-value softmax sum");
    assert(probs.every(p => isFinite(p)), "no Inf/NaN");
});

// ──────────────────────────────────────────────
// Sampling — topKTopPSample
// ──────────────────────────────────────────────

test("topKTopPSample: returns valid index", () => {
    const probs = softmaxProbsFloat([1, 2, 3]);
    const idx = topKTopPSample(probs, 0, 1.0, 0, 1.0);
    assert(idx >= 0 && idx < probs.length, "valid index");
});

test("topKTopPSample: top-k=1 always picks argmax", () => {
    const probs = softmaxProbsFloat([0, 0, 100]);
    for (let i = 0; i < 10; i++) {
        const idx = topKTopPSample(probs, 1, 1.0, 0, 1.0);
        assert(idx === 2, "k=1 should always return argmax");
    }
});

test("topKTopPSample: min_p filters low-prob tokens", () => {
    // With very high min_p, only the top token should survive
    const probs = softmaxProbsFloat([0, 0, 100]);
    for (let i = 0; i < 10; i++) {
        const idx = topKTopPSample(probs, 0, 1.0, 0.99, 1.0);
        assert(idx === 2, "min_p should filter to argmax");
    }
});

// ──────────────────────────────────────────────
// Model — GPT constructor (fix: _locked removed)
// ──────────────────────────────────────────────

test("GPT: constructor does NOT set _locked property", () => {
    const tok = new EvolvingTokenizer(["hello world"]);
    const gpt = new GPT(tok);
    assert(!("_locked" in gpt), "_locked should not exist on GPT instances");
});

test("GPT: constructor sets expected properties", () => {
    const tok = new EvolvingTokenizer(["hello world"]);
    const gpt = new GPT(tok);
    assert(gpt.nLayer === CFG.nLayer, "nLayer");
    assert(gpt.nEmbd === CFG.nEmbd, "nEmbd");
    assert(gpt.nHead === CFG.nHead, "nHead");
    assert(gpt.blockSize === CFG.blockSize, "blockSize");
    assert(gpt.headDim === Math.floor(CFG.nEmbd / CFG.nHead), "headDim");
});

test("GPT: base weights exist", () => {
    const tok = new EvolvingTokenizer(["hello"]);
    const gpt = new GPT(tok);
    assert("wte" in gpt.base, "wte");
    assert("wpe" in gpt.base, "wpe");
});

// ──────────────────────────────────────────────
// Model — MatrixParam
// ──────────────────────────────────────────────

test("MatrixParam: correct shape", () => {
    const mp = new MatrixParam(3, 4, 0.1);
    assert(mp.nout === 3, "nout");
    assert(mp.nin === 4, "nin");
    assert(mp.rows.length === 3, "rows count");
    assert(mp.rows[0].data.length === 4, "row width");
});

// ──────────────────────────────────────────────
// Model — generation (fix: no double-processing)
// ──────────────────────────────────────────────

test("GPT: generateSentence returns a string", () => {
    const tok = new EvolvingTokenizer(["hello world."]);
    const gpt = new GPT(tok);
    const out = gpt.generateSentence("hello");
    assert(typeof out === "string", "output is string");
    assert(out.length > 0, "output is non-empty");
});

test("GPT: generateSentence with empty prompt returns a string", () => {
    const tok = new EvolvingTokenizer(["test sentence."]);
    const gpt = new GPT(tok);
    const out = gpt.generateSentence("");
    assert(typeof out === "string", "output is string");
});

// ──────────────────────────────────────────────
// Text utilities — normalizeText
// ──────────────────────────────────────────────

test("normalizeText: collapses whitespace", () => {
    assert(normalizeText("  hello   world  ") === "hello world", "collapse");
});

test("normalizeText: trims", () => {
    assert(normalizeText("  hi  ") === "hi", "trim");
});

// ──────────────────────────────────────────────
// Corpus — extractCandidateSentences
// ──────────────────────────────────────────────

test("extractCandidateSentences: extracts from messages", () => {
    const msgs = [{ text: "Hello there! How are you?" }];
    const sents = extractCandidateSentences(msgs);
    assert(sents.length > 0, "should extract sentences");
});

test("extractCandidateSentences: filters short messages", () => {
    const msgs = [{ text: "Hi" }];     // < 6 chars
    const sents = extractCandidateSentences(msgs);
    assert(sents.length === 0, "short messages filtered out");
});

test("extractCandidateSentences: empty input returns empty", () => {
    assert(extractCandidateSentences([]).length === 0, "empty in → empty out");
});

test("extractCandidateSentences: splits on sentence-enders", () => {
    const msgs = [{ text: "First sentence. Second sentence! Third one?" }];
    const sents = extractCandidateSentences(msgs);
    assert(sents.length >= 2, "should split into multiple sentences");
});

// ──────────────────────────────────────────────
// Corpus — reservoirMixKeep
// ──────────────────────────────────────────────

test("reservoirMixKeep: respects maxLines", () => {
    const existing = ["a", "b", "c", "d", "e"];
    const incoming = ["f", "g", "h", "i", "j"];
    const result = reservoirMixKeep(existing, incoming, 5);
    assert(result.length <= 5, "should not exceed maxLines");
});

test("reservoirMixKeep: deduplicates case-insensitive", () => {
    const existing = ["Hello"];
    const incoming = ["hello"];
    const result = reservoirMixKeep(existing, incoming, 100);
    assert(result.length === 1, "duplicates removed");
});

test("reservoirMixKeep: preserves content", () => {
    const existing = ["alpha"];
    const incoming = ["beta"];
    const result = reservoirMixKeep(existing, incoming, 100);
    const lower = result.map(s => s.toLowerCase());
    assert(lower.includes("alpha") && lower.includes("beta"), "content preserved");
});

// ──────────────────────────────────────────────
// withNoGrad
// ──────────────────────────────────────────────

test("withNoGrad: disables grad allocation", () => {
    const v = withNoGrad(() => new VectorValue([1, 2, 3]));
    assert(v.grad === null, "grad should be null inside withNoGrad");
});

test("withNoGrad: grad re-enabled after", () => {
    withNoGrad(() => {});
    const v = new VectorValue([1, 2]);
    assert(v.grad !== null, "grad should be enabled after withNoGrad");
});

// ──────────────────────────────────────────────
// Report
// ──────────────────────────────────────────────

console.log(`\n${"=".repeat(50)}`);
console.log(`  molecule.js tests: ${_passed} passed, ${_failed} failed`);
console.log(`${"=".repeat(50)}`);

if (_failures.length > 0) {
    console.log("\nFailures:");
    for (const f of _failures) {
        console.log(`  - ${f.name}: ${f.error}`);
    }
    process.exit(1);
}
