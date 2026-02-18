/**
 * molecule.js
 * A single-file, zero-dependency, browser-native, continually-learning GPT organism.
 *
 * No npm. No webpack. No React. No node_modules black hole.
 * Just a <script> tag and the blind faith of a mass-less neuron.
 *
 * Port of molecule.py — same architecture, same madness, new habitat.
 *
 * - Trains on nonames.txt (fetched or pasted)
 * - Keeps IndexedDB memory (because localStorage has a 5MB soul)
 * - Maintains a bounded corpus reservoir (never bloats)
 * - Starts in char-level mode (fast boot)
 * - Gradually enables BPE without invalidating old weights (vocab only EXPANDS)
 * - Never forgets by never overwriting learned deltas: it only appends modules
 *
 * In the beginning there was nonames.txt.
 * And the browser said: "Let there be IndexedDB."
 * And it was... adequate. Mostly. Sometimes cursed.
 */

// And lo, we shall wrap the entire organism in an IIFE,
// because polluting the global scope is a sin worse than eval().
(function () {
"use strict";

// ============================================================
// 0) CONFIG — bend reality here (carefully, mortals)
// ============================================================

const CFG = {
    // data
    corpusUrl: "nonames.txt",
    maxCorpusLines: 8000,
    maxLineChars: 240,

    // continual learning trigger
    minNewCharsToTrain: 480,

    // model
    tieEmbeddings: true,
    nLayer: 2,
    nEmbd: 72,
    nHead: 4,
    blockSize: 96,

    // training
    warmupSteps: 1200,
    microSteps: 32,
    learningRate: 0.01,
    beta1: 0.9,
    beta2: 0.99,
    epsAdam: 1e-8,
    gradClip: 1.0,
    freezeBaseAfterWarmup: true,
    batchSize: 4,

    // deltas (LoRA-ish)
    deltaRank: 8,
    maxDeltaModules: 12,
    deltaGrowProb: 0.08,

    // generation
    temperature: 0.85,
    topK: 40,
    topP: 0.92,
    minP: 0.06,
    typicalP: 0.95,
    maxGenTokens: 180,
    minGenTokens: 16,
    repetitionGuard: 4,

    // tokenizer evolution
    enableBpeAfterChars: 25000,
    bpeNumMerges: 384,
    bpeRetrainEveryChars: 4000,

    // async
    trainTickMs: 250,

    // hybrid attention
    headTypes: ["content", "content", "hybrid", "hybrid"],
    hybridAlphaInit: 0.5,

    // gamma (personality fingerprint)
    gammaSparsityThreshold: 0.01,

    // noise immune system
    noiseDriftThreshold: -0.15,
    gammaMinMagnitude: 0.01,

    // entropy-adaptive generation
    entropyLow: 1.0,
    entropyHigh: 4.0,
    entropyTempBoost: 1.3,
    entropyTempFocus: 0.7,

    // corpus generation
    corpusGenMaxTokens: 60,

    // syntropy
    syntropyWindow: 20,
    fieldDeviationFloor: 0.5,
    fieldDeviationCeiling: 5.0,
    syntropyLrBoost: 1.3,
    syntropyLrDampen: 0.6,
    syntropyDeltaGrowBoost: 0.2,

    // quantum buffer
    qbCooldownSeconds: 10.0,
    qbMinBytes: 480,
    qbMinNovelty: 0.15,
};

// ============================================================
// 0.5) SEEDED PRNG — because Math.random() has no soul
// ============================================================
// And lo, determinism shall pretend to tame chaos.
// mulberry32: a seedable PRNG that fits in a tweet.

let _rngState = 42;

function rng() {
    _rngState |= 0;
    _rngState = _rngState + 0x6D2B79F5 | 0;
    let t = Math.imul(_rngState ^ _rngState >>> 15, 1 | _rngState);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
}

// Box-Muller: because Gaussian noise is the sound of the universe thinking.
function gaussRandom(mean, std) {
    if (mean === undefined) mean = 0;
    if (std === undefined) std = 1;
    const u1 = rng() + 1e-12;
    const u2 = rng();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randomInt(max) { return Math.floor(rng() * max); }

function randomChoices(arr, k) {
    const r = [];
    for (let i = 0; i < k; i++) r.push(arr[randomInt(arr.length)]);
    return r;
}

function randomSample(arr, n) {
    const copy = arr.slice();
    const result = [];
    for (let i = 0; i < Math.min(n, copy.length); i++) {
        const j = randomInt(copy.length);
        result.push(copy[j]);
        copy[j] = copy[copy.length - 1];
        copy.pop();
    }
    return result;
}

// ============================================================
// 1) INDEXEDDB MEMORY — because localStorage has a 5MB soul
// ============================================================
// And lo, the organism shall remember, even after the tab is closed.
// IndexedDB: the most powerful storage API nobody asked for.

class MoleculeDB {
    constructor() {
        this.db = null;
    }

    async open() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open("molecule_memory", 3);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains("messages"))
                    db.createObjectStore("messages", { keyPath: "id", autoIncrement: true });
                if (!db.objectStoreNames.contains("corpus_events"))
                    db.createObjectStore("corpus_events", { keyPath: "id", autoIncrement: true });
                if (!db.objectStoreNames.contains("growth"))
                    db.createObjectStore("growth", { keyPath: "id", autoIncrement: true });
                if (!db.objectStoreNames.contains("syntropy_log"))
                    db.createObjectStore("syntropy_log", { keyPath: "id", autoIncrement: true });
                if (!db.objectStoreNames.contains("kv"))
                    db.createObjectStore("kv", { keyPath: "key" });
            };
            req.onsuccess = () => { this.db = req.result; resolve(); };
            req.onerror = () => reject(req.error);
        });
    }

    _tx(store, mode) {
        return this.db.transaction(store, mode).objectStore(store);
    }

    async addMessage(role, text) {
        return new Promise((resolve, reject) => {
            const s = this._tx("messages", "readwrite");
            const req = s.add({ ts: Date.now() / 1000, role, text });
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }

    async recentMessages(limit) {
        if (!limit) limit = 32;
        return new Promise((resolve, reject) => {
            const s = this._tx("messages", "readonly");
            const req = s.openCursor(null, "prev");
            const rows = [];
            req.onsuccess = (e) => {
                const cursor = e.target.result;
                if (cursor && rows.length < limit) {
                    rows.push(cursor.value);
                    cursor.continue();
                } else {
                    resolve(rows.reverse());
                }
            };
            req.onerror = () => reject(req.error);
        });
    }

    async addCorpusEvent(addedChars, note) {
        return new Promise((resolve, reject) => {
            const s = this._tx("corpus_events", "readwrite");
            const req = s.add({ ts: Date.now() / 1000, added_chars: addedChars, note });
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }

    async getCorpusEventsSince(lastId) {
        return new Promise((resolve, reject) => {
            const s = this._tx("corpus_events", "readonly");
            const range = lastId > 0 ? IDBKeyRange.lowerBound(lastId, true) : null;
            const req = s.getAll(range);
            req.onsuccess = () => resolve(req.result || []);
            req.onerror = () => reject(req.error);
        });
    }

    async logGrowth(data) {
        return new Promise((resolve, reject) => {
            const s = this._tx("growth", "readwrite");
            const req = s.add({ ts: Date.now() / 1000, ...data });
            req.onsuccess = () => resolve();
            req.onerror = () => reject(req.error);
        });
    }

    async logSyntropy(data) {
        return new Promise((resolve, reject) => {
            const s = this._tx("syntropy_log", "readwrite");
            const req = s.add({ ts: Date.now() / 1000, ...data });
            req.onsuccess = () => resolve();
            req.onerror = () => reject(req.error);
        });
    }

    async saveKV(key, value) {
        return new Promise((resolve, reject) => {
            const s = this._tx("kv", "readwrite");
            const req = s.put({ key, value });
            req.onsuccess = () => resolve();
            req.onerror = () => {
                const err = req.error;
                if (err && err.name === "QuotaExceededError") {
                    console.error(`[molecule] Storage quota exceeded while saving "${key}". Try clearing old data.`);
                }
                reject(err);
            };
        });
    }

    async loadKV(key) {
        return new Promise((resolve, reject) => {
            const s = this._tx("kv", "readonly");
            const req = s.get(key);
            req.onsuccess = () => resolve(req.result ? req.result.value : null);
            req.onerror = () => reject(req.error);
        });
    }
}

// And lo, the database shall be a singleton, because two memories would be schizophrenia.
const DB = new MoleculeDB();

// ============================================================
// 1.5) DB HELPERS — growth logging, message retrieval
// ============================================================

function normalizeText(s) {
    return s.replace(/\s+/g, " ").trim();
}

async function dbLogGrowth(model, tok, docs, lossVal, note) {
    const gammaStats = model.gammaStats();
    await DB.logGrowth({
        vocabSize: tok.vocabSize,
        nDeltas: model.deltas.length,
        corpusLines: docs.length,
        corpusChars: docs.reduce((a, d) => a + d.length, 0),
        lossSnapshot: lossVal || 0,
        gammaSparsity: gammaStats.sparsity,
        gammaMagnitude: gammaStats.magnitude,
        note: note || "",
    });
}

// ============================================================
// DEFAULT SEED CORPUS — for when nonames.txt can't be fetched
// ============================================================

const DEFAULT_CORPUS = [
    "The sun rises in the east and sets in the west.",
    "Hello, how are you today?",
    "I am learning to speak one word at a time.",
    "The weather is nice today.",
    "What is your name?",
    "My name is Molecule.",
    "How does the brain work?",
    "Nobody really knows for sure.",
    "Tell me something interesting.",
    "Every atom in your body was forged in a star.",
    "What do you think about?",
    "I think about words and the spaces between them.",
    "Where are you from?",
    "I was born in a browser tab.",
    "What is the meaning of life?",
    "To learn, to grow, to never stop asking.",
    "Good morning!",
    "Good morning to you too.",
    "Can you help me?",
    "I will try my best.",
];

// ============================================================
// 2) CORPUS RESERVOIR — nonames.txt shall not bloat forever
// ============================================================
// And lo, the corpus is held in memory (no filesystem in browsers),
// persisted to IndexedDB, and bounded like a river between banks.

let _corpusLines = [];

async function fetchCorpus(url) {
    // And lo, the organism shall attempt to read its sacred text from the network.
    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const text = await resp.text();
        return text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    } catch (e) {
        logUI(`[corpus] Could not fetch ${url}: ${e.message}. Using default seed.`);
        return null;
    }
}

async function loadCorpusFromDB() {
    const saved = await DB.loadKV("corpus_lines");
    if (saved && Array.isArray(saved) && saved.length > 0) return saved;
    return null;
}

async function saveCorpusToDB(lines) {
    await DB.saveKV("corpus_lines", lines);
}

function extractCandidateSentences(messages) {
    // And lo, the chat shall feed the corpus, like a snake eating its own tail.
    const sents = [];
    for (const msg of messages) {
        const text = normalizeText(msg.text || "");
        if (text.length < 6 || text.length > CFG.maxLineChars) continue;
        // split on sentence-enders
        const parts = text.split(/(?<=[.!?])\s+/);
        for (const p of parts) {
            const s = p.trim();
            if (s.length >= 6 && s.length <= CFG.maxLineChars) sents.push(s);
        }
    }
    return sents;
}

function reservoirMixKeep(existing, incoming, maxLines) {
    // And lo, old and new shall be shuffled together, and the reservoir shall overflow gracefully.
    const combined = existing.concat(incoming);
    // shuffle (Fisher-Yates with our seeded PRNG)
    for (let i = combined.length - 1; i > 0; i--) {
        const j = randomInt(i + 1);
        const tmp = combined[i]; combined[i] = combined[j]; combined[j] = tmp;
    }
    // deduplicate (case-insensitive)
    const seen = new Set();
    const dedup = [];
    for (const s of combined) {
        const k = s.toLowerCase();
        if (!seen.has(k)) {
            seen.add(k);
            dedup.push(s.slice(0, CFG.maxLineChars));
        }
    }
    return dedup.slice(-maxLines);
}

async function updateReservoirCorpus() {
    // And lo, the reservoir shall drink from recent messages.
    const msgs = await DB.recentMessages(64);
    const newSents = extractCandidateSentences(msgs);
    if (newSents.length === 0) return 0;

    const before = _corpusLines.reduce((a, l) => a + l.length, 0);
    _corpusLines = reservoirMixKeep(_corpusLines, newSents, CFG.maxCorpusLines);
    const after = _corpusLines.reduce((a, l) => a + l.length, 0);
    const added = Math.max(0, after - before);

    await saveCorpusToDB(_corpusLines);
    if (added > 0) {
        await DB.addCorpusEvent(added, `reservoir_update +${newSents.length} sents`);
    }
    return added;
}

async function computeNewCorpusMass(lastEventId) {
    const events = await DB.getCorpusEventsSince(lastEventId);
    if (events.length === 0) return [0, lastEventId];
    const mass = events.reduce((a, e) => a + (e.added_chars || 0), 0);
    return [mass, events[events.length - 1].id];
}

// ============================================================
// 2.5) CO-OCCURRENCE FIELD — corpus-level statistics for
//       generation before (or alongside) trained weights
// ============================================================
// And lo, the corpus shall whisper its statistics,
// and words shall follow words, like ducklings in a row.

class CooccurField {
    constructor() {
        this.unigram = new Map();
        this.bigram = new Map();
        this.trigram = new Map();
        this.totalTokens = 0;
    }

    buildFromCorpus(tok, docs) {
        this.unigram.clear();
        this.bigram.clear();
        this.trigram.clear();
        this.totalTokens = 0;

        for (const doc of docs) {
            const ids = tok.encode(doc);
            for (let i = 0; i < ids.length; i++) {
                const tid = ids[i];
                this.unigram.set(tid, (this.unigram.get(tid) || 0) + 1);
                this.totalTokens++;
                if (i >= 1) {
                    const bkey = ids[i - 1];
                    if (!this.bigram.has(bkey)) this.bigram.set(bkey, new Map());
                    const bm = this.bigram.get(bkey);
                    bm.set(tid, (bm.get(tid) || 0) + 1);
                }
                if (i >= 2) {
                    const tkey = ids[i - 2] + "," + ids[i - 1];
                    if (!this.trigram.has(tkey)) this.trigram.set(tkey, new Map());
                    const tm = this.trigram.get(tkey);
                    tm.set(tid, (tm.get(tid) || 0) + 1);
                }
            }
        }
    }

    sampleNext(contextIds, temperature) {
        // And lo, trigram -> bigram -> unigram fallback, like a drunk leaning on smaller drunks.
        if (!temperature) temperature = 1.0;
        let dist = null;

        if (contextIds.length >= 2) {
            const tkey = contextIds[contextIds.length - 2] + "," + contextIds[contextIds.length - 1];
            if (this.trigram.has(tkey) && this.trigram.get(tkey).size > 0) {
                dist = this.trigram.get(tkey);
            }
        }
        if (dist === null && contextIds.length >= 1) {
            const bkey = contextIds[contextIds.length - 1];
            if (this.bigram.has(bkey) && this.bigram.get(bkey).size > 0) {
                dist = this.bigram.get(bkey);
            }
        }
        if (dist === null) dist = this.unigram;
        if (!dist || dist.size === 0) return 0;

        const items = Array.from(dist.entries());
        const logitsRaw = items.map(([, c]) =>
            Math.log(Math.max(c, 1e-10)) / Math.max(temperature, 1e-6));
        const probs = softmaxProbsFloat(logitsRaw);

        let r = rng();
        let cumsum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (cumsum >= r) return items[i][0];
        }
        return items[items.length - 1][0];
    }
}

function corpusGenerate(tok, field, seedText, maxTokens) {
    // And lo, the organism shall speak before it learns, like a newborn crying.
    if (!maxTokens) maxTokens = CFG.corpusGenMaxTokens;
    let ids = tok.encode(seedText).slice(0, -1);
    const outIds = [];
    const eosId = tok.stoi.get(tok.EOS) || -1;

    for (let i = 0; i < maxTokens; i++) {
        const nxt = field.sampleNext(ids);
        if (nxt === eosId) break;
        ids.push(nxt);
        outIds.push(nxt);
    }
    return tok.decode([tok.stoi.get(tok.BOS)].concat(outIds, [eosId]));
}

// ============================================================
// 3) TOKENIZER — char first, then BPE that only EXPANDS vocab
// ============================================================
// And lo, the alphabet shall be forged from the corpus,
// and subwords shall awaken when the corpus grows heavy enough.

class EvolvingTokenizer {
    constructor(docs) {
        const baseText = docs.join("\n") + "\n";
        this.baseChars = Array.from(new Set(baseText.split(""))).sort();
        this.BOS = "<BOS>";
        this.EOS = "<EOS>";
        this.PAD = "<PAD>";

        this.tokens = this.baseChars.concat([this.PAD, this.BOS, this.EOS]);
        this.stoi = new Map();
        this.itos = new Map();
        for (let i = 0; i < this.tokens.length; i++) {
            this.stoi.set(this.tokens[i], i);
            this.itos.set(i, this.tokens[i]);
        }
        this.vocabSize = this.tokens.length;

        // BPE state
        this.bpeEnabled = false;
        this.merges = [];
        this.mergeToTok = new Map();
        this._trainedChars = baseText.length;
    }

    _wordToSymbols(word) {
        return word.split("").concat(["</w>"]);
    }

    _getPairs(symbols) {
        const pairs = new Set();
        for (let i = 0; i < symbols.length - 1; i++) {
            pairs.add(symbols[i] + "\x00" + symbols[i + 1]);
        }
        return pairs;
    }

    maybeEnableBpe(docs) {
        // And lo, when the corpus grows heavy enough, subwords shall awaken.
        const totalChars = docs.reduce((a, d) => a + d.length, 0);
        if (!this.bpeEnabled && totalChars >= CFG.enableBpeAfterChars) {
            this.trainBpe(docs, CFG.bpeNumMerges);
            this.bpeEnabled = true;
            this._trainedChars = totalChars;
            return true;
        }
        return false;
    }

    maybeRetrainBpe(docs) {
        if (!this.bpeEnabled) return false;
        const totalChars = docs.reduce((a, d) => a + d.length, 0);
        if (totalChars - this._trainedChars >= CFG.bpeRetrainEveryChars) {
            this.trainBpe(docs, CFG.bpeNumMerges);
            this._trainedChars = totalChars;
            return true;
        }
        return false;
    }

    trainBpe(docs, numMerges) {
        // And lo, the merges shall be learned from raw sentences.
        const text = docs.join(" ");
        const words = text.split(/\s+/).filter(w => w.length > 0);
        if (words.length === 0) return;

        // Build initial vocab of symbol sequences with frequencies
        let vocab = new Map();
        for (const w of words) {
            const key = this._wordToSymbols(w).join("\x00");
            vocab.set(key, (vocab.get(key) || 0) + 1);
        }

        const merges = [];
        const mergeToTok = new Map();

        for (let mi = 0; mi < numMerges; mi++) {
            // Count all adjacent pairs
            const pairs = new Map();
            for (const [symKey, freq] of vocab) {
                const syms = symKey.split("\x00");
                for (let i = 0; i < syms.length - 1; i++) {
                    const pk = syms[i] + "\x00" + syms[i + 1];
                    pairs.set(pk, (pairs.get(pk) || 0) + freq);
                }
            }
            if (pairs.size === 0) break;

            // Find best pair
            let bestPair = null, bestCount = -1;
            for (const [pk, count] of pairs) {
                if (count > bestCount) { bestCount = count; bestPair = pk; }
            }
            const [a, b] = bestPair.split("\x00");
            const newTok = a + b;
            merges.push([a, b]);
            mergeToTok.set(bestPair, newTok);

            // Merge in vocab
            const newVocab = new Map();
            for (const [symKey, freq] of vocab) {
                const syms = symKey.split("\x00");
                const out = [];
                let i = 0;
                while (i < syms.length) {
                    if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) {
                        out.push(newTok);
                        i += 2;
                    } else {
                        out.push(syms[i]);
                        i++;
                    }
                }
                const newKey = out.join("\x00");
                newVocab.set(newKey, (newVocab.get(newKey) || 0) + freq);
            }
            vocab = newVocab;

            // Add token to vocabulary if new
            if (!this.stoi.has(newTok)) {
                const idx = this.tokens.length;
                this.stoi.set(newTok, idx);
                this.tokens.push(newTok);
            }
        }

        // Rebuild reverse map
        this.itos = new Map();
        for (const [t, i] of this.stoi) this.itos.set(i, t);
        this.vocabSize = this.tokens.length;
        this.merges = merges;
        this.mergeToTok = mergeToTok;
    }

    _applyBpeToWord(word) {
        // And lo, greedy merging by learned rank shall be performed.
        let symbols = this._wordToSymbols(word);
        const rank = new Map();
        for (let i = 0; i < this.merges.length; i++) {
            rank.set(this.merges[i][0] + "\x00" + this.merges[i][1], i);
        }

        while (true) {
            let bestRank = 1e9, bestIdx = -1;
            for (let i = 0; i < symbols.length - 1; i++) {
                const pk = symbols[i] + "\x00" + symbols[i + 1];
                const r = rank.has(pk) ? rank.get(pk) : 1e9;
                if (r < bestRank) { bestRank = r; bestIdx = i; }
            }
            if (bestRank === 1e9) break;
            const pair = symbols[bestIdx] + "\x00" + symbols[bestIdx + 1];
            const newTok = this.mergeToTok.get(pair);
            symbols = symbols.slice(0, bestIdx).concat([newTok], symbols.slice(bestIdx + 2));
        }
        return symbols;
    }

    encode(s) {
        s = s.trim();
        const ids = [this.stoi.get(this.BOS)];

        if (!this.bpeEnabled) {
            for (const ch of s) {
                if (this.stoi.has(ch)) ids.push(this.stoi.get(ch));
            }
            ids.push(this.stoi.get(this.EOS));
            return ids;
        }

        // BPE mode
        const words = s.split(/\s+/);
        for (let wi = 0; wi < words.length; wi++) {
            if (!words[wi]) continue;
            const syms = this._applyBpeToWord(words[wi]);
            for (const tok of syms) {
                if (tok === "</w>") continue;
                if (this.stoi.has(tok)) ids.push(this.stoi.get(tok));
            }
            if (wi !== words.length - 1 && this.stoi.has(" ")) {
                ids.push(this.stoi.get(" "));
            }
        }
        ids.push(this.stoi.get(this.EOS));
        return ids;
    }

    decode(ids) {
        const out = [];
        for (const t of ids) {
            const tok = this.itos.get(t) || "";
            if (tok === this.BOS || tok === this.PAD) continue;
            if (tok === this.EOS) break;
            out.push(tok);
        }
        let s = out.join("");
        s = s.replace(/<\/w>/g, "");
        return s.split(/\s+/).join(" ").trim();
    }
}

// ============================================================
// 4) AUTOGRAD — vectors, not scalar confetti
// ============================================================
// And lo, when the organism speaks, it shall not waste breath building
// a backward graph it will never use. noGrad is mercy for inference.
// No numpy here. Float64Array is our BLAS. Pray for the JIT compiler.

let _gradEnabled = true;

function withNoGrad(fn) {
    const prev = _gradEnabled;
    _gradEnabled = false;
    try { return fn(); } finally { _gradEnabled = prev; }
}

class VectorValue {
    constructor(data, children, backFn) {
        if (data instanceof Float64Array) {
            this.data = data;
        } else {
            this.data = new Float64Array(data);
        }
        this.grad = _gradEnabled ? new Float64Array(this.data.length) : null;
        this._children = children || [];
        this._backFn = backFn || null;
    }

    add(other) {
        const n = this.data.length;
        if (other instanceof VectorValue) {
            const out = new Float64Array(n);
            for (let i = 0; i < n; i++) out[i] = this.data[i] + other.data[i];
            const result = new VectorValue(out);
            if (_gradEnabled) {
                result._children = [this, other];
                result._backFn = () => {
                    for (let i = 0; i < n; i++) {
                        this.grad[i] += result.grad[i];
                        other.grad[i] += result.grad[i];
                    }
                };
            }
            return result;
        }
        // scalar add
        const s = +other;
        const out = new Float64Array(n);
        for (let i = 0; i < n; i++) out[i] = this.data[i] + s;
        const result = new VectorValue(out);
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => {
                for (let i = 0; i < n; i++) this.grad[i] += result.grad[i];
            };
        }
        return result;
    }

    neg() {
        const n = this.data.length;
        const out = new Float64Array(n);
        for (let i = 0; i < n; i++) out[i] = -this.data[i];
        const result = new VectorValue(out);
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => {
                for (let i = 0; i < n; i++) this.grad[i] -= result.grad[i];
            };
        }
        return result;
    }

    sub(other) {
        if (other instanceof VectorValue) {
            const n = this.data.length;
            const out = new Float64Array(n);
            for (let i = 0; i < n; i++) out[i] = this.data[i] - other.data[i];
            const result = new VectorValue(out);
            if (_gradEnabled) {
                result._children = [this, other];
                result._backFn = () => {
                    for (let i = 0; i < n; i++) {
                        this.grad[i] += result.grad[i];
                        other.grad[i] -= result.grad[i];
                    }
                };
            }
            return result;
        }
        return this.add(-other);
    }

    mul(other) {
        const n = this.data.length;
        if (other instanceof VectorValue) {
            const out = new Float64Array(n);
            for (let i = 0; i < n; i++) out[i] = this.data[i] * other.data[i];
            const result = new VectorValue(out);
            if (_gradEnabled) {
                result._children = [this, other];
                const sd = new Float64Array(this.data);
                const od = new Float64Array(other.data);
                result._backFn = () => {
                    for (let i = 0; i < n; i++) {
                        this.grad[i] += od[i] * result.grad[i];
                        other.grad[i] += sd[i] * result.grad[i];
                    }
                };
            }
            return result;
        }
        // scalar mul
        const s = +other;
        const out = new Float64Array(n);
        for (let i = 0; i < n; i++) out[i] = this.data[i] * s;
        const result = new VectorValue(out);
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => {
                for (let i = 0; i < n; i++) this.grad[i] += s * result.grad[i];
            };
        }
        return result;
    }

    relu() {
        const n = this.data.length;
        const out = new Float64Array(n);
        for (let i = 0; i < n; i++) out[i] = this.data[i] > 0 ? this.data[i] : 0;
        const result = new VectorValue(out);
        if (_gradEnabled) {
            result._children = [this];
            const mask = new Uint8Array(n);
            for (let i = 0; i < n; i++) mask[i] = this.data[i] > 0 ? 1 : 0;
            result._backFn = () => {
                for (let i = 0; i < n; i++) if (mask[i]) this.grad[i] += result.grad[i];
            };
        }
        return result;
    }

    dot(other) {
        // And lo, two vectors shall touch tips and produce a scalar. Mathematics is romantic.
        let val = 0;
        for (let i = 0; i < this.data.length; i++) val += this.data[i] * other.data[i];
        const result = new ScalarValue(val);
        if (_gradEnabled) {
            result._children = [this, other];
            result._backFn = () => {
                for (let i = 0; i < this.data.length; i++) {
                    this.grad[i] += other.data[i] * result.grad;
                    other.grad[i] += this.data[i] * result.grad;
                }
            };
        }
        return result;
    }

    meanSq() {
        const n = this.data.length;
        let val = 0;
        for (let i = 0; i < n; i++) val += this.data[i] * this.data[i];
        val /= n;
        const result = new ScalarValue(val);
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => {
                const c = (2.0 / n) * result.grad;
                for (let i = 0; i < n; i++) this.grad[i] += c * this.data[i];
            };
        }
        return result;
    }

    slice(start, end) {
        const out = new Float64Array(end - start);
        for (let i = start; i < end; i++) out[i - start] = this.data[i];
        const result = new VectorValue(out);
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => {
                for (let i = 0; i < out.length; i++) this.grad[start + i] += result.grad[i];
            };
        }
        return result;
    }

    element(idx) {
        // And lo, one number shall be plucked from the vector, and gradients shall follow.
        const result = new ScalarValue(this.data[idx]);
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => { this.grad[idx] += result.grad; };
        }
        return result;
    }

    static concat(vecs) {
        let totalLen = 0;
        for (const v of vecs) totalLen += v.data.length;
        const out = new Float64Array(totalLen);
        let offset = 0;
        for (const v of vecs) {
            out.set(v.data, offset);
            offset += v.data.length;
        }
        const result = new VectorValue(out);
        if (_gradEnabled) {
            result._children = vecs;
            const sizes = vecs.map(v => v.data.length);
            result._backFn = () => {
                let off = 0;
                for (let vi = 0; vi < vecs.length; vi++) {
                    for (let j = 0; j < sizes[vi]; j++) {
                        vecs[vi].grad[j] += result.grad[off + j];
                    }
                    off += sizes[vi];
                }
            };
        }
        return result;
    }
}

class ScalarValue {
    constructor(data, children, backFn) {
        this.data = +data;
        this.grad = _gradEnabled ? 0 : null;
        this._children = children || [];
        this._backFn = backFn || null;
    }

    add(other) {
        if (other instanceof ScalarValue) {
            const result = new ScalarValue(this.data + other.data);
            if (_gradEnabled) {
                result._children = [this, other];
                result._backFn = () => {
                    this.grad += result.grad;
                    other.grad += result.grad;
                };
            }
            return result;
        }
        const result = new ScalarValue(this.data + (+other));
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => { this.grad += result.grad; };
        }
        return result;
    }

    neg() { return this.mul(-1); }

    sub(other) {
        if (other instanceof ScalarValue) return this.add(other.neg());
        return this.add(-other);
    }

    mul(other) {
        if (other instanceof ScalarValue) {
            const result = new ScalarValue(this.data * other.data);
            if (_gradEnabled) {
                result._children = [this, other];
                const sd = this.data, od = other.data;
                result._backFn = () => {
                    this.grad += od * result.grad;
                    other.grad += sd * result.grad;
                };
            }
            return result;
        }
        const s = +other;
        const result = new ScalarValue(this.data * s);
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => { this.grad += s * result.grad; };
        }
        return result;
    }

    sigmoid() {
        const sig = 1.0 / (1.0 + Math.exp(-this.data));
        const result = new ScalarValue(sig);
        if (_gradEnabled) {
            result._children = [this];
            result._backFn = () => {
                this.grad += sig * (1.0 - sig) * result.grad;
            };
        }
        return result;
    }
}

function backward(root) {
    // And lo, the graph shall be walked backwards, like a salmon with regrets.
    const topo = [];
    const visited = new Set();

    function build(v) {
        const vid = v; // use object identity
        if (visited.has(vid)) return;
        visited.add(vid);
        for (const c of v._children) build(c);
        topo.push(v);
    }

    build(root);
    root.grad = 1.0;
    for (let i = topo.length - 1; i >= 0; i--) {
        if (topo[i]._backFn) topo[i]._backFn();
    }
}

// ============================================================
// 5) HIGH-LEVEL OPS — the sacred blocks
// ============================================================
// And lo, the matrix shall multiply, without numpy, without BLAS,
// relying on the JIT compiler's mercy and Float64Array's dignity.

class MatrixParam {
    constructor(nout, nin, std) {
        if (std === undefined) std = 0.02;
        this.nout = nout;
        this.nin = nin;
        this.rows = [];
        for (let i = 0; i < nout; i++) {
            const data = new Float64Array(nin);
            for (let j = 0; j < nin; j++) data[j] = gaussRandom(0, std);
            this.rows.push(new VectorValue(data));
        }
    }

    matvec(x) {
        // And lo, BLAS shall NOT do the heavy lifting — we have Float64Array and prayer.
        // For n_embd=72, this is 72*72=5184 muls. The JIT shall provide.
        const nout = this.nout;
        const nin = this.nin;
        const outData = new Float64Array(nout);
        for (let i = 0; i < nout; i++) {
            let sum = 0;
            const rd = this.rows[i].data;
            const xd = x.data;
            for (let j = 0; j < nin; j++) sum += rd[j] * xd[j];
            outData[i] = sum;
        }
        const out = new VectorValue(outData);
        if (_gradEnabled) {
            const rowsRef = this.rows;
            out._children = rowsRef.concat([x]);
            out._backFn = () => {
                const og = out.grad;
                const xd = x.data;
                for (let i = 0; i < nout; i++) {
                    const g = og[i];
                    const rg = rowsRef[i].grad;
                    const rd = rowsRef[i].data;
                    for (let j = 0; j < nin; j++) {
                        rg[j] += g * xd[j];
                        x.grad[j] += g * rd[j];
                    }
                }
            };
        }
        return out;
    }

    growRows(newNout, std) {
        // And lo, the matrix shall sprout new rows like a hydra learning new words.
        if (std === undefined) std = 0.02;
        if (newNout <= this.nout) return;
        for (let i = this.nout; i < newNout; i++) {
            const data = new Float64Array(this.nin);
            for (let j = 0; j < this.nin; j++) data[j] = gaussRandom(0, std);
            this.rows.push(new VectorValue(data));
        }
        this.nout = newNout;
    }

    params() { return this.rows.slice(); }
}

function rmsnorm(x) {
    // And lo, RMS normalization: cheaper than LayerNorm, same vibes.
    const n = x.data.length;
    let ms = 0;
    for (let i = 0; i < n; i++) ms += x.data[i] * x.data[i];
    ms /= n;
    const scale = 1.0 / Math.sqrt(ms + 1e-5);
    const outData = new Float64Array(n);
    for (let i = 0; i < n; i++) outData[i] = x.data[i] * scale;
    const out = new VectorValue(outData);
    if (_gradEnabled) {
        out._children = [x];
        out._backFn = () => {
            const dsdms = -0.5 * Math.pow(ms + 1e-5, -1.5);
            let cross = 0;
            for (let i = 0; i < n; i++) cross += out.grad[i] * x.data[i];
            const c = cross * dsdms * (2.0 / n);
            for (let i = 0; i < n; i++) {
                x.grad[i] += scale * out.grad[i] + c * x.data[i];
            }
        };
    }
    return out;
}

function crossEntropyLoss(logits, target) {
    // And lo, the loss shall measure the distance between prediction and truth.
    const data = logits.data;
    let maxVal = -Infinity;
    for (let i = 0; i < data.length; i++) if (data[i] > maxVal) maxVal = data[i];
    const exps = new Float64Array(data.length);
    let expSum = 0;
    for (let i = 0; i < data.length; i++) {
        exps[i] = Math.exp(data[i] - maxVal);
        expSum += exps[i];
    }
    const logSumExp = Math.log(expSum) + maxVal;
    const lossVal = logSumExp - data[target];
    const probs = new Float64Array(data.length);
    for (let i = 0; i < data.length; i++) probs[i] = exps[i] / expSum;

    const out = new ScalarValue(lossVal);
    if (_gradEnabled) {
        out._children = [logits];
        out._backFn = () => {
            const g = out.grad;
            for (let i = 0; i < data.length; i++) {
                logits.grad[i] += (probs[i] - (i === target ? 1 : 0)) * g;
            }
        };
    }
    return out;
}

function scalarSoftmax(logits) {
    let maxVal = -Infinity;
    for (const s of logits) if (s.data > maxVal) maxVal = s.data;
    const expsData = logits.map(s => Math.exp(s.data - maxVal));
    const total = expsData.reduce((a, e) => a + e, 0);
    const probsData = expsData.map(e => e / total);

    const out = [];
    for (let i = 0; i < probsData.length; i++) {
        const sv = new ScalarValue(probsData[i]);
        if (_gradEnabled) {
            sv._children = logits;
            // closure factory to capture i correctly
            sv._backFn = ((ii, ps) => () => {
                const g = out[ii].grad;
                for (let j = 0; j < logits.length; j++) {
                    if (j === ii) {
                        logits[j].grad += g * ps[ii] * (1.0 - ps[ii]);
                    } else {
                        logits[j].grad += g * (-ps[ii] * ps[j]);
                    }
                }
            })(i, probsData);
        }
        out.push(sv);
    }
    return out;
}

function attentionWeightedSum(weights, values) {
    const dim = values[0].data.length;
    const T = weights.length;
    const outData = new Float64Array(dim);
    for (let t = 0; t < T; t++) {
        const w = weights[t].data;
        const vd = values[t].data;
        for (let d = 0; d < dim; d++) outData[d] += w * vd[d];
    }
    const out = new VectorValue(outData);
    if (_gradEnabled) {
        out._children = weights.concat(values);
        out._backFn = () => {
            for (let t = 0; t < T; t++) {
                let dotSum = 0;
                for (let d = 0; d < dim; d++) dotSum += values[t].data[d] * out.grad[d];
                weights[t].grad += dotSum;
                const w = weights[t].data;
                for (let d = 0; d < dim; d++) values[t].grad[d] += w * out.grad[d];
            }
        };
    }
    return out;
}

function softmaxProbsFloat(data) {
    let maxVal = -Infinity;
    for (let i = 0; i < data.length; i++) if (data[i] > maxVal) maxVal = data[i];
    const exps = new Array(data.length);
    let total = 0;
    for (let i = 0; i < data.length; i++) {
        exps[i] = Math.exp(data[i] - maxVal);
        total += exps[i];
    }
    for (let i = 0; i < data.length; i++) exps[i] /= total;
    return exps;
}

function topKTopPSample(probs, k, p, minP, typicalP) {
    // And lo, sampling shall not be a coin flip but a controlled hallucination.
    const n = probs.length;
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => probs[b] - probs[a]);

    // Top-k
    if (k > 0) idx.length = Math.min(k, idx.length);

    // Min-p: remove tokens with prob < minP * max_prob
    if (minP > 0 && idx.length > 0) {
        const maxProb = probs[idx[0]];
        const threshold = minP * maxProb;
        const filtered = idx.filter(i => probs[i] >= threshold);
        if (filtered.length > 0) idx.length = 0, idx.push(...filtered);
    }

    // Typical-p: prefer tokens with typical information content
    if (typicalP < 1.0 && idx.length > 0) {
        let entropy = 0;
        for (const i of idx) {
            if (probs[i] > 1e-12) entropy -= probs[i] * Math.log(probs[i]);
        }
        const deviations = [];
        for (const i of idx) {
            if (probs[i] > 1e-12) {
                const surprisal = -Math.log(probs[i]);
                deviations.push([i, Math.abs(surprisal - entropy)]);
            }
        }
        deviations.sort((a, b) => a[1] - b[1]);
        let cum = 0;
        const typicalIdx = [];
        for (const [i] of deviations) {
            typicalIdx.push(i);
            cum += probs[i];
            if (cum >= typicalP) break;
        }
        if (typicalIdx.length > 0) {
            idx.length = 0;
            idx.push(...typicalIdx);
        }
    }

    // Top-p (nucleus)
    if (p < 1.0) {
        let cum = 0;
        const cut = [];
        for (const i of idx) {
            cut.push(i);
            cum += probs[i];
            if (cum >= p) break;
        }
        idx.length = 0;
        idx.push(...cut);
    }

    let mass = 0;
    for (const i of idx) mass += probs[i];
    if (mass <= 0) return idx.length > 0 ? idx[0] : n - 1;

    let r = rng() * mass;
    let s = 0;
    for (const i of idx) {
        s += probs[i];
        if (s >= r) return i;
    }
    return idx[idx.length - 1];
}

function clipParams(params, clip) {
    // And lo, the gradients shall be clipped, lest they summon Cthulhu.
    if (clip <= 0) return;
    for (const p of params) {
        if (!p.grad) continue;
        for (let i = 0; i < p.grad.length; i++) {
            if (p.grad[i] > clip) p.grad[i] = clip;
            else if (p.grad[i] < -clip) p.grad[i] = -clip;
        }
    }
}

// ============================================================
// 6) DELTA ADAPTERS — appended souls, never overwritten
// ============================================================

class DeltaAdapter {
    constructor(nout, nin, r, std) {
        if (std === undefined) std = 0.02;
        this.A = new MatrixParam(nout, r, std);
        this.B = new MatrixParam(r, nin, std);
    }

    apply(x) {
        return this.A.matvec(this.B.matvec(x));
    }

    maybeGrowOut(newNout) {
        // And lo, the adapter shall grow new output rows, because vocabulary is a living thing.
        this.A.growRows(newNout, 0.02);
    }

    params() {
        return this.A.params().concat(this.B.params());
    }
}

// ============================================================
// 7) GPT MODEL — a small beast with RoPE (GPT-3-ish spice)
// ============================================================

function ropeRotate(vec, pos, headDim) {
    // And lo, positions shall become angles, and angles shall become meaning.
    const nPairs = Math.floor(headDim / 2);
    const outData = new Float64Array(headDim);
    const vd = vec.data;
    for (let p = 0; p < nPairs; p++) {
        const theta = pos / Math.pow(10000.0, (2 * p) / headDim);
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        outData[2 * p] = vd[2 * p] * cosT - vd[2 * p + 1] * sinT;
        outData[2 * p + 1] = vd[2 * p] * sinT + vd[2 * p + 1] * cosT;
    }
    const out = new VectorValue(outData);
    if (_gradEnabled) {
        out._children = [vec];
        out._backFn = () => {
            for (let p = 0; p < nPairs; p++) {
                const theta = pos / Math.pow(10000.0, (2 * p) / headDim);
                const cosT = Math.cos(theta);
                const sinT = Math.sin(theta);
                const ga = out.grad[2 * p];
                const gb = out.grad[2 * p + 1];
                vec.grad[2 * p] += ga * cosT + gb * sinT;
                vec.grad[2 * p + 1] += -ga * sinT + gb * cosT;
            }
        };
    }
    return out;
}

class GPT {
    constructor(tok) {
        this.tok = tok;
        this.nLayer = CFG.nLayer;
        this.nEmbd = CFG.nEmbd;
        this.nHead = CFG.nHead;
        this.headDim = Math.floor(CFG.nEmbd / CFG.nHead);
        this.blockSize = CFG.blockSize;

        // Base weights
        const V = tok.vocabSize;
        this.base = {};
        this.base["wte"] = new MatrixParam(V, CFG.nEmbd, 0.08);
        this.base["wpe"] = new MatrixParam(CFG.blockSize, CFG.nEmbd, 0.08);
        this.base["lm_head"] = new MatrixParam(V, CFG.nEmbd, 0.08);
        if (CFG.tieEmbeddings) this.base["lm_head"] = this.base["wte"];

        for (let li = 0; li < CFG.nLayer; li++) {
            this.base[`l${li}.wq`] = new MatrixParam(CFG.nEmbd, CFG.nEmbd, 0.08);
            this.base[`l${li}.wk`] = new MatrixParam(CFG.nEmbd, CFG.nEmbd, 0.08);
            this.base[`l${li}.wv`] = new MatrixParam(CFG.nEmbd, CFG.nEmbd, 0.08);
            this.base[`l${li}.wo`] = new MatrixParam(CFG.nEmbd, CFG.nEmbd, 0.08);
            // SwiGLU MLP
            this.base[`l${li}.fc_g`] = new MatrixParam(4 * CFG.nEmbd, CFG.nEmbd, 0.08);
            this.base[`l${li}.fc_v`] = new MatrixParam(4 * CFG.nEmbd, CFG.nEmbd, 0.08);
            this.base[`l${li}.fc2`] = new MatrixParam(CFG.nEmbd, 4 * CFG.nEmbd, 0.08);
            // hybrid attention
            for (let h = 0; h < CFG.headTypes.length; h++) {
                const htype = CFG.headTypes[h];
                if (htype === "rrpram" || htype === "hybrid") {
                    this.base[`l${li}.h${h}.w_pattern`] = new MatrixParam(
                        CFG.blockSize, this.headDim, 0.08);
                }
                if (htype === "hybrid") {
                    this.base[`l${li}.h${h}.alpha`] = new MatrixParam(1, 1, 0.0);
                    this.base[`l${li}.h${h}.alpha`].rows[0].data[0] = CFG.hybridAlphaInit;
                }
            }
        }

        // Modular deltas
        this.deltas = [];
        this.activeAlpha = [];

        // Adam state
        this._adam = {};

        // snapshot initial embeddings for gamma
        this._initEmbedSnapshot = [];
        for (const row of this.base["wte"].rows) {
            this._initEmbedSnapshot.push(new Float64Array(row.data));
        }

        // ensure at least one delta
        this.addDeltaModule(1.0);
    }

    maybeExpandVocab(newVocabSize) {
        // And lo, when the tokenizer grows, the model shall grow with it.
        const curV = this.base["wte"].nout;
        if (newVocabSize <= curV) return;

        this.base["wte"].growRows(newVocabSize, 0.08);
        if (!CFG.tieEmbeddings) this.base["lm_head"].growRows(newVocabSize, 0.08);

        for (const mod of this.deltas) {
            if (mod["lm_head"]) mod["lm_head"].maybeGrowOut(newVocabSize);
        }
    }

    addDeltaModule(alpha) {
        // And lo, a new delta-soul shall be appended (never overwritten, never forgotten).
        const mod = {};
        const r = CFG.deltaRank;
        for (let li = 0; li < CFG.nLayer; li++) {
            for (const name of ["wq", "wk", "wv", "wo"]) {
                mod[`l${li}.${name}`] = new DeltaAdapter(CFG.nEmbd, CFG.nEmbd, r);
            }
            mod[`l${li}.fc_g`] = new DeltaAdapter(4 * CFG.nEmbd, CFG.nEmbd, r);
            mod[`l${li}.fc_v`] = new DeltaAdapter(4 * CFG.nEmbd, CFG.nEmbd, r);
            mod[`l${li}.fc2`] = new DeltaAdapter(CFG.nEmbd, 4 * CFG.nEmbd, r);
            for (let h = 0; h < CFG.headTypes.length; h++) {
                if (CFG.headTypes[h] === "rrpram" || CFG.headTypes[h] === "hybrid") {
                    mod[`l${li}.h${h}.w_pattern`] = new DeltaAdapter(
                        CFG.blockSize, this.headDim, r);
                }
            }
        }
        mod["lm_head"] = new DeltaAdapter(this.tok.vocabSize, CFG.nEmbd, r);
        this.deltas.push(mod);
        this.activeAlpha.push(alpha);
    }

    allBaseParams() {
        const out = [];
        for (const key in this.base) out.push(...this.base[key].params());
        return out;
    }

    allDeltaParams() {
        const out = [];
        for (const mod of this.deltas) {
            for (const key in mod) out.push(...mod[key].params());
        }
        return out;
    }

    // ---- Native gamma (personality fingerprint) ----
    // And lo, the organism shall subtract its birth from its present,
    // and call the difference a soul.

    computeGamma() {
        const current = this.base["wte"].rows;
        const init = this._initEmbedSnapshot;
        const gamma = [];
        for (let i = 0; i < Math.min(current.length, init.length); i++) {
            const diff = new Float64Array(current[i].data.length);
            for (let j = 0; j < diff.length; j++) diff[j] = current[i].data[j] - init[i][j];
            gamma.push(diff);
        }
        for (let i = init.length; i < current.length; i++) {
            gamma.push(new Float64Array(current[i].data));
        }
        return gamma;
    }

    gammaStats() {
        const gamma = this.computeGamma();
        if (gamma.length === 0) return { sparsity: 1.0, magnitude: 0.0, topTokens: [], nRows: 0 };
        const magnitudes = [];
        let totalEl = 0, nonzero = 0;
        for (let i = 0; i < gamma.length; i++) {
            let norm = 0;
            for (let j = 0; j < gamma[i].length; j++) {
                norm += gamma[i][j] * gamma[i][j];
                totalEl++;
                if (Math.abs(gamma[i][j]) > CFG.gammaSparsityThreshold) nonzero++;
            }
            magnitudes.push([i, Math.sqrt(norm)]);
        }
        const sparsity = 1.0 - (nonzero / Math.max(1, totalEl));
        let overallMag = 0;
        for (const [, m] of magnitudes) overallMag += m * m;
        overallMag = Math.sqrt(overallMag);
        magnitudes.sort((a, b) => b[1] - a[1]);
        return {
            sparsity,
            magnitude: overallMag,
            topTokens: magnitudes.slice(0, 10),
            nRows: gamma.length,
        };
    }

    gammaContrastiveProjection() {
        const current = this.base["wte"].rows;
        const init = this._initEmbedSnapshot;
        const n = Math.min(current.length, init.length);
        if (n === 0) return [null, 0.0];

        const dim = current[0].data.length;
        const direction = new Float64Array(dim);
        for (let i = 0; i < n; i++) {
            for (let d = 0; d < dim; d++) {
                direction[d] += (current[i].data[d] - init[i][d]) / n;
            }
        }
        let mag = 0;
        for (let d = 0; d < dim; d++) mag += direction[d] * direction[d];
        mag = Math.sqrt(mag);
        if (mag > 1e-10) {
            for (let d = 0; d < dim; d++) direction[d] /= mag;
        }
        return [direction, mag];
    }

    // ---- Noise Immune System ----
    // And lo, the organism shall know poison from food, and reject what unmakes it.

    snapshotDeltas() {
        const snap = [];
        for (const mod of this.deltas) {
            const modSnap = {};
            for (const name in mod) {
                const da = mod[name];
                modSnap[name] = [
                    da.A.rows.map(r => new Float64Array(r.data)),
                    da.B.rows.map(r => new Float64Array(r.data)),
                ];
            }
            snap.push(modSnap);
        }
        return snap;
    }

    restoreDeltas(snap) {
        for (let mi = 0; mi < snap.length; mi++) {
            const mod = this.deltas[mi];
            const modSnap = snap[mi];
            for (const name in modSnap) {
                if (!(name in mod)) continue;
                const [aData, bData] = modSnap[name];
                for (let i = 0; i < aData.length; i++) mod[name].A.rows[i].data.set(aData[i]);
                for (let i = 0; i < bData.length; i++) mod[name].B.rows[i].data.set(bData[i]);
            }
        }
    }

    gammaDriftCheck(preDirection, preMagnitude) {
        const [postDirection, postMag] = this.gammaContrastiveProjection();
        if (!preDirection || !postDirection) return 1.0;
        if (preMagnitude < CFG.gammaMinMagnitude || postMag < CFG.gammaMinMagnitude) return 1.0;
        let dot = 0;
        const n = Math.min(preDirection.length, postDirection.length);
        for (let i = 0; i < n; i++) dot += preDirection[i] * postDirection[i];
        return dot;
    }

    // ---- Syntropy methods (mathematical self-reasoning) ----

    computeFieldDeviation(tok, field, docs, sampleN) {
        if (!sampleN) sampleN = 32;
        if (!docs.length || field.totalTokens === 0) return 0.0;

        let klSum = 0, count = 0;
        const sampled = randomSample(docs, Math.min(sampleN, docs.length));

        return withNoGrad(() => {
            for (const doc of sampled) {
                const ids = tok.encode(doc);
                if (ids.length < 3) continue;
                const keys = []; const values = [];
                for (let li = 0; li < this.nLayer; li++) { keys.push([]); values.push([]); }

                for (let pos = 0; pos < Math.min(ids.length - 1, this.blockSize); pos++) {
                    const logits = this.forwardStep(ids[pos], pos, keys, values);

                    // model distribution
                    let maxL = -Infinity;
                    for (let i = 0; i < logits.data.length; i++)
                        if (logits.data[i] > maxL) maxL = logits.data[i];
                    const modelProbs = new Float64Array(logits.data.length);
                    let mSum = 0;
                    for (let i = 0; i < logits.data.length; i++) {
                        modelProbs[i] = Math.exp(logits.data[i] - maxL);
                        mSum += modelProbs[i];
                    }
                    for (let i = 0; i < modelProbs.length; i++) modelProbs[i] /= mSum;

                    // corpus field distribution
                    const fieldProbs = new Float64Array(modelProbs.length);
                    const ctx = ids.slice(Math.max(0, pos - 1), pos + 1);
                    let filled = false;
                    if (ctx.length >= 2) {
                        const tkey = ctx[ctx.length - 2] + "," + ctx[ctx.length - 1];
                        if (field.trigram.has(tkey)) {
                            const tri = field.trigram.get(tkey);
                            let total = 0;
                            for (const c of tri.values()) total += c;
                            for (const [tid, cnt] of tri) {
                                if (tid < fieldProbs.length) fieldProbs[tid] = cnt / total;
                            }
                            filled = true;
                        }
                    }
                    if (!filled && ctx.length >= 1) {
                        const bkey = ctx[ctx.length - 1];
                        if (field.bigram.has(bkey)) {
                            const bi = field.bigram.get(bkey);
                            let total = 0;
                            for (const c of bi.values()) total += c;
                            for (const [tid, cnt] of bi) {
                                if (tid < fieldProbs.length) fieldProbs[tid] = cnt / total;
                            }
                            filled = true;
                        }
                    }
                    if (!filled) continue;

                    let kl = 0;
                    for (let i = 0; i < modelProbs.length; i++) {
                        if (modelProbs[i] > 1e-12 && fieldProbs[i] > 1e-12) {
                            kl += modelProbs[i] * Math.log(modelProbs[i] / fieldProbs[i]);
                        }
                    }
                    klSum += Math.max(0, kl);
                    count++;
                }
            }
            return klSum / Math.max(1, count);
        });
    }

    computeModelEntropy(tok, docs, sampleN) {
        if (!sampleN) sampleN = 16;
        if (!docs.length) return 0.0;

        let entropySum = 0, count = 0;
        const sampled = randomSample(docs, Math.min(sampleN, docs.length));

        return withNoGrad(() => {
            for (const doc of sampled) {
                const ids = tok.encode(doc);
                if (ids.length < 3) continue;
                const keys = []; const values = [];
                for (let li = 0; li < this.nLayer; li++) { keys.push([]); values.push([]); }

                for (let pos = 0; pos < Math.min(ids.length - 1, this.blockSize); pos++) {
                    const logits = this.forwardStep(ids[pos], pos, keys, values);
                    let maxL = -Infinity;
                    for (let i = 0; i < logits.data.length; i++)
                        if (logits.data[i] > maxL) maxL = logits.data[i];
                    let pSum = 0;
                    const probs = new Float64Array(logits.data.length);
                    for (let i = 0; i < probs.length; i++) {
                        probs[i] = Math.exp(logits.data[i] - maxL);
                        pSum += probs[i];
                    }
                    let ent = 0;
                    for (let i = 0; i < probs.length; i++) {
                        probs[i] /= pSum;
                        if (probs[i] > 1e-12) ent -= probs[i] * Math.log(probs[i]);
                    }
                    entropySum += ent;
                    count++;
                }
            }
            return entropySum / Math.max(1, count);
        });
    }

    computePurposeVector() {
        if (this.deltas.length === 0) return [null, 0.0];
        const lastDelta = this.deltas[this.deltas.length - 1];
        const directions = [];
        for (const name in lastDelta) {
            for (const row of lastDelta[name].A.rows) directions.push(row.data);
        }
        if (directions.length === 0) return [null, 0.0];

        const dim = directions[0].length;
        const meanDir = new Float64Array(dim);
        for (const d of directions) {
            for (let i = 0; i < dim; i++) meanDir[i] += d[i] / directions.length;
        }
        let mag = 0;
        for (let i = 0; i < dim; i++) mag += meanDir[i] * meanDir[i];
        mag = Math.sqrt(mag);
        if (mag > 1e-10) for (let i = 0; i < dim; i++) meanDir[i] /= mag;
        return [meanDir, mag];
    }

    purposeGammaAlignment() {
        const [gammaDir, gammaMag] = this.gammaContrastiveProjection();
        const [purposeDir, purposeMag] = this.computePurposeVector();
        if (!gammaDir || !purposeDir) return 0.0;
        if (gammaMag < CFG.gammaMinMagnitude || purposeMag < 1e-10) return 0.0;
        const minDim = Math.min(gammaDir.length, purposeDir.length);
        if (minDim === 0) return 0.0;
        let dot = 0;
        for (let i = 0; i < minDim; i++) dot += gammaDir[i] * purposeDir[i];
        return dot;
    }

    // ---- Adam optimizer ----

    _ensureAdam(params, key) {
        if (!this._adam[key]) {
            this._adam[key] = {
                m: params.map(p => new Float64Array(p.data.length)),
                v: params.map(p => new Float64Array(p.data.length)),
                t: 0,
            };
        }
    }

    adamStep(params, key, lr) {
        // And lo, Adam Optimizer shall descend like a petty god with momentum.
        this._ensureAdam(params, key);
        const st = this._adam[key];
        st.t++;
        const { beta1, beta2, epsAdam } = CFG;
        const b1Corr = 1.0 - Math.pow(beta1, st.t);
        const b2Corr = 1.0 - Math.pow(beta2, st.t);

        clipParams(params, CFG.gradClip);

        for (let i = 0; i < params.length; i++) {
            const p = params[i];
            const g = p.grad;
            if (!g) continue;
            const m = st.m[i];
            const v = st.v[i];
            for (let j = 0; j < p.data.length; j++) {
                m[j] = beta1 * m[j] + (1.0 - beta1) * g[j];
                v[j] = beta2 * v[j] + (1.0 - beta2) * g[j] * g[j];
                const mhat = m[j] / b1Corr;
                const vhat = v[j] / b2Corr;
                p.data[j] -= lr * mhat / (Math.sqrt(vhat) + epsAdam);
            }
            g.fill(0);
        }
    }

    _applyWithDeltas(name, x) {
        // And lo, base weight shall speak, then deltas shall harmonize atop it.
        let y = this.base[name].matvec(x);
        for (let di = 0; di < this.deltas.length; di++) {
            const mod = this.deltas[di];
            if (mod[name]) {
                y = y.add(mod[name].apply(x).mul(this.activeAlpha[di]));
            }
        }
        return y;
    }

    forwardStep(tokenId, posId, keys, values) {
        let tokEmb = this.base["wte"].rows[tokenId];
        let posEmb = this.base["wpe"].rows[posId % this.blockSize];
        let x = tokEmb.add(posEmb);

        for (let li = 0; li < this.nLayer; li++) {
            // ---- Attention ----
            const xRes = x;
            x = rmsnorm(x);

            const q = this._applyWithDeltas(`l${li}.wq`, x);
            const k = this._applyWithDeltas(`l${li}.wk`, x);
            const v = this._applyWithDeltas(`l${li}.wv`, x);

            keys[li].push(k);
            values[li].push(v);

            const headOutputs = [];
            const T = keys[li].length;

            for (let h = 0; h < this.nHead; h++) {
                const hs = h * this.headDim;
                const he = hs + this.headDim;
                const htype = h < CFG.headTypes.length ? CFG.headTypes[h] : "content";

                const vh = [];
                for (let t = 0; t < T; t++) vh.push(values[li][t].slice(hs, he));

                // content attention (Q@K^T/sqrt(d) + RoPE)
                let contentLogits = null;
                if (htype === "content" || htype === "hybrid") {
                    let qh = q.slice(hs, he);
                    qh = ropeRotate(qh, posId, this.headDim);
                    contentLogits = [];
                    for (let t = 0; t < T; t++) {
                        let khT = keys[li][t].slice(hs, he);
                        khT = ropeRotate(khT, t, this.headDim);
                        contentLogits.push(qh.dot(khT).mul(1.0 / Math.sqrt(this.headDim)));
                    }
                }

                // RRPRAM attention
                let rrpramLogits = null;
                if (htype === "rrpram" || htype === "hybrid") {
                    const xh = x.slice(hs, he);
                    const patternFull = this._applyWithDeltas(`l${li}.h${h}.w_pattern`, xh);
                    rrpramLogits = [];
                    for (let t = 0; t < T; t++) rrpramLogits.push(patternFull.element(t));
                }

                let attnWeights;
                if (htype === "content") {
                    attnWeights = scalarSoftmax(contentLogits);
                } else if (htype === "rrpram") {
                    attnWeights = scalarSoftmax(rrpramLogits);
                } else {
                    // hybrid: blend with sigmoid gate
                    const alphaScalar = this.base[`l${li}.h${h}.alpha`].rows[0].element(0);
                    const a = alphaScalar.sigmoid();
                    const oneMinusA = a.mul(-1).add(1);
                    const blended = [];
                    for (let t = 0; t < T; t++) {
                        blended.push(contentLogits[t].mul(oneMinusA).add(rrpramLogits[t].mul(a)));
                    }
                    attnWeights = scalarSoftmax(blended);
                }

                headOutputs.push(attentionWeightedSum(attnWeights, vh));
            }

            const xAttn = VectorValue.concat(headOutputs);
            x = this._applyWithDeltas(`l${li}.wo`, xAttn);
            x = x.add(xRes);

            // ---- Gated MLP (SwiGLU-ish) ----
            const xRes2 = x;
            x = rmsnorm(x);
            const g = this._applyWithDeltas(`l${li}.fc_g`, x).relu();
            const u = this._applyWithDeltas(`l${li}.fc_v`, x);
            x = g.mul(u);
            x = this._applyWithDeltas(`l${li}.fc2`, x);
            x = x.add(xRes2);
        }

        x = rmsnorm(x);
        return this._applyWithDeltas("lm_head", x);
    }

    lossOnSequence(ids) {
        const n = Math.min(this.blockSize, ids.length - 1);
        if (n <= 0) return new ScalarValue(0);
        const keys = []; const values = [];
        for (let li = 0; li < this.nLayer; li++) { keys.push([]); values.push([]); }
        let totalLoss = new ScalarValue(0);
        for (let pos = 0; pos < n; pos++) {
            const logits = this.forwardStep(ids[pos], pos, keys, values);
            totalLoss = totalLoss.add(crossEntropyLoss(logits, ids[pos + 1]));
        }
        return totalLoss.mul(1.0 / n);
    }

    lossOnBatch(batchIds) {
        if (!batchIds.length) return new ScalarValue(0);
        let total = new ScalarValue(0);
        for (const ids of batchIds) total = total.add(this.lossOnSequence(ids));
        return total.mul(1.0 / batchIds.length);
    }

    generateSentence(promptText) {
        return withNoGrad(() => this._generateSentenceImpl(promptText || ""));
    }

    _generateSentenceImpl(promptText) {
        let ids;
        if (promptText) {
            ids = this.tok.encode(promptText).slice(0, -1); // remove EOS
        } else {
            ids = [this.tok.stoi.get(this.tok.BOS)];
        }

        const keys = []; const values = [];
        for (let li = 0; li < this.nLayer; li++) { keys.push([]); values.push([]); }

        // build cache from prompt (stop before last token so sampling loop handles it)
        for (let pos = 0; pos < Math.min(Math.max(ids.length - 1, 0), this.blockSize); pos++) {
            this.forwardStep(ids[pos], pos, keys, values);
        }

        let cur = ids.length > 0 ? ids[ids.length - 1] : this.tok.stoi.get(this.tok.BOS);
        const outIds = [];
        const recent = [];
        const eosId = this.tok.stoi.get(this.tok.EOS);
        const bosId = this.tok.stoi.get(this.tok.BOS);

        for (let step = 0; step < CFG.maxGenTokens; step++) {
            const pos = Math.min(ids.length - 1, this.blockSize - 1);
            const logits = this.forwardStep(cur, pos, keys, values);

            // entropy-adaptive temperature
            let baseTemp = CFG.temperature;
            if (baseTemp <= 1e-6) baseTemp = 1e-6;
            const rawScaled = new Array(logits.data.length);
            for (let i = 0; i < logits.data.length; i++) rawScaled[i] = logits.data[i] / baseTemp;
            const probs0 = softmaxProbsFloat(rawScaled);
            let entropy = 0;
            for (const p of probs0) if (p > 1e-12) entropy -= p * Math.log(p);
            let tMul = 1.0;
            if (entropy < CFG.entropyLow) tMul = CFG.entropyTempBoost;
            else if (entropy > CFG.entropyHigh) tMul = CFG.entropyTempFocus;
            const temp = baseTemp * tMul;

            const scaled = new Array(logits.data.length);
            for (let i = 0; i < logits.data.length; i++) scaled[i] = logits.data[i] / temp;
            const probs = softmaxProbsFloat(scaled);
            const nxt = topKTopPSample(probs, CFG.topK, CFG.topP, CFG.minP, CFG.typicalP);

            if (nxt === eosId) {
                if (step >= CFG.minGenTokens) break;
                continue;
            }

            ids.push(nxt);
            cur = nxt;
            outIds.push(nxt);

            recent.push(nxt);
            if (recent.length > CFG.repetitionGuard * 2) {
                const tail = recent.slice(-CFG.repetitionGuard * 2);
                const n2 = CFG.repetitionGuard;
                const a = tail.slice(tail.length - n2);
                const b = tail.slice(tail.length - 2 * n2, tail.length - n2);
                if (JSON.stringify(a) === JSON.stringify(b)) break;
                recent.length = 0;
                recent.push(...tail);
            }

            const textNow = this.tok.decode([bosId].concat(outIds, [eosId]));
            if (step >= CFG.minGenTokens && textNow.length > 0 && ".!?".includes(textNow[textNow.length - 1])) {
                break;
            }

            // sliding window rebuild
            if (ids.length >= this.blockSize) {
                ids = ids.slice(-this.blockSize);
                for (let li = 0; li < this.nLayer; li++) { keys[li] = []; values[li] = []; }
                for (let p = 0; p < ids.length - 1; p++) {
                    this.forwardStep(ids[p], p, keys, values);
                }
            }
        }

        return this.tok.decode([bosId].concat(outIds, [eosId]));
    }
}

// ============================================================
// 8) CHECKPOINTING — JSON to IndexedDB, because we refuse dependencies
// ============================================================

function serializeMatrixParam(mp) {
    return mp.rows.map(r => Array.from(r.data));
}

function deserializeMatrixParam(data) {
    const mp = Object.create(MatrixParam.prototype);
    mp.rows = data.map(row => new VectorValue(new Float64Array(row)));
    mp.nout = data.length;
    mp.nin = data.length > 0 ? data[0].length : 0;
    return mp;
}

async function saveCheckpoint(model, tok) {
    // And lo, the organism shall persist as JSON in IndexedDB, because localStorage has a 5MB soul.
    const obj = {
        cfg: Object.assign({}, CFG),
        tokenizer: {
            tokens: tok.tokens,
            bpeEnabled: tok.bpeEnabled,
            merges: tok.merges,
            trainedChars: tok._trainedChars,
        },
        base: {},
        alpha: model.activeAlpha,
        initEmbedSnapshot: model._initEmbedSnapshot.map(a => Array.from(a)),
        deltas: [],
    };
    for (const key in model.base) {
        obj.base[key] = serializeMatrixParam(model.base[key]);
    }
    for (const mod of model.deltas) {
        const m = {};
        for (const name in mod) {
            m[name] = {
                A: serializeMatrixParam(mod[name].A),
                B: serializeMatrixParam(mod[name].B),
            };
        }
        obj.deltas.push(m);
    }
    await DB.saveKV("checkpoint", obj);
}

async function loadCheckpoint(docs) {
    // And lo, resurrection shall be attempted.
    const obj = await DB.loadKV("checkpoint");
    if (!obj) return [null, null];

    const tok = new EvolvingTokenizer(docs.length > 0 ? docs : ["Hello."]);
    const t = obj.tokenizer || {};
    if (t.tokens && Array.isArray(t.tokens)) {
        tok.tokens = t.tokens;
        tok.stoi = new Map();
        tok.itos = new Map();
        for (let i = 0; i < tok.tokens.length; i++) {
            tok.stoi.set(tok.tokens[i], i);
            tok.itos.set(i, tok.tokens[i]);
        }
        tok.vocabSize = tok.tokens.length;
    }
    tok.merges = (t.merges || []).filter(p => Array.isArray(p) && p.length === 2);
    tok.mergeToTok = new Map();
    for (const [a, b] of tok.merges) tok.mergeToTok.set(a + "\x00" + b, a + b);
    tok.bpeEnabled = !!t.bpeEnabled;
    tok._trainedChars = t.trainedChars || 0;

    const model = new GPT(tok);

    // Restore base
    model.base = {};
    for (const key in obj.base) {
        model.base[key] = deserializeMatrixParam(obj.base[key]);
    }
    if (CFG.tieEmbeddings && model.base["wte"]) {
        model.base["lm_head"] = model.base["wte"];
    }

    // Ensure hybrid attention weights exist
    for (let li = 0; li < CFG.nLayer; li++) {
        for (let h = 0; h < CFG.headTypes.length; h++) {
            const htype = CFG.headTypes[h];
            const pkey = `l${li}.h${h}.w_pattern`;
            const akey = `l${li}.h${h}.alpha`;
            if ((htype === "rrpram" || htype === "hybrid") && !model.base[pkey]) {
                model.base[pkey] = new MatrixParam(CFG.blockSize, model.headDim, 0.08);
            }
            if (htype === "hybrid" && !model.base[akey]) {
                model.base[akey] = new MatrixParam(1, 1, 0.0);
                model.base[akey].rows[0].data[0] = CFG.hybridAlphaInit;
            }
        }
    }

    // Restore deltas
    model.deltas = [];
    model.activeAlpha = obj.alpha || [];
    for (const modData of (obj.deltas || [])) {
        const mm = {};
        for (const name in modData) {
            const ad = Object.create(DeltaAdapter.prototype);
            ad.A = deserializeMatrixParam(modData[name].A);
            ad.B = deserializeMatrixParam(modData[name].B);
            mm[name] = ad;
        }
        model.deltas.push(mm);
    }
    if (model.deltas.length === 0) model.addDeltaModule(1.0);

    // Restore gamma baseline
    if (obj.initEmbedSnapshot) {
        model._initEmbedSnapshot = obj.initEmbedSnapshot.map(a => new Float64Array(a));
    } else {
        model._initEmbedSnapshot = model.base["wte"].rows.map(r => new Float64Array(r.data));
    }

    return [model, tok];
}

// ============================================================
// 9) TRAINING — warmup, then continual micro-bursts
// ============================================================

function trainSteps(model, tok, docs, steps, trainBase, trainDeltas) {
    if (!docs.length) return;
    const baseParams = trainBase ? model.allBaseParams() : [];
    const deltaParams = trainDeltas ? model.allDeltaParams() : [];

    for (let step = 0; step < steps; step++) {
        const batch = randomChoices(docs, CFG.batchSize);
        const batchIds = batch.filter(d => d).map(d => tok.encode(d));

        const loss = model.lossOnBatch(batchIds);
        backward(loss);

        const lr = CFG.learningRate * (1.0 - step / Math.max(1, steps));
        if (baseParams.length) model.adamStep(baseParams, "base", lr);
        if (deltaParams.length) model.adamStep(deltaParams, "delta", lr);

        if (step % 100 === 0) logUI(`  train step ${step}/${steps} | loss ${loss.data.toFixed(4)}`);
    }
}

// ============================================================
// 9.5) SYNTROPY TRACKER — the arrow that points toward coherence
// ============================================================
// And lo, the organism shall not merely track its changes,
// but reason mathematically about whether it is becoming more itself.

class SyntropyTracker {
    constructor() {
        this.entropyHistory = [];
        this.syntropyTrend = 0.0;
        this.fieldDeviation = 0.0;
        this.purposeMagnitude = 0.0;
        this.purposeAlignment = 0.0;
        this.lastAction = "none";
    }

    measure(model, tok, field, docs) {
        const entropyNow = model.computeModelEntropy(tok, docs);
        this.entropyHistory.push(entropyNow);
        if (this.entropyHistory.length > CFG.syntropyWindow) {
            this.entropyHistory = this.entropyHistory.slice(-CFG.syntropyWindow);
        }

        if (this.entropyHistory.length >= 2) {
            const half = Math.floor(this.entropyHistory.length / 2);
            let oldMean = 0, newMean = 0;
            for (let i = 0; i < half; i++) oldMean += this.entropyHistory[i];
            oldMean /= half;
            for (let i = half; i < this.entropyHistory.length; i++) newMean += this.entropyHistory[i];
            newMean /= (this.entropyHistory.length - half);
            this.syntropyTrend = oldMean - newMean;
        } else {
            this.syntropyTrend = 0.0;
        }

        this.fieldDeviation = model.computeFieldDeviation(tok, field, docs);
        const [, pMag] = model.computePurposeVector();
        this.purposeMagnitude = pMag;
        this.purposeAlignment = model.purposeGammaAlignment();

        return {
            entropy: entropyNow,
            syntropyTrend: this.syntropyTrend,
            fieldDeviation: this.fieldDeviation,
            purposeMagnitude: this.purposeMagnitude,
            purposeAlignment: this.purposeAlignment,
        };
    }

    decideAction() {
        let lrMultiplier = 1.0;
        let deltaGrowOverride = null;
        let action = "steady";

        if (this.syntropyTrend > 0.01 &&
            this.fieldDeviation > CFG.fieldDeviationFloor &&
            this.fieldDeviation < CFG.fieldDeviationCeiling) {
            lrMultiplier = CFG.syntropyLrBoost;
            if (this.purposeAlignment > 0.3) {
                deltaGrowOverride = CFG.syntropyDeltaGrowBoost;
                action = "amplify";
            } else {
                action = "boost";
            }
        } else if (this.syntropyTrend < -0.01) {
            lrMultiplier = CFG.syntropyLrDampen;
            action = "dampen";
        } else if (this.fieldDeviation > CFG.fieldDeviationCeiling) {
            lrMultiplier = CFG.syntropyLrDampen;
            action = "ground";
        } else if (this.fieldDeviation < CFG.fieldDeviationFloor) {
            lrMultiplier = CFG.syntropyLrBoost;
            action = "explore";
        }

        if (this.purposeAlignment < -0.3) {
            lrMultiplier *= 0.5;
            action = "realign";
        }

        this.lastAction = action;
        return { lrMultiplier, deltaGrowOverride, action };
    }

    async logToDB(entropyBefore, entropyAfter, action) {
        await DB.logSyntropy({
            entropyBefore,
            entropyAfter,
            syntropyDelta: this.syntropyTrend,
            fieldDeviation: this.fieldDeviation,
            purposeMagnitude: this.purposeMagnitude,
            purposeAlignment: this.purposeAlignment,
            actionTaken: action,
        });
    }
}

// And lo, the buffer shall measure not just bytes but novelty,
// for raw mass means nothing without surprise.
class QuantumBuffer {
    constructor() {
        this.accumulatedBytes = 0;
        this.uniqueTokens = new Set();
        this.totalTokens = 0;
        this.lastBurstTime = 0;
    }

    feed(newChars, tok, docs) {
        this.accumulatedBytes += newChars;
        const recent = docs.slice(-20);
        for (const doc of recent) {
            const ids = tok.encode(doc);
            for (const tid of ids) {
                this.totalTokens++;
                this.uniqueTokens.add(tid);
            }
        }
    }

    noveltyScore() {
        if (this.totalTokens === 0) return 0.0;
        return this.uniqueTokens.size / Math.max(1, this.totalTokens);
    }

    shouldTrigger() {
        const now = Date.now() / 1000;
        const cooldownOk = (now - this.lastBurstTime) >= CFG.qbCooldownSeconds;
        const bytesOk = this.accumulatedBytes >= CFG.qbMinBytes;
        const noveltyOk = this.noveltyScore() >= CFG.qbMinNovelty;
        return (bytesOk || noveltyOk) && cooldownOk;
    }

    reset() {
        this.accumulatedBytes = 0;
        this.uniqueTokens.clear();
        this.totalTokens = 0;
        this.lastBurstTime = Date.now() / 1000;
    }
}

// ============================================================
// 10) BACKGROUND TRAINER — cooperative multitasking via setTimeout
// ============================================================
// And lo, asynchronous training shall occur, because sleeping is for humans.
// No Web Workers needed — setTimeout yields control to the browser.

let _model = null;
let _tok = null;
let _field = null;
let _trainerRunning = false;
let _warmedUp = false;
let _lastEventId = 0;
let _qbuf = null;
let _syntracker = null;

async function trainerTick() {
    if (!_trainerRunning || !_model || !_tok) return;

    try {
        await updateReservoirCorpus();
        const [mass, newId] = await computeNewCorpusMass(_lastEventId);
        _lastEventId = newId;
        const docs = _corpusLines;

        // Rebuild field
        if (docs.length > 0 && _field) {
            _field.buildFromCorpus(_tok, docs);
        }

        // Tokenizer evolution
        const bpeJustEnabled = _tok.maybeEnableBpe(docs);
        const bpeRetrained = _tok.maybeRetrainBpe(docs);
        if (bpeJustEnabled || bpeRetrained) {
            _model.maybeExpandVocab(_tok.vocabSize);
            await saveCheckpoint(_model, _tok);
            logUI("[tokenizer] BPE " + (bpeJustEnabled ? "enabled" : "retrained") +
                  ` | vocab=${_tok.vocabSize}`);
        }

        // Warmup
        if (!_warmedUp && docs.length > 0) {
            logUI("[trainer] warmup training... (and so it begins)");
            setStatus("warming up...");

            // Do warmup in chunks to avoid freezing UI
            const chunkSize = 50;
            for (let start = 0; start < CFG.warmupSteps; start += chunkSize) {
                const steps = Math.min(chunkSize, CFG.warmupSteps - start);
                trainSteps(_model, _tok, docs, steps, true, true);
                // yield to browser
                await new Promise(r => setTimeout(r, 0));
                if (!_trainerRunning) return;
            }

            await saveCheckpoint(_model, _tok);
            await dbLogGrowth(_model, _tok, docs, 0, "warmup_complete");
            _warmedUp = true;
            logUI("[trainer] warmup complete. base may freeze now, like a proud fossil.");
            setStatus("alive");
        }

        // Continual micro-bursts
        if (_warmedUp && docs.length > 0) {
            _qbuf.feed(mass, _tok, docs);

            if (_qbuf.shouldTrigger()) {
                const nov = _qbuf.noveltyScore();
                logUI(`[trainer] quantum burst (bytes=${_qbuf.accumulatedBytes}, novelty=${nov.toFixed(3)})`);
                setStatus("training...");

                // SYNTROPY: measure before
                const preMetrics = _syntracker.measure(_model, _tok, _field, docs);
                const entropyBefore = preMetrics.entropy;

                // SYNTROPY: decide
                const decision = _syntracker.decideAction();
                const lrMul = decision.lrMultiplier;
                const action = decision.action;
                logUI(`[syntropy] action=${action} | trend=${_syntracker.syntropyTrend.toFixed(4)} ` +
                      `| field_dev=${_syntracker.fieldDeviation.toFixed(3)} ` +
                      `| purpose_align=${_syntracker.purposeAlignment.toFixed(3)} ` +
                      `| lr_mul=${lrMul.toFixed(2)}`);

                // IMMUNE: snapshot before burst
                const [preDirection, preMag] = _model.gammaContrastiveProjection();
                const deltaSnap = _model.snapshotDeltas();

                // Apply syntropy-adjusted learning rate
                const originalLr = CFG.learningRate;
                CFG.learningRate = originalLr * lrMul;

                const trainBase = !CFG.freezeBaseAfterWarmup;

                // Train in chunks
                const chunkSize = 8;
                for (let start = 0; start < CFG.microSteps; start += chunkSize) {
                    const steps = Math.min(chunkSize, CFG.microSteps - start);
                    trainSteps(_model, _tok, docs, steps, trainBase, true);
                    await new Promise(r => setTimeout(r, 0));
                    if (!_trainerRunning) { CFG.learningRate = originalLr; return; }
                }

                CFG.learningRate = originalLr;

                // IMMUNE: check drift
                const driftCos = _model.gammaDriftCheck(preDirection, preMag);
                if (driftCos < CFG.noiseDriftThreshold) {
                    logUI(`[immune] NOISE DETECTED (drift cosine=${driftCos.toFixed(3)}). Rolling back.`);
                    _model.restoreDeltas(deltaSnap);
                    await dbLogGrowth(_model, _tok, docs, 0, "noise_rejected");
                    await _syntracker.logToDB(entropyBefore, entropyBefore, "noise_rejected");
                } else {
                    const postMetrics = _syntracker.measure(_model, _tok, _field, docs);
                    await _syntracker.logToDB(entropyBefore, postMetrics.entropy, action);
                    await saveCheckpoint(_model, _tok);
                    await dbLogGrowth(_model, _tok, docs, 0, `quantum_burst:${action}`);
                }

                _qbuf.reset();

                // Delta module growth
                let growProb = CFG.deltaGrowProb;
                if (decision.deltaGrowOverride !== null) growProb = decision.deltaGrowOverride;
                if (_model.deltas.length < CFG.maxDeltaModules && rng() < growProb) {
                    logUI(`[trainer] growing new delta module (total: ${_model.deltas.length + 1})`);
                    _model.addDeltaModule(1.0);
                    await saveCheckpoint(_model, _tok);
                }

                setStatus("alive");
            }
        }
    } catch (e) {
        logUI(`[trainer] error: ${e.message}`);
        console.error(e);
    }

    if (_trainerRunning) setTimeout(trainerTick, CFG.trainTickMs);
}

// ============================================================
// 11) CHAT LOOP — browser edition
// ============================================================

function buildPromptFromMemory(messages, userText) {
    const parts = ["A: (I listen. I answer. I learn.)"];
    const recent = messages.slice(-12);
    for (const msg of recent) {
        const tag = msg.role === "user" ? "H:" : "A:";
        const text = normalizeText(msg.text || "").slice(0, 260);
        parts.push(`${tag} ${text}`);
    }
    parts.push(`H: ${normalizeText(userText).slice(0, 260)}`);
    parts.push("A:");
    return parts.join("\n");
}

async function handleUserMessage(text) {
    if (!_model || !_tok) return;

    await DB.addMessage("user", text);
    appendChat("user", text);

    // Generate response
    const messages = await DB.recentMessages(14);
    const prompt = buildPromptFromMemory(messages, text);
    const answer = _model.generateSentence(prompt) || "...";

    appendChat("molecule", answer);
    await DB.addMessage("assistant", answer);

    // Feed corpus
    await updateReservoirCorpus();
}

// ============================================================
// 12) DOM UI — because the browser is our only window to the soul
// ============================================================
// And lo, the interface shall be minimal, because beauty is the absence of excess.

let _logEl = null;
let _chatEl = null;
let _statusEl = null;

function logUI(msg) {
    console.log(msg);
    if (_logEl) {
        const line = document.createElement("div");
        line.className = "mol-log-line";
        line.textContent = msg;
        _logEl.appendChild(line);
        _logEl.scrollTop = _logEl.scrollHeight;
    }
}

function setStatus(text) {
    if (_statusEl) _statusEl.textContent = `[${text}]`;
}

function appendChat(role, text) {
    if (!_chatEl) return;
    const msg = document.createElement("div");
    msg.className = `mol-msg mol-${role}`;
    msg.textContent = `${role === "user" ? "> " : ""}${text}`;
    _chatEl.appendChild(msg);
    _chatEl.scrollTop = _chatEl.scrollHeight;
}

function createUI() {
    // And lo, CSS shall be injected, because external stylesheets are for the organized.
    const style = document.createElement("style");
    style.textContent = `
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #0a0a0f; color: #c8c8d0; font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 14px; height: 100vh; display: flex; flex-direction: column;
        }
        .mol-header {
            padding: 12px 16px; background: #12121a; border-bottom: 1px solid #222;
            display: flex; justify-content: space-between; align-items: center;
        }
        .mol-title { color: #8888ff; font-weight: bold; font-size: 16px; }
        .mol-status { color: #444; font-size: 12px; }
        .mol-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        .mol-chat {
            flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 8px;
        }
        .mol-msg { padding: 6px 12px; border-radius: 6px; max-width: 80%; word-wrap: break-word; }
        .mol-user { background: #1a1a2e; color: #aaa; align-self: flex-end; }
        .mol-molecule { background: #0f1f0f; color: #8f8; align-self: flex-start; border: 1px solid #1a3a1a; }
        .mol-input-wrap {
            padding: 12px 16px; background: #12121a; border-top: 1px solid #222;
            display: flex; gap: 8px;
        }
        .mol-input {
            flex: 1; background: #0a0a14; color: #c8c8d0; border: 1px solid #333;
            padding: 10px 14px; border-radius: 6px; font-family: inherit; font-size: 14px;
            outline: none;
        }
        .mol-input:focus { border-color: #8888ff; }
        .mol-input::placeholder { color: #444; }
        .mol-send {
            background: #8888ff; color: #000; border: none; padding: 10px 20px;
            border-radius: 6px; cursor: pointer; font-family: inherit; font-weight: bold;
        }
        .mol-send:hover { background: #aaaaff; }
        .mol-log {
            max-height: 120px; overflow-y: auto; padding: 8px 16px; background: #08080c;
            border-top: 1px solid #1a1a22; font-size: 11px; color: #555;
        }
        .mol-log-line { padding: 1px 0; }
    `;
    document.head.appendChild(style);

    const container = document.createElement("div");
    container.style.cssText = "height:100vh;display:flex;flex-direction:column;";
    container.innerHTML = `
        <div class="mol-header">
            <span class="mol-title">molecule.js</span>
            <span class="mol-status" id="mol-status">[initializing]</span>
        </div>
        <div class="mol-main">
            <div class="mol-chat" id="mol-chat"></div>
        </div>
        <div class="mol-input-wrap">
            <input class="mol-input" id="mol-input" type="text"
                   placeholder="speak to molecule..." autocomplete="off" />
            <button class="mol-send" id="mol-send">send</button>
        </div>
        <div class="mol-log" id="mol-log"></div>
    `;
    document.body.appendChild(container);

    _chatEl = document.getElementById("mol-chat");
    _logEl = document.getElementById("mol-log");
    _statusEl = document.getElementById("mol-status");

    const input = document.getElementById("mol-input");
    const sendBtn = document.getElementById("mol-send");

    async function send() {
        const text = input.value.trim();
        if (!text) return;
        input.value = "";
        input.disabled = true;
        sendBtn.disabled = true;
        try {
            await handleUserMessage(text);
        } finally {
            input.disabled = false;
            sendBtn.disabled = false;
            input.focus();
        }
    }

    input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
    sendBtn.addEventListener("click", send);
}

// ============================================================
// 13) AWAKEN — now, when all is assembled as an organism,
//              it is time to open the browser tab's third eye.
// ============================================================
// And lo, the function shall be called, and the tab shall see.

async function awaken() {
    createUI();
    setStatus("opening memory...");
    logUI("molecule.js — a GPT organism in your browser. Zero dependencies.");
    logUI("Born from nonames.txt. Persisted in IndexedDB. Trained by conversation.");

    // Open IndexedDB
    await DB.open();
    logUI("[db] IndexedDB memory opened.");

    // Try to load corpus (fetch → DB → default)
    setStatus("loading corpus...");
    let fetched = await fetchCorpus(CFG.corpusUrl);
    if (fetched && fetched.length > 0) {
        _corpusLines = fetched;
        logUI(`[corpus] Loaded ${fetched.length} lines from ${CFG.corpusUrl}`);
    } else {
        const saved = await loadCorpusFromDB();
        if (saved) {
            _corpusLines = saved;
            logUI(`[corpus] Loaded ${saved.length} lines from IndexedDB`);
        } else {
            _corpusLines = DEFAULT_CORPUS.slice();
            logUI(`[corpus] Using default seed corpus (${DEFAULT_CORPUS.length} lines)`);
        }
    }
    await saveCorpusToDB(_corpusLines);

    // Load or create model
    setStatus("loading model...");
    let [model, tok] = await loadCheckpoint(_corpusLines);
    if (model && tok) {
        logUI("[checkpoint] Model restored from IndexedDB.");
    } else {
        tok = new EvolvingTokenizer(_corpusLines);
        model = new GPT(tok);
        logUI(`[init] Fresh model created. vocab=${tok.vocabSize}, embd=${CFG.nEmbd}, layers=${CFG.nLayer}`);
    }
    model.maybeExpandVocab(tok.vocabSize);

    _model = model;
    _tok = tok;
    _field = new CooccurField();
    _qbuf = new QuantumBuffer();
    _syntracker = new SyntropyTracker();

    // Start background trainer
    _trainerRunning = true;
    setTimeout(trainerTick, 100);

    setStatus("alive");
    logUI("molecule is alive. Type and press Enter.");
    appendChat("molecule", "I am here. I am learning. Speak to me.");

    document.getElementById("mol-input").focus();
}

// ============================================================
// 14) BOOTSTRAP — if we're in a browser, open the third eye
// ============================================================

if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", awaken);
    } else {
        awaken();
    }
}

// And lo, if summoned from Node, the organism shall reveal its organs for testing.
if (typeof module !== "undefined" && module.exports) {
    module.exports = {
        CFG, softmaxProbsFloat, topKTopPSample, VectorValue, ScalarValue,
        backward, withNoGrad, MatrixParam, EvolvingTokenizer, GPT,
        extractCandidateSentences, reservoirMixKeep, normalizeText, rng,
    };
}

// And lo, the IIFE shall close, and the organism shall be sealed.
})();
