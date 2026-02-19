/**
 * molequla.js
 * A single-file, zero-dependency, browser-native, continually-learning GPT organism.
 *
 * No npm. No webpack. No React. No node_modules black hole.
 * Just a <script> tag and the blind faith of a mass-less neuron.
 *
 * Port of molequla.py — same architecture, same madness, new habitat.
 *
 * - Trains on nonames.txt (fetched or pasted)
 * - Keeps IndexedDB memory (because localStorage has a 5MB soul)
 * - Maintains a bounded corpus reservoir (never bloats)
 * - Starts in byte-level mode (256 byte tokens + specials)
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
    nLayer: 1,
    nEmbd: 16,
    nHead: 1,
    blockSize: 96,

    // ontogenesis — growth stages [corpus_chars, n_embd, n_layer, n_head]
    growthStages: [
        [0,      16, 1, 1],    // embryo: ~25K params
        [20000,  32, 1, 2],    // infant: ~100K params
        [50000,  64, 2, 4],    // child: ~500K params
        [200000, 128, 4, 4],   // adolescent: ~2M params
        [500000, 256, 6, 8],   // adult: ~10M params
    ],
    freezeAfterGrowthSteps: 200,

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
    enableBpeAfterChars: 20000,
    bpeNumMerges: 384,
    bpeRetrainEveryChars: 4000,

    // async
    trainTickMs: 250,

    // hybrid attention
    headTypes: ["content"],
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
    corpusFadeK: 3.0,            // sigmoid steepness for corpus→model transition
    corpusFadeThreshold: 1.5,    // entropy at which blend is 50/50

    // syntropy
    syntropyWindow: 20,
    fieldDeviationFloor: 0.5,
    fieldDeviationCeiling: 5.0,
    syntropyLrBoost: 1.3,
    syntropyLrDampen: 0.6,
    syntropyDeltaGrowBoost: 0.2,

    // cosine LR schedule
    lrMin: 0.001,
    maxTotalSteps: 50000,
    cosineWarmupSteps: 200,

    // gradient accumulation
    accumSteps: 1,

    // quantum buffer
    qbCooldownSeconds: 10.0,
    qbMinBytes: 480,
    qbMinNovelty: 0.15,
};

function headTypesForNHead(n) {
    // Compute head type array for a given number of heads.
    if (n <= 1) return ["content"];
    if (n === 2) return ["content", "hybrid"];
    const half = Math.floor(n / 2);
    const result = [];
    for (let i = 0; i < half; i++) result.push("content");
    for (let i = 0; i < n - half; i++) result.push("hybrid");
    return result;
}

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

class MolequlaDB {
    constructor() {
        this.db = null;
    }

    async open() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open("molequla_memory", 3);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains("messages")) {
                    const store = db.createObjectStore("messages", { keyPath: "id", autoIncrement: true });
                    store.createIndex("ts", "ts");
                }
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
                    console.error(`[molequla] Storage quota exceeded while saving "${key}". Try clearing old data.`);
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
const DB = new MolequlaDB();

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
    "My name is Molequla.",
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
                    const key1 = ids[i - 2];
                    if (!this.trigram.has(key1)) this.trigram.set(key1, new Map());
                    const m2 = this.trigram.get(key1);
                    const key2 = ids[i - 1];
                    if (!m2.has(key2)) m2.set(key2, new Map());
                    const tm = m2.get(key2);
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
            const k1 = contextIds[contextIds.length - 2];
            const k2 = contextIds[contextIds.length - 1];
            const m2 = this.trigram.get(k1);
            if (m2) {
                const tm = m2.get(k2);
                if (tm && tm.size > 0) dist = tm;
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
        if (docs === undefined) docs = [];
        this.BOS = "<BOS>";
        this.EOS = "<EOS>";
        this.PAD = "<PAD>";

        // 256 byte tokens (hex strings "0x00" through "0xff") + 3 special tokens = 259
        this.tokens = [];
        for (let i = 0; i < 256; i++) {
            this.tokens.push("0x" + i.toString(16).padStart(2, "0"));
        }
        this.tokens.push(this.BOS, this.EOS, this.PAD);

        this.stoi = new Map();
        this.itos = new Map();
        for (let i = 0; i < this.tokens.length; i++) {
            this.stoi.set(this.tokens[i], i);
            this.itos.set(i, this.tokens[i]);
        }
        this.vocabSize = this.tokens.length; // 259

        // BPE state
        this.bpeEnabled = false;
        this.merges = [];
        this.mergeToTok = new Map();
        this._trainedChars = docs.reduce((s, d) => s + d.length, 0);

        // Reusable TextEncoder/TextDecoder
        this._encoder = new TextEncoder();
        this._decoder = new TextDecoder("utf-8", { fatal: false });
    }

    _unicodeSegment(text) {
        // Pre-segmentation by Unicode category. BPE merges happen WITHIN segments only.
        // Categories: letters (+marks), digits, whitespace, punctuation/symbols.
        const segments = [];
        let current = [];
        let currentCat = null;
        for (const ch of text) {
            let cat;
            if (/\p{L}|\p{M}/u.test(ch)) cat = "L";
            else if (/\p{N}/u.test(ch)) cat = "N";
            else if (/\s/.test(ch)) cat = "Z";
            else cat = "P";
            if (cat !== currentCat && current.length > 0) {
                segments.push(this._encoder.encode(current.join("")));
                current = [];
            }
            currentCat = cat;
            current.push(ch);
        }
        if (current.length > 0) segments.push(this._encoder.encode(current.join("")));
        return segments;
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
        // And lo, the merges shall be learned from byte sequences within Unicode segments.
        const text = docs.join(" ");
        if (!text) return;

        // Segment and convert to byte-token sequences, count frequencies
        const segments = this._unicodeSegment(text);
        let vocab = new Map();
        for (const seg of segments) {
            const tokSeq = [];
            for (let i = 0; i < seg.length; i++) {
                tokSeq.push(this.tokens[seg[i]]); // e.g. "0x48", "0x65", "0x6c"
            }
            const key = tokSeq.join("\x00");
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
            const newTok = a + "+" + b; // e.g. "0x48+0x65"
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

    _applyBPE(tokens) {
        // And lo, greedy merging by learned rank shall be performed.
        if (this.merges.length === 0) return tokens;
        const rank = new Map();
        for (let i = 0; i < this.merges.length; i++) {
            rank.set(this.merges[i][0] + "\x00" + this.merges[i][1], i);
        }

        let symbols = tokens.slice();
        while (symbols.length >= 2) {
            let bestRank = Infinity, bestIdx = -1;
            for (let i = 0; i < symbols.length - 1; i++) {
                const key = symbols[i] + "\x00" + symbols[i + 1];
                const r = rank.get(key);
                if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
            }
            if (bestIdx === -1) break;
            const pairKey = symbols[bestIdx] + "\x00" + symbols[bestIdx + 1];
            const merged = this.mergeToTok.get(pairKey) || (symbols[bestIdx] + "+" + symbols[bestIdx + 1]);
            symbols = symbols.slice(0, bestIdx).concat([merged], symbols.slice(bestIdx + 2));
        }
        return symbols;
    }

    encode(s) {
        // Encode text to token IDs: text -> segments -> bytes -> BPE -> IDs
        s = s.trim();
        const ids = [this.stoi.get(this.BOS)];

        if (!s) {
            ids.push(this.stoi.get(this.EOS));
            return ids;
        }

        const segments = this._unicodeSegment(s);
        for (const seg of segments) {
            // Convert bytes to base token names
            const baseTokens = [];
            for (let i = 0; i < seg.length; i++) {
                baseTokens.push(this.tokens[seg[i]]);
            }
            const merged = this.bpeEnabled ? this._applyBPE(baseTokens) : baseTokens;
            for (const tok of merged) {
                if (this.stoi.has(tok)) ids.push(this.stoi.get(tok));
            }
        }

        ids.push(this.stoi.get(this.EOS));
        return ids;
    }

    _tokenToBytes(tok) {
        // Convert a token string back to bytes.
        if (tok.startsWith("0x") && tok.indexOf("+") === -1 && tok.length === 4) {
            // Single byte token: "0x41" -> [0x41]
            return [parseInt(tok, 16)];
        } else if (tok.indexOf("+") !== -1) {
            // Merged token: "0x48+0x65" -> split by "+", each "0xNN" -> byte
            const parts = tok.split("+");
            const result = [];
            for (let i = 0; i < parts.length; i++) {
                result.push(parseInt(parts[i], 16));
            }
            return result;
        }
        return [];
    }

    decode(ids) {
        // Decode token IDs back to text: IDs -> bytes -> UTF-8
        const rawBytes = [];
        for (const t of ids) {
            const tok = this.itos.get(t) || "";
            if (tok === this.BOS || tok === this.PAD) continue;
            if (tok === this.EOS) break;
            const bytes = this._tokenToBytes(tok);
            for (let i = 0; i < bytes.length; i++) rawBytes.push(bytes[i]);
        }
        return this._decoder.decode(new Uint8Array(rawBytes)).trim();
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
                result._backFn = () => {
                    for (let i = 0; i < n; i++) {
                        this.grad[i] += other.data[i] * result.grad[i];
                        other.grad[i] += this.data[i] * result.grad[i];
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

    silu() {
        // SiLU (Sigmoid Linear Unit): x * sigmoid(x)
        // The real activation for SwiGLU, not the relu impostor.
        const n = this.data.length;
        const out = new Float64Array(n);
        const sig = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            sig[i] = 1.0 / (1.0 + Math.exp(-this.data[i]));
            out[i] = this.data[i] * sig[i];
        }
        const result = new VectorValue(out);
        if (_gradEnabled) {
            result._children = [this];
            const xSnap = new Float64Array(this.data);
            const sigSnap = new Float64Array(sig);
            result._backFn = () => {
                // d_silu/dx = sig * (1 + x * (1 - sig))
                for (let i = 0; i < n; i++) {
                    const dsilu = sigSnap[i] * (1.0 + xSnap[i] * (1.0 - sigSnap[i]));
                    this.grad[i] += dsilu * result.grad[i];
                }
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
    // Cleanup: release the computation graph to avoid memory leaks
    for (const v of topo) {
        v._children = [];
        v._backFn = null;
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

    growCols(newNin, std) {
        // And lo, the matrix shall widen its reach, each row stretching into new dimensions.
        // Float64Array is NOT resizable — must create new array + copy.
        if (std === undefined) std = 0.02;
        if (newNin <= this.nin) return;
        for (let i = 0; i < this.rows.length; i++) {
            const row = this.rows[i];
            const newData = new Float64Array(newNin);
            newData.set(row.data);
            for (let j = this.nin; j < newNin; j++) newData[j] = gaussRandom(0, std);
            row.data = newData;
            if (row.grad) {
                const newGrad = new Float64Array(newNin);
                newGrad.set(row.grad);
                row.grad = newGrad;
            }
        }
        this.nin = newNin;
    }

    grow(newNout, newNin, std) {
        // Ontogenesis: grow both dimensions. Cols first so new rows get full width.
        if (std === undefined) std = 0.02;
        this.growCols(newNin, std);
        this.growRows(newNout, std);
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

    growDims(newNout, newNin) {
        // Ontogenesis: grow both outer dimensions of the adapter. Rank stays the same.
        this.A.growRows(newNout);    // A: (nout, r) -> extend output
        this.B.growCols(newNin);     // B: (r, nin) -> extend input
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

        // Residual scaling: deeper networks need gentler residual branches
        this.residualAlpha = 1.0 / Math.sqrt(Math.max(1, this.nLayer));
        this.globalStep = 0;

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

        // syntropy temperature offset (adjusted by syntropy tracker decisions)
        this.syntropyTempOffset = 0;

        // ontogenesis: freeze base after growth
        this._growthFreezeRemaining = 0;

        // adaptive corpus blend: set by trainerTick
        this._corpusField = null;

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

    // ---- Ontogenesis (architecture growth) ----
    // And lo, the organism shall not be born adult but shall grow, stage by stage,
    // from embryo to child to adolescent, each growth a small death and rebirth.

    currentGrowthStage() {
        for (let i = 0; i < CFG.growthStages.length; i++) {
            const [, embd, layer, head] = CFG.growthStages[i];
            if (this.nEmbd === embd && this.nLayer === layer && this.nHead === head) return i;
        }
        return -1; // dimensions don't match any stage (legacy checkpoint)
    }

    targetGrowthStage(corpusChars) {
        let target = 0;
        for (let i = 0; i < CFG.growthStages.length; i++) {
            if (corpusChars >= CFG.growthStages[i][0]) target = i;
        }
        return target;
    }

    maybeGrowArchitecture(corpusChars) {
        const current = this.currentGrowthStage();
        if (current < 0) return false; // legacy checkpoint, skip growth
        const target = this.targetGrowthStage(corpusChars);
        if (target <= current) return false;

        const [, newEmbd, newLayer, newHead] = CFG.growthStages[target];
        const oldEmbd = this.nEmbd;
        const oldLayer = this.nLayer;
        const oldHead = this.nHead;
        const newHeadDim = Math.floor(newEmbd / newHead);

        logUI(`[growth] ONTOGENESIS: stage ${current} -> ${target}`);
        logUI(`  embd: ${oldEmbd} -> ${newEmbd}, layer: ${oldLayer} -> ${newLayer}, head: ${oldHead} -> ${newHead}`);

        // 1. Grow embedding matrices (columns only — vocab rows stay)
        this.base["wte"].growCols(newEmbd);
        this.base["wpe"].growCols(newEmbd);
        if (!CFG.tieEmbeddings) {
            this.base["lm_head"].growCols(newEmbd);
        }

        // 2. Grow existing layer matrices
        const newHtypes = headTypesForNHead(newHead);
        for (let li = 0; li < oldLayer; li++) {
            for (const name of ["wq", "wk", "wv", "wo"]) {
                this.base[`l${li}.${name}`].grow(newEmbd, newEmbd);
            }
            this.base[`l${li}.fc_g`].grow(4 * newEmbd, newEmbd);
            this.base[`l${li}.fc_v`].grow(4 * newEmbd, newEmbd);
            this.base[`l${li}.fc2`].grow(newEmbd, 4 * newEmbd);
            // Grow existing head pattern matrices
            for (let h = 0; h < oldHead; h++) {
                const pkey = `l${li}.h${h}.w_pattern`;
                if (this.base[pkey]) this.base[pkey].growCols(newHeadDim);
            }
            // Add new heads for existing layer
            for (let h = oldHead; h < newHead; h++) {
                const htype = h < newHtypes.length ? newHtypes[h] : "content";
                if (htype === "rrpram" || htype === "hybrid") {
                    this.base[`l${li}.h${h}.w_pattern`] = new MatrixParam(
                        CFG.blockSize, newHeadDim, 0.08);
                }
                if (htype === "hybrid") {
                    this.base[`l${li}.h${h}.alpha`] = new MatrixParam(1, 1, 0.0);
                    this.base[`l${li}.h${h}.alpha`].rows[0].data[0] = CFG.hybridAlphaInit;
                }
            }
        }

        // 3. Add entirely new layers
        for (let li = oldLayer; li < newLayer; li++) {
            this.base[`l${li}.wq`] = new MatrixParam(newEmbd, newEmbd, 0.08);
            this.base[`l${li}.wk`] = new MatrixParam(newEmbd, newEmbd, 0.08);
            this.base[`l${li}.wv`] = new MatrixParam(newEmbd, newEmbd, 0.08);
            this.base[`l${li}.wo`] = new MatrixParam(newEmbd, newEmbd, 0.08);
            this.base[`l${li}.fc_g`] = new MatrixParam(4 * newEmbd, newEmbd, 0.08);
            this.base[`l${li}.fc_v`] = new MatrixParam(4 * newEmbd, newEmbd, 0.08);
            this.base[`l${li}.fc2`] = new MatrixParam(newEmbd, 4 * newEmbd, 0.08);
            for (let h = 0; h < newHead; h++) {
                const htype = h < newHtypes.length ? newHtypes[h] : "content";
                if (htype === "rrpram" || htype === "hybrid") {
                    this.base[`l${li}.h${h}.w_pattern`] = new MatrixParam(
                        CFG.blockSize, newHeadDim, 0.08);
                }
                if (htype === "hybrid") {
                    this.base[`l${li}.h${h}.alpha`] = new MatrixParam(1, 1, 0.0);
                    this.base[`l${li}.h${h}.alpha`].rows[0].data[0] = CFG.hybridAlphaInit;
                }
            }
        }

        // 4. Grow delta adapters
        const r = CFG.deltaRank;
        for (const mod of this.deltas) {
            // Grow existing layer adapters
            for (let li = 0; li < oldLayer; li++) {
                for (const name of ["wq", "wk", "wv", "wo"]) {
                    const key = `l${li}.${name}`;
                    if (mod[key]) mod[key].growDims(newEmbd, newEmbd);
                }
                for (const [key, noutM, ninM] of [
                    [`l${li}.fc_g`, 4, 1],
                    [`l${li}.fc_v`, 4, 1],
                    [`l${li}.fc2`, 1, 4],
                ]) {
                    if (mod[key]) mod[key].growDims(noutM * newEmbd, ninM * newEmbd);
                }
                for (let h = 0; h < oldHead; h++) {
                    const pkey = `l${li}.h${h}.w_pattern`;
                    if (mod[pkey]) mod[pkey].growDims(CFG.blockSize, newHeadDim);
                }
                for (let h = oldHead; h < newHead; h++) {
                    const htype = h < newHtypes.length ? newHtypes[h] : "content";
                    if (htype === "rrpram" || htype === "hybrid") {
                        mod[`l${li}.h${h}.w_pattern`] = new DeltaAdapter(
                            CFG.blockSize, newHeadDim, r);
                    }
                }
            }
            // New layers: entirely new adapters
            for (let li = oldLayer; li < newLayer; li++) {
                for (const name of ["wq", "wk", "wv", "wo"]) {
                    mod[`l${li}.${name}`] = new DeltaAdapter(newEmbd, newEmbd, r);
                }
                mod[`l${li}.fc_g`] = new DeltaAdapter(4 * newEmbd, newEmbd, r);
                mod[`l${li}.fc_v`] = new DeltaAdapter(4 * newEmbd, newEmbd, r);
                mod[`l${li}.fc2`] = new DeltaAdapter(newEmbd, 4 * newEmbd, r);
                for (let h = 0; h < newHead; h++) {
                    const htype = h < newHtypes.length ? newHtypes[h] : "content";
                    if (htype === "rrpram" || htype === "hybrid") {
                        mod[`l${li}.h${h}.w_pattern`] = new DeltaAdapter(
                            CFG.blockSize, newHeadDim, r);
                    }
                }
            }
            // lm_head adapter input grew
            if (mod["lm_head"]) {
                mod["lm_head"].growDims(this.tok.vocabSize, newEmbd);
            }
        }

        // 5. Update model state
        this.nEmbd = newEmbd;
        this.nLayer = newLayer;
        this.nHead = newHead;
        this.headDim = newHeadDim;
        this.residualAlpha = 1.0 / Math.sqrt(Math.max(1, newLayer));

        // 6. Update CFG runtime
        CFG.nEmbd = newEmbd;
        CFG.nLayer = newLayer;
        CFG.nHead = newHead;
        CFG.headTypes = headTypesForNHead(newHead);

        // 7. Reset Adam state (old momentum is meaningless after arch change)
        this._adam = {};

        // 8. Extend gamma snapshot for new embedding dimensions
        for (let i = 0; i < this._initEmbedSnapshot.length; i++) {
            const old = this._initEmbedSnapshot[i];
            if (old.length < newEmbd) {
                const ext = new Float64Array(newEmbd);
                ext.set(old);
                this._initEmbedSnapshot[i] = ext;
            }
        }

        // 9. Set freeze (only train deltas until new weights stabilize)
        this._growthFreezeRemaining = CFG.freezeAfterGrowthSteps;

        logUI(`[growth] Done. Freeze for ${CFG.freezeAfterGrowthSteps} steps.`);
        return true;
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
                        const tk1 = ctx[ctx.length - 2];
                        const tk2 = ctx[ctx.length - 1];
                        const m2 = field.trigram.get(tk1);
                        if (m2) {
                            const tri = m2.get(tk2);
                            if (tri) {
                                let total = 0;
                                for (const c of tri.values()) total += c;
                                for (const [tid, cnt] of tri) {
                                    if (tid < fieldProbs.length) fieldProbs[tid] = cnt / total;
                                }
                                filled = true;
                            }
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
            x = xRes.add(x.mul(this.residualAlpha));

            // ---- Gated MLP (SwiGLU) ----
            const xRes2 = x;
            x = rmsnorm(x);
            const g = this._applyWithDeltas(`l${li}.fc_g`, x).silu();
            const u = this._applyWithDeltas(`l${li}.fc_v`, x);
            x = g.mul(u);
            x = this._applyWithDeltas(`l${li}.fc2`, x);
            x = xRes2.add(x.mul(this.residualAlpha));
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

    quickLoss(tok, docs, n) {
        if (!n) n = 4;
        if (!docs.length) return 0.0;
        return withNoGrad(() => {
            const sampled = randomSample(docs, Math.min(n, docs.length));
            const batchIds = sampled.filter(d => d).map(d => tok.encode(d));
            if (!batchIds.length) return 0.0;
            let total = 0;
            for (const ids of batchIds) {
                total += this.lossOnSequence(ids).data;
            }
            return total / batchIds.length;
        });
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
        let scaled = null; // pre-allocated across steps
        const bosId = this.tok.stoi.get(this.tok.BOS);

        for (let step = 0; step < CFG.maxGenTokens; step++) {
            const pos = Math.min(ids.length - 1, this.blockSize - 1);
            const logits = this.forwardStep(cur, pos, keys, values);

            // entropy-adaptive temperature (with syntropy offset)
            let baseTemp = CFG.temperature + this.syntropyTempOffset;
            if (baseTemp <= 1e-6) baseTemp = 1e-6;
            const vocabLen = logits.data.length;
            if (!scaled || scaled.length !== vocabLen) scaled = new Array(vocabLen);
            for (let i = 0; i < vocabLen; i++) scaled[i] = logits.data[i] / baseTemp;
            let probs = softmaxProbsFloat(scaled);
            let entropy = 0;
            for (const p of probs) if (p > 1e-12) entropy -= p * Math.log(p);
            let tMul = 1.0;
            if (entropy < CFG.entropyLow) tMul = CFG.entropyTempBoost;
            else if (entropy > CFG.entropyHigh) tMul = CFG.entropyTempFocus;
            if (tMul !== 1.0) {
                const temp = baseTemp * tMul;
                for (let i = 0; i < vocabLen; i++) scaled[i] = logits.data[i] / temp;
                probs = softmaxProbsFloat(scaled);
            }

            // Adaptive corpus blend: corpus field fades as model becomes coherent
            if (this._corpusField && this._corpusField.bigram.size > 0) {
                const modelAlpha = 1.0 / (1.0 + Math.exp(-CFG.corpusFadeK * (CFG.corpusFadeThreshold - entropy)));
                if (modelAlpha < 0.99) {
                    let corpusDist = null;
                    // Try trigram first
                    if (ids.length >= 2) {
                        const tk1 = ids[ids.length - 2];
                        const tk2 = ids[ids.length - 1];
                        const m2 = this._corpusField.trigram.get(tk1);
                        if (m2) {
                            const tm = m2.get(tk2);
                            if (tm) corpusDist = tm;
                        }
                    }
                    // Fallback to bigram
                    if (!corpusDist && ids.length >= 1) {
                        const bkey = ids[ids.length - 1];
                        if (this._corpusField.bigram.has(bkey)) {
                            corpusDist = this._corpusField.bigram.get(bkey);
                        }
                    }
                    if (corpusDist) {
                        let totalC = 0;
                        for (const cnt of corpusDist.values()) totalC += cnt;
                        const corpusProbs = new Array(probs.length).fill(0);
                        for (const [tid, cnt] of corpusDist.entries()) {
                            if (tid < corpusProbs.length) corpusProbs[tid] = cnt / totalC;
                        }
                        let totalB = 0;
                        for (let i = 0; i < probs.length; i++) {
                            probs[i] = modelAlpha * probs[i] + (1.0 - modelAlpha) * corpusProbs[i];
                            totalB += probs[i];
                        }
                        if (totalB > 0) {
                            for (let i = 0; i < probs.length; i++) probs[i] /= totalB;
                        }
                    }
                }
            }

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
        globalStep: model.globalStep,
        initEmbedSnapshot: model._initEmbedSnapshot.map(a => Array.from(a)),
        deltas: [],
        // ontogenesis state
        modelDims: {
            nEmbd: model.nEmbd,
            nLayer: model.nLayer,
            nHead: model.nHead,
            growthFreezeRemaining: model._growthFreezeRemaining,
        },
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
    for (const [a, b] of tok.merges) tok.mergeToTok.set(a + "\x00" + b, a + "+" + b);
    tok.bpeEnabled = !!t.bpeEnabled;
    tok._trainedChars = t.trainedChars || 0;

    // Restore model dimensions from checkpoint (ontogenesis state)
    if (obj.modelDims) {
        CFG.nEmbd = obj.modelDims.nEmbd || CFG.nEmbd;
        CFG.nLayer = obj.modelDims.nLayer || CFG.nLayer;
        CFG.nHead = obj.modelDims.nHead || CFG.nHead;
        CFG.headTypes = headTypesForNHead(CFG.nHead);
    } else if (obj.cfg) {
        // Fallback: restore from saved CFG for pre-ontogenesis checkpoints
        if (obj.cfg.nEmbd) CFG.nEmbd = obj.cfg.nEmbd;
        if (obj.cfg.nLayer) CFG.nLayer = obj.cfg.nLayer;
        if (obj.cfg.nHead) CFG.nHead = obj.cfg.nHead;
        if (obj.cfg.headTypes) CFG.headTypes = obj.cfg.headTypes;
    }

    const model = new GPT(tok);

    // Restore globalStep for cosine LR continuity
    model.globalStep = obj.globalStep || 0;

    // Restore ontogenesis freeze
    if (obj.modelDims && obj.modelDims.growthFreezeRemaining > 0) {
        model._growthFreezeRemaining = obj.modelDims.growthFreezeRemaining;
    }

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

function cosineLR(globalStep) {
    const lrMax = CFG.learningRate;
    const lrMin = CFG.lrMin;
    if (globalStep < CFG.cosineWarmupSteps) {
        // Linear warmup
        return lrMin + (lrMax - lrMin) * (globalStep / Math.max(1, CFG.cosineWarmupSteps));
    }
    const progress = (globalStep - CFG.cosineWarmupSteps) /
        Math.max(1, CFG.maxTotalSteps - CFG.cosineWarmupSteps);
    return lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * Math.min(progress, 1.0)));
}

function trainSteps(model, tok, docs, steps, trainBase, trainDeltas) {
    if (!docs.length) return;

    // Ontogenesis freeze: after growth, only train deltas until new weights stabilize
    let baseParams, deltaParams;
    if (model._growthFreezeRemaining > 0) {
        baseParams = [];
        deltaParams = trainDeltas ? model.allDeltaParams() : [];
        model._growthFreezeRemaining = Math.max(0, model._growthFreezeRemaining - steps);
    } else {
        baseParams = trainBase ? model.allBaseParams() : [];
        deltaParams = trainDeltas ? model.allDeltaParams() : [];
    }

    for (let step = 0; step < steps; step++) {
        // --- Gradient accumulation loop ---
        let accumLoss = null;
        for (let acc = 0; acc < CFG.accumSteps; acc++) {
            const batch = randomChoices(docs, CFG.batchSize);
            const batchIds = batch.filter(d => d).map(d => tok.encode(d));

            const loss = model.lossOnBatch(batchIds).mul(1.0 / CFG.accumSteps);
            backward(loss);

            if (accumLoss === null) accumLoss = loss.data;
            else accumLoss += loss.data;
        }

        const lr = cosineLR(model.globalStep);
        if (baseParams.length) model.adamStep(baseParams, "base", lr);
        if (deltaParams.length) model.adamStep(deltaParams, "delta", lr);
        model.globalStep++;

        if (step % 100 === 0) logUI(`  train step ${step}/${steps} | loss ${(accumLoss || 0).toFixed(4)} | lr ${lr.toFixed(6)}`);
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
        this.burstHistory = [];
        this.modelStage = 0;          // current growth stage (set during measure)
        this._lastMitosisTime = 0.0;  // cooldown for divide
        this._swarmInfo = null;        // peer state from swarm (set externally)
    }

    recordBurst(action, lossBefore, lossAfter) {
        this.burstHistory.push({ action, lossBefore, lossAfter });
        if (this.burstHistory.length > 16) {
            this.burstHistory = this.burstHistory.slice(-16);
        }
    }

    actionEffectiveness(action) {
        const matching = this.burstHistory.filter(b => b.action === action);
        if (matching.length === 0) return { mean: 0.0, count: 0 };
        let sum = 0;
        for (const b of matching) sum += (b.lossBefore - b.lossAfter);
        return { mean: sum / matching.length, count: matching.length };
    }

    measure(model, tok, field, docs) {
        this.modelStage = model.currentGrowthStage();
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
        let tempOffset = 0.0;
        let accumOverride = 0;

        if (this.syntropyTrend > 0.01 &&
            this.fieldDeviation > CFG.fieldDeviationFloor &&
            this.fieldDeviation < CFG.fieldDeviationCeiling) {
            lrMultiplier = CFG.syntropyLrBoost;
            if (this.purposeAlignment > 0.3) {
                deltaGrowOverride = CFG.syntropyDeltaGrowBoost;
                action = "amplify";
                tempOffset = -0.05;
                accumOverride = 2;
            } else {
                action = "boost";
            }
        } else if (this.syntropyTrend < -0.01) {
            lrMultiplier = CFG.syntropyLrDampen;
            action = "dampen";
            tempOffset = +0.05;
        } else if (this.fieldDeviation > CFG.fieldDeviationCeiling) {
            lrMultiplier = CFG.syntropyLrDampen;
            action = "ground";
            tempOffset = -0.05;
        } else if (this.fieldDeviation < CFG.fieldDeviationFloor) {
            lrMultiplier = CFG.syntropyLrBoost;
            action = "explore";
            tempOffset = +0.05;
        }

        if (this.purposeAlignment < -0.3) {
            lrMultiplier *= 0.5;
            action = "realign";
            tempOffset = 0.0;
        }

        // CASE 6: Adult + sustained overload -> divide (mitosis)
        const maxStage = CFG.growthStages.length - 1;
        if (this.modelStage >= maxStage &&
                this._isSustainedOverload() &&
                (Date.now() / 1000) - this._lastMitosisTime > 300) {
            action = "divide";
            lrMultiplier = CFG.syntropyLrDampen; // slow down while preparing to split
        }

        // CASE 7: Plateau + young peer thriving -> hibernate (cooperative scheduling)
        if (action === "steady" && this._shouldHibernate()) {
            action = "hibernate";
        }

        // SELF-META-LEARNING: downgrade actions that historically hurt loss
        if (action !== "divide" && action !== "hibernate" && this.burstHistory.length >= 4) {
            const eff = this.actionEffectiveness(action);
            if (eff.count > 0 && eff.mean > 0.05) {
                if (action === "amplify") {
                    action = "boost";
                    tempOffset = 0.0;
                    accumOverride = 0;
                    deltaGrowOverride = null;
                } else if (action === "boost" || action === "explore") {
                    action = "steady";
                    tempOffset = 0.0;
                    accumOverride = 0;
                }
            }
        }

        this.lastAction = action;
        return { lrMultiplier, deltaGrowOverride, action, tempOffset, accumOverride };
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

    _isSustainedOverload() {
        // High entropy for >75% of window + falling syntropy = overloaded.
        if (this.entropyHistory.length < CFG.syntropyWindow) return false;
        const recent = this.entropyHistory.slice(-CFG.syntropyWindow);
        let highCount = 0;
        for (const e of recent) {
            if (e > CFG.entropyHigh) highCount++;
        }
        return highCount > CFG.syntropyWindow * 0.75 && this.syntropyTrend < -0.02;
    }

    _shouldHibernate() {
        // Should this organism sleep to give resources to peers?
        // Conditions: loss on plateau + a peer is in amplify/boost state.
        if (!this._swarmInfo || !this._swarmInfo.peers || this._swarmInfo.peers.length === 0) {
            return false;
        }
        for (const peer of this._swarmInfo.peers) {
            if ((peer.syntropy || 0) > 0.05) {
                // A young peer is thriving. If we're stale, hibernate.
                if (this.burstHistory.length >= 8) {
                    const recentDeltas = this.burstHistory.slice(-8).map(
                        b => b.lossAfter - b.lossBefore);
                    const avgDelta = recentDeltas.reduce((a, v) => a + v, 0) / recentDeltas.length;
                    if (Math.abs(avgDelta) < 0.01) return true; // loss plateau
                }
            }
        }
        return false;
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
// 9.7) SWARM ECOLOGY — the organism learns it is not alone
// ============================================================
// And lo, the first cell shall call into the void and hear only silence.
// But the second shall call and hear an answer.
// In the browser, BroadcastChannel is our mesh — tabs are organisms.

class SwarmRegistry {
    constructor(organismId) {
        this.organismId = organismId || `org_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
        this.channel = null;
        this.peers = new Map(); // id -> {stage, nParams, syntropy, entropy, lastSeen, status}
        this._onMessage = null;
    }

    register() {
        if (typeof BroadcastChannel === "undefined") return; // fallback: no swarm
        this.channel = new BroadcastChannel("molequla_swarm");
        this._onMessage = (event) => {
            const msg = event.data;
            if (!msg || !msg.id || msg.id === this.organismId) return;
            if (msg.type === "heartbeat") {
                this.peers.set(msg.id, {
                    stage: msg.stage || 0,
                    nParams: msg.nParams || 0,
                    syntropy: msg.syntropy || 0,
                    entropy: msg.entropy || 0,
                    lastSeen: Date.now(),
                    status: msg.status || "alive",
                });
            } else if (msg.type === "register") {
                this.peers.set(msg.id, {
                    stage: 0, nParams: 0, syntropy: 0, entropy: 0,
                    lastSeen: Date.now(), status: "alive",
                });
            } else if (msg.type === "dead") {
                this.peers.delete(msg.id);
            } else if (msg.type === "sleeping") {
                if (this.peers.has(msg.id)) {
                    this.peers.get(msg.id).status = "sleeping";
                }
            }
        };
        this.channel.addEventListener("message", this._onMessage);
        // Announce ourselves
        this.channel.postMessage({ type: "register", id: this.organismId });
    }

    heartbeat(stage, nParams, syntropy, entropy) {
        if (!this.channel) return;
        this.channel.postMessage({
            type: "heartbeat", id: this.organismId,
            stage, nParams, syntropy, entropy, status: "alive",
        });
    }

    discoverPeers(timeoutMs) {
        if (!timeoutMs) timeoutMs = 60000;
        const cutoff = Date.now() - timeoutMs;
        const alive = [];
        for (const [id, info] of this.peers) {
            if (info.lastSeen > cutoff && info.status === "alive") {
                alive.push({ id, ...info });
            }
        }
        return alive;
    }

    markHibernating() {
        if (!this.channel) return;
        this.channel.postMessage({ type: "sleeping", id: this.organismId });
    }

    unregister() {
        if (!this.channel) return;
        this.channel.postMessage({ type: "dead", id: this.organismId });
        if (this._onMessage) {
            this.channel.removeEventListener("message", this._onMessage);
        }
        this.channel.close();
        this.channel = null;
    }
}

async function idbPut(storeName, key, value) {
    // Helper: write to IndexedDB kv store with a namespaced key
    await DB.saveKV(`${storeName}:${key}`, value);
}

async function idbGet(storeName, key) {
    return await DB.loadKV(`${storeName}:${key}`);
}

async function performMitosis(model, tok, swarm, syntracker) {
    // And lo, the organism divides. In the browser, mitosis = opening a new tab.
    const childId = `org_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;

    // Save birth config to IndexedDB
    const birth = {
        organism_id: childId,
        parent_id: swarm.organismId,
        burst_history: syntracker.burstHistory,
    };
    await idbPut("births", childId, birth);

    // Open new tab — it will read birth config on startup
    window.open(`${location.href.split("?")[0]}?organism=${childId}`, "_blank");

    syntracker._lastMitosisTime = Date.now() / 1000;
    logUI(`[ecology] Child ${childId} spawned (new tab)`);
    return childId;
}

function performHibernation(model, tok, swarm) {
    // And lo, the organism sleeps. In the browser: stop training, save state.
    // The tab stays open but stops consuming CPU.
    logUI(`[ecology] HIBERNATION — organism ${swarm.organismId} going to sleep`);
    swarm.markHibernating();
    _trainerRunning = false;
    setStatus("hibernating");
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
let _swarm = null;
let _tickCount = 0;

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
            _model._corpusField = _field; // share with generateSentence for adaptive blend
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

                // Phase 1.5: quickLoss before burst
                const lossBefore = _model.quickLoss(_tok, docs, 4);

                // SYNTROPY: decide
                const decision = _syntracker.decideAction();
                const lrMul = decision.lrMultiplier;
                const action = decision.action;
                logUI(`[syntropy] action=${action} | trend=${_syntracker.syntropyTrend.toFixed(4)} ` +
                      `| field_dev=${_syntracker.fieldDeviation.toFixed(3)} ` +
                      `| purpose_align=${_syntracker.purposeAlignment.toFixed(3)} ` +
                      `| lr_mul=${lrMul.toFixed(2)}`);

                // Phase 1.5: apply accumOverride and syntropyTempOffset
                const originalAccumSteps = CFG.accumSteps;
                if (decision.accumOverride > 0) {
                    CFG.accumSteps = decision.accumOverride;
                }
                _model.syntropyTempOffset = decision.tempOffset;

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
                    if (!_trainerRunning) {
                        CFG.learningRate = originalLr;
                        CFG.accumSteps = originalAccumSteps;
                        return;
                    }
                }

                CFG.learningRate = originalLr;
                CFG.accumSteps = originalAccumSteps;

                // Phase 1.5: quickLoss after burst
                const lossAfter = _model.quickLoss(_tok, docs, 4);
                const deltaLoss = lossBefore - lossAfter;
                _syntracker.recordBurst(action, lossBefore, lossAfter);

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
                    await dbLogGrowth(_model, _tok, docs, 0,
                        `quantum_burst:${action}|Δloss=${deltaLoss.toFixed(4)}`);
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

                // Ontogenesis: check if architecture should grow
                const corpusChars = docs.reduce((a, d) => a + d.length, 0);
                if (_model.maybeGrowArchitecture(corpusChars)) {
                    await saveCheckpoint(_model, _tok);
                    await dbLogGrowth(_model, _tok, docs, 0,
                        `ontogenesis:stage=${_model.currentGrowthStage()}`);
                }

                // Ecology: mitosis / hibernation
                if (_swarm && action === "divide") {
                    logUI("[ecology] MITOSIS triggered — organism overloaded, spawning child");
                    await performMitosis(_model, _tok, _swarm, _syntracker);
                }
                if (_swarm && action === "hibernate") {
                    performHibernation(_model, _tok, _swarm);
                    await saveCheckpoint(_model, _tok);
                    return; // exit training loop
                }

                setStatus("alive");
            }
        }

        // Swarm heartbeat (every 10 ticks)
        _tickCount++;
        if (_swarm && _tickCount % 10 === 0) {
            const stage = _model.currentGrowthStage();
            const nP = _model.allBaseParams().reduce((a, p) => a + p.data.length, 0)
                     + _model.allDeltaParams().reduce((a, p) => a + p.data.length, 0);
            _syntracker._swarmInfo = { peers: _swarm.discoverPeers() };
            _swarm.heartbeat(stage, nP, _syntracker.syntropyTrend,
                _syntracker.entropyHistory.length > 0
                    ? _syntracker.entropyHistory[_syntracker.entropyHistory.length - 1]
                    : 0.0);
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

    appendChat("molequla", answer);
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

let _scrollPending = false;
function logUI(msg) {
    console.log(msg);
    if (_logEl) {
        const line = document.createElement("div");
        line.className = "mol-log-line";
        line.textContent = msg;
        _logEl.appendChild(line);
        while (_logEl.children.length > 200) {
            _logEl.removeChild(_logEl.children[0]);
        }
        if (!_scrollPending) {
            _scrollPending = true;
            requestAnimationFrame(() => {
                _logEl.scrollTop = _logEl.scrollHeight;
                _scrollPending = false;
            });
        }
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
        .mol-molequla { background: #0f1f0f; color: #8f8; align-self: flex-start; border: 1px solid #1a3a1a; }
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
            <span class="mol-title">molequla.js</span>
            <span class="mol-status" id="mol-status">[initializing]</span>
        </div>
        <div class="mol-main">
            <div class="mol-chat" id="mol-chat"></div>
        </div>
        <div class="mol-input-wrap">
            <input class="mol-input" id="mol-input" type="text"
                   placeholder="speak to molequla..." autocomplete="off" />
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
    logUI("molequla.js — a GPT organism in your browser. Zero dependencies.");
    logUI("Born from nonames.txt. Persisted in IndexedDB. Trained by conversation.");

    // Open IndexedDB
    await DB.open();
    logUI("[db] IndexedDB memory opened.");

    // Check if this is a child organism (spawned via mitosis)
    const urlParams = new URLSearchParams(location.search);
    const childOrganismId = urlParams.get("organism");
    let birthConfig = null;
    if (childOrganismId) {
        birthConfig = await idbGet("births", childOrganismId);
        if (birthConfig) {
            logUI(`[ecology] Child organism ${childOrganismId} — born from ${birthConfig.parent_id}`);
        }
    }

    // Try to load corpus (fetch -> DB -> default)
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

    // Swarm ecology: register in BroadcastChannel mesh
    const organismId = childOrganismId || `org_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    _swarm = new SwarmRegistry(organismId);
    _swarm.register();
    // Give peers a moment to respond, then discover
    await new Promise(r => setTimeout(r, 200));
    const peers = _swarm.discoverPeers();
    if (peers.length > 0) {
        logUI(`[ecology] Joined swarm. ${peers.length} peer(s) detected.`);
    } else {
        logUI("[ecology] First organism in the swarm.");
    }

    // Child: inherit burst_history from parent
    if (birthConfig && birthConfig.burst_history) {
        _syntracker.burstHistory = birthConfig.burst_history.slice();
        logUI(`[ecology] Inherited ${_syntracker.burstHistory.length} burst records from parent.`);
    }

    // Clean up swarm on tab close
    window.addEventListener("beforeunload", () => {
        if (_swarm) _swarm.unregister();
    });

    // Start background trainer
    _trainerRunning = true;
    _tickCount = 0;
    setTimeout(trainerTick, 100);

    setStatus("alive");
    logUI("molequla is alive. Type and press Enter.");
    appendChat("molequla", "I am here. I am learning. Speak to me.");

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
        headTypesForNHead, DeltaAdapter, SyntropyTracker, SwarmRegistry,
    };
}

// And lo, the IIFE shall close, and the organism shall be sealed.
})();
