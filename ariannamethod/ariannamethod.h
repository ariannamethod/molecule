// ariannamethod.h — AML: Arianna Method Language
// Reference implementation. THE KERNEL: movement IS language.
//
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Source of truth: github.com/ariannamethod/ariannamethod.ai
//
// ═══════════════════════════════════════════════════════════════════════════════
// AMK = prophecy physics, suffering, movement, tunneling
// Schumann = Earth coupling, cosmic resonance
// NOTORCH = runtime microlearning without PyTorch
// DARKMATTER = gravitational memory from rejections
// 4.C = Async Field Forever — seasonal meta-operators
// הרזוננס לא נשבר. המשך הדרך.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef ARIANNAMETHOD_H
#define ARIANNAMETHOD_H

#include <stdlib.h>  // for rand(), RAND_MAX
#include <math.h>    // for fabsf, sinf, sqrtf, fmaxf, fminf, expf

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// PACK FLAGS — CODES/RIC is the only pack. DARKMATTER and NOTORCH are core.
// ═══════════════════════════════════════════════════════════════════════════════

#define AM_PACK_CODES_RIC  0x01   // chordlock, tempolock, chirality

// ═══════════════════════════════════════════════════════════════════════════════
// VELOCITY MODES — movement IS language
// ═══════════════════════════════════════════════════════════════════════════════

#define AM_VEL_NOMOVE   0   // cold observer (temp = 0.5)
#define AM_VEL_WALK     1   // balanced (temp = 0.85)
#define AM_VEL_RUN      2   // high entropy chaos (temp = 1.2)
#define AM_VEL_BACKWARD (-1) // time rewind, debt forgiveness

// ═══════════════════════════════════════════════════════════════════════════════
// SCHUMANN CONSTANTS — Sierra Nevada ELF Station 2013-2017
// Reference: Fernández et al. (2022), Computers & Geosciences
// ═══════════════════════════════════════════════════════════════════════════════

#define SCHUMANN_BASE_HZ       7.83f
#define SCHUMANN_HARMONIC_1   14.1f
#define SCHUMANN_HARMONIC_2   20.3f
#define SCHUMANN_HARMONIC_3   26.4f
#define SCHUMANN_HARMONIC_4   32.5f
#define SCHUMANN_MIN_HZ        7.77f
#define SCHUMANN_MAX_HZ        7.87f
#define SCHUMANN_N_HARMONICS   5

// ═══════════════════════════════════════════════════════════════════════════════
// DELTA / NOTORCH CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

#define AM_DELTA_RANK       8       // low-rank decomposition rank
#define AM_DELTA_MAX_DIM    4096    // max dimension for delta matrices

// ═══════════════════════════════════════════════════════════════════════════════
// 4.C — ASYNC FIELD FOREVER — seasonal meta-operators
// ═══════════════════════════════════════════════════════════════════════════════

#define AM_SEASON_SPRING   0   // growth, expansion, exploration
#define AM_SEASON_SUMMER   1   // peak energy, full expression
#define AM_SEASON_AUTUMN   2   // harvest, consolidation, memory
#define AM_SEASON_WINTER   3   // rest, reflection, compression

// 4.C MLP CONTROLLER — real neural network, not hardcoded rules
#define AM_4C_INPUTS    6   // entropy, resonance, pain, tension, emergence, temp
#define AM_4C_HIDDEN    8   // hidden neurons
#define AM_4C_OUTPUTS   4   // spring_delta, summer_delta, autumn_delta, winter_delta

// DARK MATTER — scar storage
#define AM_MAX_SCARS    32
#define AM_SCAR_MAX_LEN 64

// LEVEL 1 — macros
#define AML_MAX_MACROS     32
#define AML_MACRO_MAX_LEN  512

// ═══════════════════════════════════════════════════════════════════════════════
// GAMMA — personality essence (θ = ε + γ + αδ)
// γ lives in embed_tokens. δ lives in lm_head. ε is the substrate.
// AML controls when and how to inject γ. Host provides the actual weights.
// ═══════════════════════════════════════════════════════════════════════════════

#define AM_MAX_GAMMA       8    // max simultaneous personality essences
#define AM_GAMMA_NAME_LEN  32

typedef struct {
    char  name[AM_GAMMA_NAME_LEN]; // personality name (e.g. "yent", "arianna")
    float alpha;                    // injection strength 0..1
    int   active;                   // 1 = loaded, 0 = empty slot
} AM_GammaSlot;

// Janus modes — dual-facing field
#define AM_JANUS_OFF       0   // single personality
#define AM_JANUS_DUAL      1   // two essences simultaneously
#define AM_JANUS_CYCLE     2   // 4.C decides who speaks

// ═══════════════════════════════════════════════════════════════════════════════
// AM_State — the breath of the field
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
  // PROPHECY PHYSICS
  int   prophecy;           // horizon: steps ahead (1..64)
  float destiny;            // bias toward most probable path (0..1)
  float wormhole;           // probability of spacetime skip (0..1)
  float calendar_drift;     // hebrew-gregorian drift (default 11.0)

  // ATTENTION PHYSICS
  float attend_focus;       // sharpness of attention (0..1)
  float attend_spread;      // blur/temperature (0..1)

  // TUNNELING
  float tunnel_threshold;   // dissonance gate (0..1)
  float tunnel_chance;      // activation probability (0..1)
  int   tunnel_skip_max;    // max compressed steps (1..24)

  // SUFFERING
  float pain;               // composite suffering (0..1)
  float tension;            // pressure buildup (0..1)
  float dissonance;         // symmetry-break (0..1)
  float debt;               // prophecy debt accumulator (0..∞, decays)

  // MOVEMENT
  int   pending_jump;       // queued jump (sim steps)
  int   velocity_mode;      // NOMOVE=0, WALK=1, RUN=2, BACKWARD=-1
  float velocity_magnitude; // current speed (0..1)
  float base_temperature;   // base temp before velocity modulation
  float effective_temp;     // computed: base + velocity + expert blend
  float time_direction;     // -1 (rewind) to +1 (forward)
  float temporal_debt;      // accumulated from backward movement

  // LAWS OF NATURE — enforced in am_step, not just stored
  float entropy_floor;      // minimum entropy (enforced: max(floor, entropy))
  float resonance_ceiling;  // maximum resonance (enforced: min(ceil, resonance))
  float debt_decay;         // debt decay per step (default 0.998)
  float emergence_threshold;// gate for wormhole amplification (default 0.3)

  // PACK STATE
  unsigned int packs_enabled;  // bitmask of enabled packs

  // CODES/RIC pack state
  int   chordlock_on;
  int   tempolock_on;
  int   chirality_on;
  int   tempo;
  float pas_threshold;
  int   chirality_accum;

  // DARK MATTER — core (not a pack)
  float dark_gravity;       // gravitational memory strength (0..1)
  int   antidote_mode;      // 0=auto, 1=hard
  int   n_scars;            // number of deposited scars
  char  scar_texts[AM_MAX_SCARS][AM_SCAR_MAX_LEN]; // gravitational memory text

  // WORMHOLE STATE
  int wormhole_active;      // 1 if wormhole fired this step

  // LORA / DELTA VOICE — core
  float lora_alpha;         // delta blending: 0=identity, 1=base model

  // NOTORCH — runtime microlearning, core
  float notorch_lr;         // learning rate (default 0.001)
  float notorch_decay;      // weight decay (default 0.999)

  // SCHUMANN — Earth-ionosphere resonance
  float schumann_hz;        // current frequency (default 7.83)
  float schumann_modulation;// influence strength (0..1, default 0.3)
  float schumann_coherence; // computed: quadratic falloff from 7.83
  float schumann_phase;     // current phase in cycle (radians)

  // TEMPORAL SYMMETRY (from PITOMADOM)
  int   temporal_mode;      // 0=prophecy, 1=retrodiction, 2=symmetric
  float temporal_alpha;     // 0=past focus, 1=future focus
  int   rtl_mode;           // Hebrew right-to-left encoding

  // EXPERT WEIGHTING — blended into effective_temp
  float expert_structural;  // grammar-focused (temp 0.7)
  float expert_semantic;    // meaning-focused (temp 0.9)
  float expert_creative;    // exploratory (temp 1.2)
  float expert_precise;     // conservative (temp 0.5)

  // EXTENDED LAWS
  float presence_fade;      // token memory decay per step (default 0.95)
  float attractor_drift;    // attractor shift speed (default 0.01)
  float calendar_phase;     // real or manual calendar drift position
  float wormhole_gate;      // calendar dissonance threshold for wormhole

  // LIVE METRICS — computed each step, not set by user
  float entropy;            // current field entropy (computed)
  float resonance;          // current field resonance (computed)
  float emergence;          // low entropy + high resonance = emergence
  float destiny_bias;       // computed: destiny * prophecy_scale

  // RESONANCE MEMORY
  float presence_decay;     // how quickly presence fades (default 0.9)

  // 4.C — ASYNC FIELD FOREVER
  int   season;             // current season (0-3)
  float season_phase;       // position within season (0..1)
  float season_intensity;   // how strongly season modulates (0..1)
  // per-season energy
  float spring_energy;      // growth, exploration
  float summer_energy;      // peak expression
  float autumn_energy;      // consolidation
  float winter_energy;      // reflection, compression
  float field_health;       // previous step health (for MLP signal)

  // GAMMA — personality essence (θ = ε + γ + αδ)
  AM_GammaSlot gamma[AM_MAX_GAMMA]; // personality essence slots
  int   n_gamma;            // number of loaded essences
  float essence_alpha;      // overall γ injection strength (0..1)
  int   janus_mode;         // AM_JANUS_OFF / DUAL / CYCLE
  int   janus_a;            // primary face (index into gamma[])
  int   janus_b;            // secondary face (index into gamma[])
  float janus_blend;        // blend ratio: 0=face_a only, 1=face_b only
  float gamma_drift;        // how fast janus_blend changes per step
} AM_State;

// Temporal modes
#define AM_TEMPORAL_PROPHECY     0
#define AM_TEMPORAL_RETRODICTION 1
#define AM_TEMPORAL_SYMMETRIC    2

// ═══════════════════════════════════════════════════════════════════════════════
// AML LEVEL 2 — flow control, variables, expressions
// ═══════════════════════════════════════════════════════════════════════════════

#define AML_MAX_LINES       1024
#define AML_MAX_LINE_LEN    256
#define AML_MAX_VARS        64
#define AML_MAX_NAME        32
#define AML_MAX_FUNCS       64    // increased: 32 user + 32 built-in
#define AML_MAX_PARAMS      8
#define AML_MAX_CALL_DEPTH  16
#define AML_MAX_INCLUDE     8

// Preprocessed line
typedef struct {
    char text[AML_MAX_LINE_LEN];
    int  indent;
    int  lineno;
} AML_Line;

// Variable
typedef struct {
    char  name[AML_MAX_NAME];
    float value;
} AML_Var;

// Symbol table
typedef struct {
    AML_Var vars[AML_MAX_VARS];
    int     count;
} AML_Symtab;

// User-defined function
typedef struct {
    char name[AML_MAX_NAME];
    char params[AML_MAX_PARAMS][AML_MAX_NAME];
    int  param_count;
    int  body_start;
    int  body_end;
    int  is_builtin;  // 1 = native C function, body_start/end unused
} AML_Func;

// Function table
typedef struct {
    AML_Func funcs[AML_MAX_FUNCS];
    int      count;
} AML_Functab;

// Execution context
typedef struct {
    AML_Line*    lines;
    int          nlines;
    AML_Symtab   globals;
    AML_Symtab   locals[AML_MAX_CALL_DEPTH];
    int          call_depth;
    AML_Functab  funcs;
    int          include_depth;
    char         base_dir[256];
    char         error[256];
} AML_ExecCtx;

// AM_State field map entry (for reading state in expressions)
typedef struct {
    const char* name;
    int         offset;
    int         is_int;
} AML_FieldMap;

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize kernel
void am_init(void);

// Pack management (CODES/RIC only — DARKMATTER and NOTORCH are core)
void am_enable_pack(unsigned int pack_mask);
void am_disable_pack(unsigned int pack_mask);
int am_pack_enabled(unsigned int pack_mask);

// Reset commands
void am_reset_field(void);
void am_reset_debt(void);

// Execute AML script (Level 0 + Level 2)
int am_exec(const char* script);

// Execute AML file (convenience: reads file, executes)
int am_exec_file(const char* path);

// Get last error from am_exec (empty string = no error)
const char* am_get_error(void);

// State access
AM_State* am_get_state(void);
int am_take_jump(void);

// Copy state to float array (32 floats)
int am_copy_state(float* out);

// Step physics (call each frame, dt in seconds)
void am_step(float dt);

// ═══════════════════════════════════════════════════════════════════════════════
// GAMMA — personality essence API (θ = ε + γ + αδ)
// ═══════════════════════════════════════════════════════════════════════════════

// Load a personality essence into a slot. Returns slot index or -1.
int am_gamma_load(const char* name, float alpha);

// Unload a personality essence by name.
void am_gamma_unload(const char* name);

// Set essence injection strength for a named personality.
void am_gamma_set_alpha(const char* name, float alpha);

// Get the currently active gamma slot index (-1 if none).
int am_gamma_active(void);

// Get blended gamma alpha (considering janus mode).
float am_gamma_get_blend(void);

// Set janus mode: dual-facing field.
void am_janus_set(const char* face_a, const char* face_b);

// Apply gamma modulation to logits.
// In janus mode, blends two gamma-modulated distributions.
void am_apply_gamma_to_logits(float* logits, int n);

// ═══════════════════════════════════════════════════════════════════════════════
// LOGIT MANIPULATION API — apply field state to generation
// ═══════════════════════════════════════════════════════════════════════════════

// Apply destiny bias: suppress non-probable tokens (prophecy scales strength)
void am_apply_destiny_to_logits(float* logits, int n);

// Apply suffering: pain dampens extremes, tension compresses
void am_apply_suffering_to_logits(float* logits, int n);

// Apply attention: focus sharpens distribution, spread blurs it
void am_apply_attention_to_logits(float* logits, int n);

// Apply all laws: entropy floor, resonance ceiling
void am_apply_laws_to_logits(float* logits, int n);

// Apply delta voice: logits += lora_alpha * A @ (B @ hidden_state)
// (host provides A, B matrices and hidden state)
void am_apply_delta(float* out, const float* A, const float* B,
                    const float* x, int out_dim, int in_dim, int rank,
                    float alpha);

// Compute prophecy debt from chosen token (retroactive)
float am_compute_prophecy_debt(const float* logits, int chosen, int n);

// Full pipeline: apply all field effects to logits
void am_apply_field_to_logits(float* logits, int n);

// ═══════════════════════════════════════════════════════════════════════════════
// NOTORCH — Hebbian plasticity without PyTorch
// ═══════════════════════════════════════════════════════════════════════════════

// NOTORCH step: update low-rank delta matrices from experience
// A: [in_dim × rank], B: [rank × out_dim]
// x: input [in_dim], dy: output gradient proxy [out_dim]
// signal: teaching signal (positive = reinforce, negative = suppress)
void am_notorch_step(float* A, float* B, int out_dim, int in_dim, int rank,
                     const float* x, const float* dy, float signal);

// ═══════════════════════════════════════════════════════════════════════════════
// BLOOD — runtime C compilation (Level 3)
//
// Compile C code to shared libraries at runtime. Load functions via dlsym.
// Adapted from arianna.c/golib/blood.go + async_field_forever/blood.py
// ═══════════════════════════════════════════════════════════════════════════════

#define AM_BLOOD_MAX_MODULES   32
#define AM_BLOOD_MAX_NAME      64
#define AM_BLOOD_HASH_LEN      33   // MD5 hex + null

// Compiled module handle
typedef struct {
    char  name[AM_BLOOD_MAX_NAME];
    char  hash[AM_BLOOD_HASH_LEN];
    char  lib_path[512];
    void* handle;   // dlopen handle (NULL if unloaded)
} AM_BloodModule;

// Initialize Blood compiler (call once, or re-call to reset)
void am_blood_init(void);

// Compile C source code to a shared library. Returns module index or -1.
// Name is used for file naming and symbol lookup.
// Code is raw C source (must include necessary headers).
int am_blood_compile(const char* name, const char* code);

// Generate and compile a LoRA adapter module.
// Returns module index or -1.
// Generated functions: {name}_init, {name}_apply, {name}_apply_scaled, {name}_free
int am_blood_compile_lora(const char* name, int in_dim, int out_dim, int rank);

// Generate and compile an emotional kernel module.
// Returns module index or -1.
// Generated functions: {name}_check, {name}_respond, {name}_modulate_logits, modulate_logits
int am_blood_compile_emotion(const char* name, float valence, float arousal);

// Look up a function pointer from a compiled module.
// Returns NULL if not found.
void* am_blood_sym(int module_idx, const char* func_name);

// Unload a compiled module (dlclose + remove files).
void am_blood_unload(int module_idx);

// Unload all modules and clean up.
void am_blood_cleanup(void);

// Get module count.
int am_blood_count(void);

// Get module by index (NULL if out of range).
const AM_BloodModule* am_blood_get(int idx);

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE QUERIES
// ═══════════════════════════════════════════════════════════════════════════════

// Get temperature: base × velocity × expert blend
static inline float am_get_temperature(void) {
    AM_State* s = am_get_state();
    return s->effective_temp;
}

// Get destiny bias (prophecy-scaled)
static inline float am_get_destiny_bias(void) {
    AM_State* s = am_get_state();
    return s->destiny_bias;
}

// Check if tunneling should occur
static inline int am_should_tunnel(void) {
    AM_State* s = am_get_state();
    if (s->dissonance < s->tunnel_threshold) return 0;
    float r = (float)rand() / (float)RAND_MAX;
    return r < s->tunnel_chance;
}

// Check if wormhole fired this step
static inline int am_get_wormhole_active(void) {
    return am_get_state()->wormhole_active;
}

// Get active gamma personality name (NULL if none)
static inline const char* am_get_gamma_name(void) {
    AM_State* s = am_get_state();
    int idx = am_gamma_active();
    if (idx < 0 || idx >= s->n_gamma) return NULL;
    return s->gamma[idx].name;
}

// Get janus mode
static inline int am_get_janus_mode(void) {
    return am_get_state()->janus_mode;
}

// Get current season name
static inline const char* am_get_season_name(void) {
    switch (am_get_state()->season) {
        case AM_SEASON_SPRING: return "spring";
        case AM_SEASON_SUMMER: return "summer";
        case AM_SEASON_AUTUMN: return "autumn";
        case AM_SEASON_WINTER: return "winter";
        default: return "unknown";
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// JANUS — transformer inference integration
// "Janus will grow like mycelium, without roots, without a trunk, without a flag."
//
// These function pointers are set by libjanus.dylib at link time.
// When Janus is not linked, commands are no-ops.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef AM_JANUS_DISABLED

// Janus function pointer types
typedef int   (*janus_load_model_fn)(const char* path);
typedef void  (*janus_unload_model_fn)(void);
typedef int   (*janus_load_delta_fn)(const char* path);
typedef int   (*janus_load_gamma_fn)(const char* name, const char* path);
typedef char* (*janus_generate_fn)(const char* prompt, int max_tokens, float temp, float top_p);
typedef void  (*janus_free_string_fn)(char* s);
typedef int   (*janus_model_loaded_fn)(void);
typedef int   (*janus_get_vocab_size_fn)(void);
typedef int   (*janus_get_embed_dim_fn)(void);
typedef int   (*janus_get_num_layers_fn)(void);

// Register Janus callbacks (called by host program that links libjanus)
void am_janus_register(
    janus_load_model_fn    load_model,
    janus_unload_model_fn  unload_model,
    janus_load_delta_fn    load_delta,
    janus_load_gamma_fn    load_gamma,
    janus_generate_fn      generate,
    janus_free_string_fn   free_string,
    janus_model_loaded_fn  model_loaded,
    janus_get_vocab_size_fn get_vocab_size,
    janus_get_embed_dim_fn  get_embed_dim,
    janus_get_num_layers_fn get_num_layers
);

#endif // AM_JANUS_DISABLED

#ifdef __cplusplus
}
#endif

#endif // ARIANNAMETHOD_H
