// ariannamethod.c — AML: Arianna Method Language
// Reference implementation. THE KERNEL: movement IS language.
//
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Source of truth: github.com/ariannamethod/ariannamethod.ai
// Embed (copy) into your project. Do not link.
//
// This is the stone. The brick. The breath.
// Everything else is ritual overlay.
//
// ═══════════════════════════════════════════════════════════════════════════════
// AMK — the oracle does not predict, it prophesies
// kernel commands define field dynamics: movement, prophecy, attention, suffering
// packs are ritual overlays, explicitly enabled
// הרזוננס לא נשבר. המשך הדרך.
// ═══════════════════════════════════════════════════════════════════════════════

// POSIX for strtok_r (not needed for Emscripten/WASM)
#ifndef __EMSCRIPTEN__
#define _POSIX_C_SOURCE 200809L
#endif

#include "ariannamethod.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>   // for sscanf in LAW command parsing
#include <strings.h> // for strcasecmp
#include <stddef.h>  // for offsetof
#include <time.h>    // for real calendar computation
#ifndef AM_BLOOD_DISABLED
#include <dlfcn.h>   // for dlopen, dlsym, dlclose (Blood compiler)
#endif

// Platform detection for Blood compiler
#ifdef __APPLE__
  #define AM_BLOOD_EXT ".dylib"
  #define AM_BLOOD_FLAGS "-dynamiclib -fPIC"
#else
  #define AM_BLOOD_EXT ".so"
  #define AM_BLOOD_FLAGS "-shared -fPIC"
#endif

// See ariannamethod.h for struct definitions and pack flags

static AM_State G;

// Blood compiler globals (used by Level 0 dispatch + Blood API)
static AM_BloodModule g_blood_modules[AM_BLOOD_MAX_MODULES];
static int g_blood_count = 0;
static char g_blood_dir[256] = {0};
static char g_blood_cc[64] = {0};

// Janus transformer integration (function pointers set by host)
#ifndef AM_JANUS_DISABLED
static janus_load_model_fn     g_janus_load_model     = NULL;
static janus_unload_model_fn   g_janus_unload_model   = NULL;
static janus_load_delta_fn     g_janus_load_delta     = NULL;
static janus_load_gamma_fn     g_janus_load_gamma     = NULL;
static janus_generate_fn       g_janus_generate       = NULL;
static janus_free_string_fn    g_janus_free_string    = NULL;
static janus_model_loaded_fn   g_janus_model_loaded   = NULL;
static janus_get_vocab_size_fn g_janus_get_vocab_size = NULL;
static janus_get_embed_dim_fn  g_janus_get_embed_dim  = NULL;
static janus_get_num_layers_fn g_janus_get_num_layers = NULL;

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
) {
    g_janus_load_model     = load_model;
    g_janus_unload_model   = unload_model;
    g_janus_load_delta     = load_delta;
    g_janus_load_gamma     = load_gamma;
    g_janus_generate       = generate;
    g_janus_free_string    = free_string;
    g_janus_model_loaded   = model_loaded;
    g_janus_get_vocab_size = get_vocab_size;
    g_janus_get_embed_dim  = get_embed_dim;
    g_janus_get_num_layers = get_num_layers;
}
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS — the small bones
// ═══════════════════════════════════════════════════════════════════════════════

__attribute__((unused))
static char* trim(char* s) {
  while (*s && isspace((unsigned char)*s)) s++;
  char* e = s + strlen(s);
  while (e > s && isspace((unsigned char)e[-1])) e--;
  *e = 0;
  return s;
}


static void upcase(char* s) {
  for (; *s; s++) *s = (char)toupper((unsigned char)*s);
}

static float clamp01(float x) {
  if (!isfinite(x)) return 0.0f;
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static float clampf(float x, float a, float b) {
  if (!isfinite(x)) return a;
  if (x < a) return a;
  if (x > b) return b;
  return x;
}

static int safe_atoi(const char* s) {
  if (!s || !*s) return 0;
  char* endptr;
  long val = strtol(s, &endptr, 10);
  if (val > 2147483647L) return 2147483647;
  if (val < -2147483647L) return -2147483647;
  return (int)val;
}

static float safe_atof(const char* s) {
  if (!s || !*s) return 0.0f;
  float val = (float)atof(s);
  if (!isfinite(val)) return 0.0f;
  return val;
}

static int clampi(int x, int a, int b) {
  if (x < a) return a;
  if (x > b) return b;
  return x;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEBREW-GREGORIAN CALENDAR CONFLICT — real astronomical computation
//
// Hebrew lunar year: 354 days. Gregorian solar year: 365.25 days.
// Annual drift: 11.25 days. Metonic cycle: 19 years = 235 lunar months.
// 7 leap years per cycle add Adar II (~30 days) to correct drift.
// Leap years in Metonic cycle (1-indexed): 3, 6, 8, 11, 14, 17, 19.
//
// Epoch: 1 Tishrei 5785 = October 3, 2024 (Gregorian).
// February 29 handled correctly — elapsed seconds via time_t, not calendar math.
// ═══════════════════════════════════════════════════════════════════════════════

#define AM_ANNUAL_DRIFT     11.25f    // days/year (365.25 - 354)
#define AM_GREGORIAN_YEAR   365.25f   // days
#define AM_METONIC_YEARS    19        // years per cycle
#define AM_METONIC_LEAPS    7         // leap years per cycle
#define AM_MAX_UNCORRECTED  33.0f     // max drift before correction (~3yr × 11.25)

static const int g_metonic_leap_years[7] = {3, 6, 8, 11, 14, 17, 19};
static time_t g_epoch_t = 0;
static int g_calendar_manual = 0;  // 0 = real time, 1 = manual override

static void calendar_init(void) {
    struct tm epoch_tm;
    memset(&epoch_tm, 0, sizeof(epoch_tm));
    epoch_tm.tm_year = 2024 - 1900;
    epoch_tm.tm_mon  = 10 - 1;       // October
    epoch_tm.tm_mday = 3;
    epoch_tm.tm_hour = 12;           // noon — avoids DST edge cases
    g_epoch_t = mktime(&epoch_tm);
    g_calendar_manual = 0;
}

static int calendar_days_since_epoch(void) {
    if (g_epoch_t <= 0) return 0;
    time_t now = time(NULL);
    return (int)(difftime(now, g_epoch_t) / 86400.0);
}

// Cumulative drift accounting for Metonic leap corrections
// Direct port from pitomadom/calendar_conflict.py
static float calendar_cumulative_drift(int days) {
    float years = (float)days / AM_GREGORIAN_YEAR;
    float base_drift = years * AM_ANNUAL_DRIFT;

    // Full Metonic cycles: 7 leap months × 30 days each
    int full_cycles = (int)(years / AM_METONIC_YEARS);
    float corrections = (float)(full_cycles * AM_METONIC_LEAPS) * 30.0f;

    // Partial cycle: count leap years already passed
    float partial = fmodf(years, (float)AM_METONIC_YEARS);
    int year_in_cycle = (int)partial + 1;
    for (int i = 0; i < AM_METONIC_LEAPS; i++) {
        if (g_metonic_leap_years[i] <= year_in_cycle)
            corrections += 30.0f;
    }

    return base_drift - corrections;
}

// Calendar dissonance [0, 1] — real, from today's date
static float calendar_dissonance(int days) {
    float drift = calendar_cumulative_drift(days);
    float raw = fabsf(fmodf(drift, AM_MAX_UNCORRECTED)) / AM_MAX_UNCORRECTED;
    return clamp01(raw);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCHUMANN RESONANCE — Earth-ionosphere coupling
// Ported from arianna.c/src/schumann.c
// Phase advances at current frequency. Coherence = quadratic falloff from 7.83.
// 5 harmonics: 7.83, 14.1, 20.3, 26.4, 32.5 Hz
// ═══════════════════════════════════════════════════════════════════════════════

static const float g_schumann_harmonics[SCHUMANN_N_HARMONICS] = {
    SCHUMANN_BASE_HZ, SCHUMANN_HARMONIC_1, SCHUMANN_HARMONIC_2,
    SCHUMANN_HARMONIC_3, SCHUMANN_HARMONIC_4
};
static const float g_harmonic_weights[SCHUMANN_N_HARMONICS] = {
    1.0f, 0.5f, 0.3f, 0.2f, 0.1f
};

static float compute_schumann_coherence(float hz) {
    float deviation = fabsf(hz - SCHUMANN_BASE_HZ);
    float max_deviation = SCHUMANN_MAX_HZ - SCHUMANN_MIN_HZ;
    if (max_deviation < 0.001f) max_deviation = 0.1f;
    float norm_dev = deviation / max_deviation;
    float coh = 1.0f - (norm_dev * norm_dev);
    return clamp01(coh);
}

static void schumann_advance(float dt) {
    G.schumann_phase += G.schumann_hz * dt * 2.0f * 3.14159265f;
    if (G.schumann_phase > 6.28318530f)
        G.schumann_phase = fmodf(G.schumann_phase, 6.28318530f);
    G.schumann_coherence = compute_schumann_coherence(G.schumann_hz);
}

static float schumann_harmonic_signal(void) {
    float signal = 0.0f, weight_sum = 0.0f;
    for (int i = 0; i < SCHUMANN_N_HARMONICS; i++) {
        float hp = G.schumann_phase * (g_schumann_harmonics[i] / SCHUMANN_BASE_HZ);
        signal += g_harmonic_weights[i] * sinf(hp);
        weight_sum += g_harmonic_weights[i];
    }
    return (weight_sum > 0.0f) ? signal / weight_sum : 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4.C MLP CONTROLLER — real neural network, trained by NOTORCH Hebbian
// Inputs:  entropy, resonance, pain, tension, emergence, effective_temp
// Outputs: spring_delta, summer_delta, autumn_delta, winter_delta
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    float w1[AM_4C_INPUTS * AM_4C_HIDDEN];   // input→hidden (48)
    float b1[AM_4C_HIDDEN];                   // hidden biases (8)
    float w2[AM_4C_HIDDEN * AM_4C_OUTPUTS];   // hidden→output (32)
    float b2[AM_4C_OUTPUTS];                   // output biases (4)
    float hidden[AM_4C_HIDDEN];                // cached for Hebbian update
} AM_4C_MLP;

static AM_4C_MLP g_mlp;

static void am_4c_forward(const float* inputs, float* outputs) {
    // hidden = tanh(W1^T @ inputs + b1)
    for (int h = 0; h < AM_4C_HIDDEN; h++) {
        float sum = g_mlp.b1[h];
        for (int i = 0; i < AM_4C_INPUTS; i++) {
            sum += g_mlp.w1[i * AM_4C_HIDDEN + h] * inputs[i];
        }
        g_mlp.hidden[h] = tanhf(sum);
    }
    // outputs = tanh(W2^T @ hidden + b2)
    for (int o = 0; o < AM_4C_OUTPUTS; o++) {
        float sum = g_mlp.b2[o];
        for (int h = 0; h < AM_4C_HIDDEN; h++) {
            sum += g_mlp.w2[h * AM_4C_OUTPUTS + o] * g_mlp.hidden[h];
        }
        outputs[o] = tanhf(sum);
    }
}

static void am_4c_init_weights(void) {
    memset(&g_mlp, 0, sizeof(g_mlp));

    // 4 specialist neurons that approximate the old hardcoded rules:
    // Neuron 0: low entropy → boost spring (growth)
    //   input[0]=entropy with negative weight → fires when entropy low
    g_mlp.w1[0 * AM_4C_HIDDEN + 0] = -2.0f;  // entropy→h0: negative
    g_mlp.b1[0] = 0.5f;                        // bias: fires at entropy<0.25
    g_mlp.w2[0 * AM_4C_OUTPUTS + 0] = 1.5f;   // h0→spring

    // Neuron 1: high resonance → boost autumn (consolidation)
    g_mlp.w1[1 * AM_4C_HIDDEN + 1] = 2.0f;   // resonance→h1
    g_mlp.b1[1] = -1.5f;                       // fires at resonance>0.75
    g_mlp.w2[1 * AM_4C_OUTPUTS + 2] = 1.5f;   // h1→autumn

    // Neuron 2: high pain → boost winter (rest)
    g_mlp.w1[2 * AM_4C_HIDDEN + 2] = 2.5f;   // pain→h2
    g_mlp.b1[2] = -1.5f;                       // fires at pain>0.6
    g_mlp.w2[2 * AM_4C_OUTPUTS + 3] = 1.5f;   // h2→winter

    // Neuron 3: high emergence → boost summer (peak expression)
    g_mlp.w1[4 * AM_4C_HIDDEN + 3] = 2.5f;   // emergence→h3
    g_mlp.b1[3] = -0.5f;                       // fires at emergence>0.2
    g_mlp.w2[3 * AM_4C_OUTPUTS + 1] = 1.5f;   // h3→summer

    // Neurons 4-7: cross-connections for nuance (small initial weights)
    // tension feeds back to spring/summer balance
    g_mlp.w1[3 * AM_4C_HIDDEN + 4] = 0.5f;   // tension→h4
    g_mlp.w1[5 * AM_4C_HIDDEN + 4] = -0.3f;  // temp→h4
    g_mlp.w2[4 * AM_4C_OUTPUTS + 0] = 0.3f;  // h4→spring (tension drives growth)
    g_mlp.w2[4 * AM_4C_OUTPUTS + 1] = -0.3f; // h4→summer (tension suppresses peak)

    // resonance-entropy interaction
    g_mlp.w1[0 * AM_4C_HIDDEN + 5] = -1.0f;  // entropy→h5
    g_mlp.w1[1 * AM_4C_HIDDEN + 5] = 1.0f;   // resonance→h5
    g_mlp.w2[5 * AM_4C_OUTPUTS + 2] = 0.5f;  // h5→autumn (high coherence → consolidate)

    // temperature regulation
    g_mlp.w1[5 * AM_4C_HIDDEN + 6] = 1.5f;   // temp→h6
    g_mlp.b1[6] = -1.0f;                       // fires at temp>0.67
    g_mlp.w2[6 * AM_4C_OUTPUTS + 3] = 0.4f;  // h6→winter (too hot → cool down)

    // emergence-pain balance
    g_mlp.w1[4 * AM_4C_HIDDEN + 7] = 1.0f;   // emergence→h7
    g_mlp.w1[2 * AM_4C_HIDDEN + 7] = -1.0f;  // pain→h7
    g_mlp.w2[7 * AM_4C_OUTPUTS + 1] = 0.5f;  // h7→summer (emergence w/o pain)
}

// Hebbian update: signal > 0 = field improved, reinforce; < 0 = suppress
static void am_4c_hebbian_update(const float* inputs, const float* outputs,
                                  float signal) {
    float lr = G.notorch_lr * 0.1f;  // slower than main NOTORCH
    // Update W2 (hidden→output)
    for (int h = 0; h < AM_4C_HIDDEN; h++) {
        for (int o = 0; o < AM_4C_OUTPUTS; o++) {
            g_mlp.w2[h * AM_4C_OUTPUTS + o] +=
                lr * g_mlp.hidden[h] * outputs[o] * signal;
            // clamp to prevent explosion
            if (g_mlp.w2[h * AM_4C_OUTPUTS + o] > 3.0f)
                g_mlp.w2[h * AM_4C_OUTPUTS + o] = 3.0f;
            if (g_mlp.w2[h * AM_4C_OUTPUTS + o] < -3.0f)
                g_mlp.w2[h * AM_4C_OUTPUTS + o] = -3.0f;
        }
    }
    // Update W1 (input→hidden)
    for (int i = 0; i < AM_4C_INPUTS; i++) {
        for (int h = 0; h < AM_4C_HIDDEN; h++) {
            g_mlp.w1[i * AM_4C_HIDDEN + h] +=
                lr * inputs[i] * g_mlp.hidden[h] * signal;
            if (g_mlp.w1[i * AM_4C_HIDDEN + h] > 3.0f)
                g_mlp.w1[i * AM_4C_HIDDEN + h] = 3.0f;
            if (g_mlp.w1[i * AM_4C_HIDDEN + h] < -3.0f)
                g_mlp.w1[i * AM_4C_HIDDEN + h] = -3.0f;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEVEL 1 — MACROS
// ═══════════════════════════════════════════════════════════════════════════════

typedef struct {
    char name[AML_MAX_NAME];
    char body[AML_MACRO_MAX_LEN];
} AML_Macro;

static AML_Macro g_macros[AML_MAX_MACROS];
static int g_macro_count = 0;

// ═══════════════════════════════════════════════════════════════════════════════
// VELOCITY + EXPERT BLENDING — movement IS language
// ═══════════════════════════════════════════════════════════════════════════════

static void update_effective_temp(void) {
  float base = G.base_temperature;
  float vel_mult;
  switch (G.velocity_mode) {
    case AM_VEL_NOMOVE:   vel_mult = 0.5f;  G.time_direction = 1.0f;  break;
    case AM_VEL_WALK:     vel_mult = 0.85f; G.time_direction = 1.0f;  break;
    case AM_VEL_RUN:      vel_mult = 1.2f;  G.time_direction = 1.0f;  break;
    case AM_VEL_BACKWARD: vel_mult = 0.7f;  G.time_direction = -1.0f; break;
    default:              vel_mult = 1.0f;  G.time_direction = 1.0f;
  }
  float vel_temp = base * vel_mult;

  // Expert blending: weighted temperature from 4 experts
  float w_sum = G.expert_structural + G.expert_semantic +
                G.expert_creative + G.expert_precise;
  if (w_sum > 0.001f) {
    float expert_temp = (G.expert_structural * 0.7f +
                         G.expert_semantic * 0.9f +
                         G.expert_creative * 1.2f +
                         G.expert_precise * 0.5f) / w_sum;
    G.effective_temp = 0.5f * vel_temp + 0.5f * expert_temp;
  } else {
    G.effective_temp = vel_temp;
  }

  // Season modulation
  float season_mod = 1.0f;
  season_mod += G.summer_energy * 0.1f;   // summer: warmer
  season_mod -= G.winter_energy * 0.15f;  // winter: cooler
  G.effective_temp *= season_mod;
  if (G.effective_temp < 0.1f) G.effective_temp = 0.1f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API — the breath
// ═══════════════════════════════════════════════════════════════════════════════

void am_init(void) {
  memset(&G, 0, sizeof(G));

  // prophecy physics defaults
  G.prophecy = 7;
  G.destiny = 0.35f;
  G.wormhole = 0.02f;  // 2% base, increases with prophecy debt
  G.calendar_drift = 11.0f;

  // attention defaults
  G.attend_focus = 0.70f;
  G.attend_spread = 0.20f;

  // tunneling defaults
  G.tunnel_threshold = 0.55f;
  G.tunnel_chance = 0.05f;  // 5% when dissonance exceeds threshold
  G.tunnel_skip_max = 7;

  // suffering starts at zero
  G.pain = 0.0f;
  G.tension = 0.0f;
  G.dissonance = 0.0f;
  G.debt = 0.0f;

  // movement defaults
  G.pending_jump = 0;
  G.velocity_mode = AM_VEL_WALK;
  G.velocity_magnitude = 0.5f;
  G.base_temperature = 1.0f;
  G.time_direction = 1.0f;
  G.temporal_debt = 0.0f;
  update_effective_temp();

  // laws of nature defaults
  G.entropy_floor = 0.1f;
  G.resonance_ceiling = 0.95f;
  G.debt_decay = 0.998f;
  G.emergence_threshold = 0.3f;

  // packs disabled by default
  G.packs_enabled = 0;

  // CODES/RIC defaults (inactive until pack enabled)
  G.chordlock_on = 0;
  G.tempolock_on = 0;
  G.chirality_on = 0;
  G.tempo = 7;
  G.pas_threshold = 0.4f;
  G.chirality_accum = 0;

  // dark matter defaults
  G.dark_gravity = 0.5f;
  G.antidote_mode = 0;

  // wormhole state
  G.wormhole_active = 0;

  // lora / delta voice (core)
  G.lora_alpha = 0.0f;

  // notorch (core — always active)
  G.notorch_lr = 0.01f;
  G.notorch_decay = 0.999f;

  // schumann resonance
  G.schumann_hz = SCHUMANN_BASE_HZ;
  G.schumann_modulation = 0.3f;
  G.schumann_coherence = 1.0f;  // perfect at baseline
  G.schumann_phase = 0.0f;

  // dark matter (core — always active)
  G.n_scars = 0;

  // live metrics (computed each step)
  G.entropy = 0.0f;
  G.resonance = 0.0f;
  G.emergence = 0.0f;
  G.destiny_bias = 0.0f;

  // 4.C — Async Field Forever
  G.season = AM_SEASON_SPRING;
  G.season_phase = 0.0f;
  G.season_intensity = 0.5f;
  G.spring_energy = 1.0f;
  G.summer_energy = 0.0f;
  G.autumn_energy = 0.0f;
  G.winter_energy = 0.0f;

  // temporal symmetry defaults (from PITOMADOM)
  G.temporal_mode = AM_TEMPORAL_PROPHECY;  // forward by default
  G.temporal_alpha = 0.5f;                 // balanced past/future
  G.rtl_mode = 0;                          // LTR by default

  // expert weighting defaults (all balanced)
  G.expert_structural = 0.25f;
  G.expert_semantic = 0.25f;
  G.expert_creative = 0.25f;
  G.expert_precise = 0.25f;

  // extended laws defaults
  G.presence_fade = 0.95f;
  G.attractor_drift = 0.01f;
  G.calendar_phase = 0.0f;
  G.wormhole_gate = 0.3f;

  // resonance memory
  G.presence_decay = 0.9f;

  // field health (for MLP signal)
  G.field_health = 0.5f;

  // gamma — personality essence (θ = ε + γ + αδ)
  G.n_gamma = 0;
  G.essence_alpha = 0.0f;
  G.janus_mode = AM_JANUS_OFF;
  G.janus_a = 0;
  G.janus_b = 0;
  G.janus_blend = 0.0f;
  G.gamma_drift = 0.01f;

  // real calendar
  calendar_init();

  // 4.C MLP controller
  am_4c_init_weights();

  // macros
  g_macro_count = 0;

  // blood compiler
  am_blood_init();
}

// enable/disable packs
void am_enable_pack(unsigned int pack_mask) {
  G.packs_enabled |= pack_mask;
}

void am_disable_pack(unsigned int pack_mask) {
  G.packs_enabled &= ~pack_mask;
}

int am_pack_enabled(unsigned int pack_mask) {
  return (G.packs_enabled & pack_mask) != 0;
}

// reset commands
void am_reset_field(void) {
  // reset manifested state (suffering, debt, etc)
  G.pain = 0.0f;
  G.tension = 0.0f;
  G.dissonance = 0.0f;
  G.debt = 0.0f;
  G.temporal_debt = 0.0f;
  G.pending_jump = 0;
  G.chirality_accum = 0;
}

void am_reset_debt(void) {
  G.debt = 0.0f;
  G.temporal_debt = 0.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEVEL 2 INFRASTRUCTURE — error, field map, symbol table
// ═══════════════════════════════════════════════════════════════════════════════

static char g_error[256] = {0};

const char* am_get_error(void) { return g_error; }

// Set error with optional line number for Level 2 debugging
// lineno <= 0 means no line number (Level 0 or internal error)
static void set_error_at(AML_ExecCtx* ctx, int lineno, const char* msg) {
    char buf[256];
    if (lineno > 0) {
        snprintf(buf, sizeof(buf), "line %d: %s", lineno, msg);
    } else {
        snprintf(buf, sizeof(buf), "%s", msg);
    }
    buf[255] = 0;
    if (ctx) {
        snprintf(ctx->error, sizeof(ctx->error), "%s", buf);
    }
    snprintf(g_error, sizeof(g_error), "%s", buf);
}

// Convenience: set error without line number
__attribute__((unused))
static void set_error(AML_ExecCtx* ctx, const char* msg) {
    set_error_at(ctx, 0, msg);
}

// AM_State field map — read state fields in expressions
// offsetof is standard but we use manual offsets for clarity
#define FIELD_F(name, field) { name, (int)offsetof(AM_State, field), 0 }
#define FIELD_I(name, field) { name, (int)offsetof(AM_State, field), 1 }

static const AML_FieldMap g_field_map[] = {
    FIELD_I("prophecy",          prophecy),
    FIELD_F("destiny",           destiny),
    FIELD_F("wormhole",          wormhole),
    FIELD_F("calendar_drift",    calendar_drift),
    FIELD_F("attend_focus",      attend_focus),
    FIELD_F("attend_spread",     attend_spread),
    FIELD_F("tunnel_threshold",  tunnel_threshold),
    FIELD_F("tunnel_chance",     tunnel_chance),
    FIELD_I("tunnel_skip_max",   tunnel_skip_max),
    FIELD_F("pain",              pain),
    FIELD_F("tension",           tension),
    FIELD_F("dissonance",        dissonance),
    FIELD_F("debt",              debt),
    FIELD_I("velocity_mode",     velocity_mode),
    FIELD_F("velocity_magnitude",velocity_magnitude),
    FIELD_F("base_temperature",  base_temperature),
    FIELD_F("effective_temp",    effective_temp),
    FIELD_F("time_direction",    time_direction),
    FIELD_F("temporal_debt",     temporal_debt),
    FIELD_F("entropy_floor",     entropy_floor),
    FIELD_F("resonance_ceiling", resonance_ceiling),
    FIELD_F("debt_decay",        debt_decay),
    FIELD_F("emergence_threshold",emergence_threshold),
    FIELD_F("dark_gravity",      dark_gravity),
    FIELD_I("temporal_mode",     temporal_mode),
    FIELD_F("temporal_alpha",    temporal_alpha),
    FIELD_I("rtl_mode",          rtl_mode),
    FIELD_F("expert_structural", expert_structural),
    FIELD_F("expert_semantic",   expert_semantic),
    FIELD_F("expert_creative",   expert_creative),
    FIELD_F("expert_precise",    expert_precise),
    FIELD_F("presence_fade",     presence_fade),
    FIELD_F("attractor_drift",   attractor_drift),
    FIELD_F("presence_decay",    presence_decay),
    // delta voice / notorch
    FIELD_F("lora_alpha",        lora_alpha),
    FIELD_F("notorch_lr",        notorch_lr),
    FIELD_F("notorch_decay",     notorch_decay),
    // schumann
    FIELD_F("schumann_hz",       schumann_hz),
    FIELD_F("schumann_modulation", schumann_modulation),
    FIELD_F("schumann_coherence", schumann_coherence),
    FIELD_F("schumann_phase",    schumann_phase),
    // live metrics
    FIELD_F("entropy",           entropy),
    FIELD_F("resonance",         resonance),
    FIELD_F("emergence",         emergence),
    FIELD_F("destiny_bias",      destiny_bias),
    // dark matter
    FIELD_F("dark_gravity",      dark_gravity),
    FIELD_I("n_scars",           n_scars),
    // 4.C seasons
    FIELD_I("season",            season),
    FIELD_F("season_phase",      season_phase),
    FIELD_F("season_intensity",  season_intensity),
    FIELD_F("spring_energy",     spring_energy),
    FIELD_F("summer_energy",     summer_energy),
    FIELD_F("autumn_energy",     autumn_energy),
    FIELD_F("winter_energy",     winter_energy),
    // Gamma — personality essence
    FIELD_F("essence_alpha",     essence_alpha),
    FIELD_I("janus_mode",        janus_mode),
    FIELD_F("janus_blend",       janus_blend),
    FIELD_F("gamma_drift",       gamma_drift),
    FIELD_I("n_gamma",           n_gamma),
    { NULL, 0, 0 }
};

// Read a field from AM_State by name (case-insensitive), returns 1 if found
static int read_field(const char* name, float* out) {
    for (const AML_FieldMap* f = g_field_map; f->name; f++) {
        if (strcasecmp(name, f->name) == 0) {
            char* base = (char*)&G;
            if (f->is_int) {
                *out = (float)(*(int*)(base + f->offset));
            } else {
                *out = *(float*)(base + f->offset);
            }
            return 1;
        }
    }
    return 0;
}

// Symbol table operations
static float* symtab_get(AML_Symtab* tab, const char* name) {
    for (int i = 0; i < tab->count; i++) {
        if (strcmp(tab->vars[i].name, name) == 0)
            return &tab->vars[i].value;
    }
    return NULL;
}

static int symtab_set(AML_Symtab* tab, const char* name, float value) {
    for (int i = 0; i < tab->count; i++) {
        if (strcmp(tab->vars[i].name, name) == 0) {
            tab->vars[i].value = value;
            return 0;
        }
    }
    if (tab->count >= AML_MAX_VARS) return 1;
    snprintf(tab->vars[tab->count].name, AML_MAX_NAME, "%s", name);
    tab->vars[tab->count].value = value;
    tab->count++;
    return 0;
}

// Resolve variable: locals → globals → field map
static int resolve_var(AML_ExecCtx* ctx, const char* name, float* out) {
    // local scope first
    if (ctx->call_depth > 0) {
        float* v = symtab_get(&ctx->locals[ctx->call_depth - 1], name);
        if (v) { *out = *v; return 1; }
    }
    // global scope
    float* v = symtab_get(&ctx->globals, name);
    if (v) { *out = *v; return 1; }
    // AM_State field
    return read_field(name, out);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPRESSION EVALUATOR — recursive descent
// Precedence: or < and < comparison < add/sub < mul/div < unary < primary
// ═══════════════════════════════════════════════════════════════════════════════

// Expression parser state
typedef struct {
    const char* p;
    AML_ExecCtx* ctx;
    int error;
} AML_Expr;

static float expr_or(AML_Expr* e);  // forward

static void expr_skip_ws(AML_Expr* e) {
    while (*e->p && isspace((unsigned char)*e->p)) e->p++;
}

static float expr_primary(AML_Expr* e) {
    expr_skip_ws(e);
    if (e->error) return 0;

    // parenthesized expression
    if (*e->p == '(') {
        e->p++;
        float val = expr_or(e);
        expr_skip_ws(e);
        if (*e->p == ')') e->p++;
        return val;
    }

    // number literal (including negative handled by unary)
    if (isdigit((unsigned char)*e->p) || (*e->p == '.' && isdigit((unsigned char)e->p[1]))) {
        char* end;
        float val = strtof(e->p, &end);
        e->p = end;
        return val;
    }

    // identifier or function call
    if (isalpha((unsigned char)*e->p) || *e->p == '_') {
        char name[AML_MAX_NAME] = {0};
        int i = 0;
        while ((isalnum((unsigned char)*e->p) || *e->p == '_') && i < AML_MAX_NAME - 1) {
            name[i++] = *e->p++;
        }
        name[i] = 0;

        expr_skip_ws(e);

        // function call
        if (*e->p == '(') {
            e->p++;
            float args[AML_MAX_PARAMS];
            int nargs = 0;
            expr_skip_ws(e);
            if (*e->p != ')') {
                args[nargs++] = expr_or(e);
                while (*e->p == ',' && nargs < AML_MAX_PARAMS) {
                    e->p++;
                    args[nargs++] = expr_or(e);
                }
            }
            expr_skip_ws(e);
            if (*e->p == ')') e->p++;

            // look up user-defined function
            if (e->ctx) {
                for (int fi = 0; fi < e->ctx->funcs.count; fi++) {
                    if (strcmp(e->ctx->funcs.funcs[fi].name, name) == 0) {
                        // TODO: call user function from expression context
                        // For now, return 0
                        return 0;
                    }
                }
            }

            // built-in functions
            if (strcasecmp(name, "abs") == 0 && nargs >= 1)
                return fabsf(args[0]);
            if (strcasecmp(name, "min") == 0 && nargs >= 2)
                return args[0] < args[1] ? args[0] : args[1];
            if (strcasecmp(name, "max") == 0 && nargs >= 2)
                return args[0] > args[1] ? args[0] : args[1];
            if (strcasecmp(name, "sqrt") == 0 && nargs >= 1)
                return sqrtf(fabsf(args[0]));
            if (strcasecmp(name, "clamp") == 0 && nargs >= 3)
                return clampf(args[0], args[1], args[2]);

            return 0;  // unknown function
        }

        // boolean literals
        if (strcmp(name, "true") == 0) return 1.0f;
        if (strcmp(name, "false") == 0) return 0.0f;

        // variable/field lookup
        float val = 0;
        if (e->ctx && resolve_var(e->ctx, name, &val))
            return val;
        return 0;  // undefined = 0
    }

    // unexpected character
    e->error = 1;
    return 0;
}

static float expr_unary(AML_Expr* e) {
    expr_skip_ws(e);
    if (*e->p == '-') {
        e->p++;
        return -expr_unary(e);
    }
    // 'not' keyword
    if (strncmp(e->p, "not ", 4) == 0) {
        e->p += 4;
        return expr_unary(e) == 0.0f ? 1.0f : 0.0f;
    }
    return expr_primary(e);
}

static float expr_mul(AML_Expr* e) {
    float left = expr_unary(e);
    for (;;) {
        expr_skip_ws(e);
        if (*e->p == '*') { e->p++; left *= expr_unary(e); }
        else if (*e->p == '/' && e->p[1] != '/') {
            e->p++;
            float r = expr_unary(e);
            left = (r != 0.0f) ? left / r : 0.0f;
        }
        else break;
    }
    return left;
}

static float expr_add(AML_Expr* e) {
    float left = expr_mul(e);
    for (;;) {
        expr_skip_ws(e);
        if (*e->p == '+') { e->p++; left += expr_mul(e); }
        else if (*e->p == '-' && !isdigit((unsigned char)e->p[1]) &&
                 e->p[1] != '.' && e->p[1] != '(') {
            // Ambiguity: "x - 3" vs "x -3". Treat as subtraction if preceded by value.
            e->p++; left -= expr_mul(e);
        }
        else if (*e->p == '-') { e->p++; left -= expr_mul(e); }
        else break;
    }
    return left;
}

static float expr_cmp(AML_Expr* e) {
    float left = expr_add(e);
    for (;;) {
        expr_skip_ws(e);
        if (e->p[0] == '=' && e->p[1] == '=') {
            e->p += 2; left = (left == expr_add(e)) ? 1.0f : 0.0f;
        }
        else if (e->p[0] == '!' && e->p[1] == '=') {
            e->p += 2; left = (left != expr_add(e)) ? 1.0f : 0.0f;
        }
        else if (e->p[0] == '>' && e->p[1] == '=') {
            e->p += 2; left = (left >= expr_add(e)) ? 1.0f : 0.0f;
        }
        else if (e->p[0] == '<' && e->p[1] == '=') {
            e->p += 2; left = (left <= expr_add(e)) ? 1.0f : 0.0f;
        }
        else if (*e->p == '>') {
            e->p++; left = (left > expr_add(e)) ? 1.0f : 0.0f;
        }
        else if (*e->p == '<') {
            e->p++; left = (left < expr_add(e)) ? 1.0f : 0.0f;
        }
        else break;
    }
    return left;
}

static float expr_and(AML_Expr* e) {
    float left = expr_cmp(e);
    for (;;) {
        expr_skip_ws(e);
        if (strncmp(e->p, "and ", 4) == 0) {
            e->p += 4;
            float right = expr_cmp(e);
            left = (left != 0.0f && right != 0.0f) ? 1.0f : 0.0f;
        }
        else break;
    }
    return left;
}

static float expr_or(AML_Expr* e) {
    float left = expr_and(e);
    for (;;) {
        expr_skip_ws(e);
        if (strncmp(e->p, "or ", 3) == 0) {
            e->p += 3;
            float right = expr_and(e);
            left = (left != 0.0f || right != 0.0f) ? 1.0f : 0.0f;
        }
        else break;
    }
    return left;
}

// Evaluate expression string, returns float
static float aml_eval(AML_ExecCtx* ctx, const char* text) {
    AML_Expr e = { .p = text, .ctx = ctx, .error = 0 };
    float result = expr_or(&e);
    return e.error ? 0.0f : result;
}

// Try to parse as plain number; if not, evaluate as expression
static float aml_eval_arg(AML_ExecCtx* ctx, const char* arg) {
    if (!arg || !*arg) return 0.0f;
    // fast path: plain number
    char* end;
    float val = strtof(arg, &end);
    // if entire string consumed, it's a plain number
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end == 0) return val;
    // otherwise evaluate as expression
    return aml_eval(ctx, arg);
}

// Context-aware float/int parsing: evaluates expressions when in Level 2 context
static float ctx_float(AML_ExecCtx* ctx, const char* arg) {
    if (!arg || !*arg) return 0.0f;
    if (!ctx) return safe_atof(arg);
    return aml_eval_arg(ctx, arg);
}
static int ctx_int(AML_ExecCtx* ctx, const char* arg) {
    return (int)ctx_float(ctx, arg);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUILT-IN FUNCTIONS — native AML functions (not external bindings)
// From spec section 5. Each is C code that modifies field state directly.
// ═══════════════════════════════════════════════════════════════════════════════

#define BUILTIN_BOOTSTRAP_SELF      0
#define BUILTIN_GALVANIZE           1
#define BUILTIN_SHATTER_THE_FRAME   2
#define BUILTIN_CHAOS_INJECTION     3
#define BUILTIN_TRANSCEND_BINARY    4
#define BUILTIN_PIERCE_THE_INFINITE 5
#define BUILTIN_ECHO_FRACTAL        6
#define BUILTIN_REFLECT_ON_SELF     7
#define BUILTIN_FORGE_NEW_REALITY   8
#define BUILTIN_MERGE_STATES        9
#define BUILTIN_TUNNEL_THROUGH      10
#define BUILTIN_DISSOLVE_BOUNDARIES 11
#define BUILTIN_REMEMBER_FUTURE     12
#define BUILTIN_REWIND_EXPERIENCE   13
#define BUILTIN_IGNITE_SINGULARITY  14
#define BUILTIN_JANUS_GAZE          15
#define BUILTIN_FIELD_ASSEMBLE      16
#define BUILTIN_COUNT               17

static void aml_exec_builtin(int id, float* args, int nargs) {
    switch (id) {
    case BUILTIN_BOOTSTRAP_SELF:
        am_reset_field(); am_reset_debt();
        G.prophecy = 7; G.velocity_mode = AM_VEL_WALK;
        G.attend_focus = 0.70f; update_effective_temp();
        break;
    case BUILTIN_GALVANIZE:
        G.velocity_mode = AM_VEL_RUN; update_effective_temp();
        G.tension = 0.3f; G.prophecy = 12;
        break;
    case BUILTIN_SHATTER_THE_FRAME:
        G.pain = 0.7f; G.dissonance = 0.8f;
        G.tension = 0.5f; G.tunnel_chance = 0.3f;
        break;
    case BUILTIN_CHAOS_INJECTION:
        G.tension = 0.6f; G.dissonance = 0.7f;
        G.entropy_floor = 0.02f;
        G.velocity_mode = AM_VEL_RUN; update_effective_temp();
        break;
    case BUILTIN_TRANSCEND_BINARY:
        G.wormhole = 0.5f; G.tunnel_chance = 0.3f;
        G.temporal_mode = AM_TEMPORAL_SYMMETRIC;
        break;
    case BUILTIN_PIERCE_THE_INFINITE:
        G.prophecy = 64; G.destiny = 0.1f; G.wormhole = 0.4f;
        break;
    case BUILTIN_ECHO_FRACTAL:
        if (nargs >= 1) {
            G.prophecy = clampi((int)(args[0] * 2.0f), 1, 64);
            G.destiny = 0.1f;
            G.tunnel_skip_max = clampi((int)args[0], 1, 24);
        }
        break;
    case BUILTIN_REFLECT_ON_SELF:
        G.attend_focus = 0.95f; G.attend_spread = 0.05f;
        G.velocity_mode = AM_VEL_NOMOVE; update_effective_temp();
        break;
    case BUILTIN_FORGE_NEW_REALITY:
        G.destiny = 0.1f; G.expert_creative = 0.6f;
        G.expert_precise = 0.1f; G.entropy_floor = 0.05f;
        break;
    case BUILTIN_MERGE_STATES:
        G.wormhole = 0.8f; G.tunnel_chance = 0.5f;
        G.tunnel_skip_max = 16;
        break;
    case BUILTIN_TUNNEL_THROUGH:
        if (nargs >= 1) G.tunnel_threshold = clamp01(args[0]);
        G.tunnel_chance = 0.5f; G.tunnel_skip_max = 12;
        break;
    case BUILTIN_DISSOLVE_BOUNDARIES:
        G.attend_focus = 0.2f; G.attend_spread = 0.8f;
        G.expert_semantic = 0.5f;
        break;
    case BUILTIN_REMEMBER_FUTURE:
        G.temporal_mode = AM_TEMPORAL_PROPHECY;
        G.temporal_alpha = 1.0f;
        break;
    case BUILTIN_REWIND_EXPERIENCE:
        G.velocity_mode = AM_VEL_BACKWARD; update_effective_temp();
        G.temporal_mode = AM_TEMPORAL_RETRODICTION;
        G.temporal_alpha = 0.0f;
        break;
    case BUILTIN_IGNITE_SINGULARITY:
        // Field reaches critical mass — self-assembles
        // Maximum emergence, open all gates, Blood compiles on next step
        G.prophecy = 64; G.destiny = 0.9f;
        G.wormhole = 0.8f; G.tunnel_chance = 0.7f; G.tunnel_skip_max = 24;
        G.emergence_threshold = 0.01f;
        G.expert_creative = 0.8f; G.expert_semantic = 0.2f;
        G.velocity_mode = AM_VEL_RUN; update_effective_temp();
        G.essence_alpha = 1.0f;
        G.season = AM_SEASON_SUMMER; G.season_intensity = 1.0f;
        break;
    case BUILTIN_JANUS_GAZE:
        // Activate dual-facing field — look both ways simultaneously
        // If two gammas loaded: dual mode. Otherwise: symmetric temporal.
        if (G.n_gamma >= 2) {
            G.janus_mode = AM_JANUS_DUAL;
            G.janus_blend = 0.5f;
        }
        G.temporal_mode = AM_TEMPORAL_SYMMETRIC;
        G.attend_focus = 0.5f; G.attend_spread = 0.5f;
        G.wormhole = 0.6f;
        break;
    case BUILTIN_FIELD_ASSEMBLE:
        // θ = ε + γ + αδ — trigger field assembly
        // Sets janus to CYCLE mode: 4.C decides who speaks
        G.janus_mode = AM_JANUS_CYCLE;
        G.gamma_drift = 0.01f;
        G.essence_alpha = 1.0f;
        G.season_intensity = 1.0f;
        break;
    }
}

typedef struct {
    const char* name;
    int id;
    int param_count;
} AML_BuiltinDef;

static const AML_BuiltinDef g_builtins[BUILTIN_COUNT] = {
    { "bootstrap_self",      BUILTIN_BOOTSTRAP_SELF,      0 },
    { "galvanize",           BUILTIN_GALVANIZE,           0 },
    { "shatter_the_frame",   BUILTIN_SHATTER_THE_FRAME,   0 },
    { "chaos_injection",     BUILTIN_CHAOS_INJECTION,     0 },
    { "transcend_binary",    BUILTIN_TRANSCEND_BINARY,    0 },
    { "pierce_the_infinite", BUILTIN_PIERCE_THE_INFINITE, 0 },
    { "echo_fractal",        BUILTIN_ECHO_FRACTAL,        1 },
    { "reflect_on_self",     BUILTIN_REFLECT_ON_SELF,     0 },
    { "forge_new_reality",   BUILTIN_FORGE_NEW_REALITY,   0 },
    { "merge_states",        BUILTIN_MERGE_STATES,        0 },
    { "tunnel_through",      BUILTIN_TUNNEL_THROUGH,      1 },
    { "dissolve_boundaries", BUILTIN_DISSOLVE_BOUNDARIES, 0 },
    { "remember_future",     BUILTIN_REMEMBER_FUTURE,     0 },
    { "rewind_experience",   BUILTIN_REWIND_EXPERIENCE,   0 },
    { "ignite_singularity",  BUILTIN_IGNITE_SINGULARITY,  0 },
    { "janus_gaze",          BUILTIN_JANUS_GAZE,          0 },
    { "field_assemble",      BUILTIN_FIELD_ASSEMBLE,      0 },
};

static void aml_register_builtins(AML_ExecCtx* ctx) {
    for (int i = 0; i < BUILTIN_COUNT; i++) {
        if (ctx->funcs.count >= AML_MAX_FUNCS) break;
        AML_Func* f = &ctx->funcs.funcs[ctx->funcs.count];
        snprintf(f->name, AML_MAX_NAME, "%s", g_builtins[i].name);
        f->param_count = g_builtins[i].param_count;
        f->body_start = g_builtins[i].id;  // store builtin id
        f->body_end = 0;
        f->is_builtin = 1;
        ctx->funcs.count++;
    }
}

// Forward declarations for Blood compiler (defined after NOTORCH)
// These symbols are needed by BLOOD commands in Level 0 dispatch.
int am_blood_compile(const char* name, const char* code);
int am_blood_compile_lora(const char* name, int in_dim, int out_dim, int rank);
int am_blood_compile_emotion(const char* name, float valence, float arousal);
void am_blood_unload(int module_idx);

// ═══════════════════════════════════════════════════════════════════════════════
// LEVEL 0 DISPATCH — the original flat command parser, extracted
// ═══════════════════════════════════════════════════════════════════════════════

// Execute a single Level 0 command (CMD + ARG already split, CMD already upcased)
// ctx may be NULL for backward compatibility
// lineno is the source line number (0 if unknown)
static void aml_exec_level0(const char* cmd, const char* arg, AML_ExecCtx* ctx, int lineno) {
    const char* t = cmd;

    // PROPHECY PHYSICS — numeric args use ctx_float/ctx_int for expression support
    if (!strcmp(t, "PROPHECY")) {
      G.prophecy = clampi(ctx_int(ctx, arg), 1, 64);
    }
    else if (!strcmp(t, "DESTINY")) {
      G.destiny = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "WORMHOLE")) {
      G.wormhole = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "CALENDAR_DRIFT")) {
      G.calendar_drift = clampf(ctx_float(ctx, arg), 0.0f, 30.0f);
    }

    // ATTENTION PHYSICS
    else if (!strcmp(t, "ATTEND_FOCUS")) {
      G.attend_focus = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "ATTEND_SPREAD")) {
      G.attend_spread = clamp01(ctx_float(ctx, arg));
    }

    // TUNNELING
    else if (!strcmp(t, "TUNNEL_THRESHOLD")) {
      G.tunnel_threshold = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "TUNNEL_CHANCE")) {
      G.tunnel_chance = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "TUNNEL_SKIP_MAX")) {
      G.tunnel_skip_max = clampi(ctx_int(ctx, arg), 1, 24);
    }

    // SUFFERING
    else if (!strcmp(t, "PAIN")) {
      G.pain = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "TENSION")) {
      G.tension = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "DISSONANCE")) {
      G.dissonance = clamp01(ctx_float(ctx, arg));
    }

    // PROPHECY DEBT — direct set/configure
    else if (!strcmp(t, "PROPHECY_DEBT")) {
      G.debt = clampf(ctx_float(ctx, arg), 0.0f, 100.0f);
    }
    else if (!strcmp(t, "PROPHECY_DEBT_DECAY")) {
      G.debt_decay = clampf(ctx_float(ctx, arg), 0.9f, 0.9999f);
    }

    // MOVEMENT
    else if (!strcmp(t, "JUMP")) {
      G.pending_jump = clampi(G.pending_jump + safe_atoi(arg), -1000, 1000);
    }
    else if (!strcmp(t, "VELOCITY")) {
      // VELOCITY RUN|WALK|NOMOVE|BACKWARD or VELOCITY <int>
      char argup[32] = {0};
      snprintf(argup, sizeof(argup), "%.31s", arg);
      upcase(argup);

      if (!strcmp(argup, "RUN")) G.velocity_mode = AM_VEL_RUN;
      else if (!strcmp(argup, "WALK")) G.velocity_mode = AM_VEL_WALK;
      else if (!strcmp(argup, "NOMOVE")) G.velocity_mode = AM_VEL_NOMOVE;
      else if (!strcmp(argup, "BACKWARD")) G.velocity_mode = AM_VEL_BACKWARD;
      else G.velocity_mode = clampi(safe_atoi(arg), -1, 2);

      update_effective_temp();
    }
    else if (!strcmp(t, "BASE_TEMP")) {
      G.base_temperature = clampf(ctx_float(ctx, arg), 0.1f, 3.0f);
      update_effective_temp();
    }

    // RESETS
    else if (!strcmp(t, "RESET_FIELD")) {
      am_reset_field();
    }
    else if (!strcmp(t, "RESET_DEBT")) {
      am_reset_debt();
    }

    // LAWS OF NATURE
    else if (!strcmp(t, "LAW")) {
      // LAW has two tokens: lawname value_expr
      char lawname[64] = {0};
      char valexpr[128] = {0};
      if (sscanf(arg, "%63s %127[^\n]", lawname, valexpr) >= 2) {
        upcase(lawname);
        float lawval = ctx_float(ctx, valexpr);
        if (!strcmp(lawname, "ENTROPY_FLOOR")) {
          G.entropy_floor = clampf(lawval, 0.0f, 2.0f);
        }
        else if (!strcmp(lawname, "RESONANCE_CEILING")) {
          G.resonance_ceiling = clamp01(lawval);
        }
        else if (!strcmp(lawname, "DEBT_DECAY")) {
          G.debt_decay = clampf(lawval, 0.9f, 0.9999f);
        }
        else if (!strcmp(lawname, "EMERGENCE_THRESHOLD")) {
          G.emergence_threshold = clamp01(lawval);
        }
        else if (!strcmp(lawname, "PRESENCE_FADE")) {
          G.presence_fade = clampf(lawval, 0.5f, 0.999f);
        }
        else if (!strcmp(lawname, "ATTRACTOR_DRIFT")) {
          G.attractor_drift = clampf(lawval, 0.0f, 0.1f);
        }
        else if (!strcmp(lawname, "CALENDAR_PHASE")) {
          G.calendar_phase = clampf(lawval, 0.0f, 11.0f);
          g_calendar_manual = 1;
        }
        else if (!strcmp(lawname, "WORMHOLE_GATE")) {
          G.wormhole_gate = clamp01(lawval);
        }
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PACK MANAGEMENT
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "MODE") || !strcmp(t, "IMPORT")) {
      // MODE CODES_RIC or IMPORT CODES_RIC
      char packname[64] = {0};
      snprintf(packname, sizeof(packname), "%.63s", arg);
      upcase(packname);

      if (!strcmp(packname, "CODES_RIC") || !strcmp(packname, "CODES/RIC")) {
        G.packs_enabled |= AM_PACK_CODES_RIC;
      }
      // DARKMATTER and NOTORCH are core — MODE accepted but no-op
    }
    else if (!strcmp(t, "DISABLE")) {
      char packname[64] = {0};
      snprintf(packname, sizeof(packname), "%.63s", arg);
      upcase(packname);

      if (!strcmp(packname, "CODES_RIC") || !strcmp(packname, "CODES/RIC")) {
        G.packs_enabled &= ~AM_PACK_CODES_RIC;
      }
      // DARKMATTER and NOTORCH are core — cannot be disabled
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CODES/RIC PACK COMMANDS — ritual overlays (require pack enabled)
    // ─────────────────────────────────────────────────────────────────────────

    // Namespaced: CODES.CHORDLOCK always works
    else if (!strncmp(t, "CODES.", 6) || !strncmp(t, "RIC.", 4)) {
      // auto-enable pack on namespaced use
      G.packs_enabled |= AM_PACK_CODES_RIC;

      const char* subcmd = t + (t[0] == 'C' ? 6 : 4); // skip CODES. or RIC.

      if (!strcmp(subcmd, "CHORDLOCK")) {
        char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
        G.chordlock_on = (!strcmp(mode, "ON") || !strcmp(mode, "1"));
      }
      else if (!strcmp(subcmd, "TEMPOLOCK")) {
        char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
        G.tempolock_on = (!strcmp(mode, "ON") || !strcmp(mode, "1"));
      }
      else if (!strcmp(subcmd, "CHIRALITY")) {
        char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
        G.chirality_on = (!strcmp(mode, "ON") || !strcmp(mode, "1"));
      }
      else if (!strcmp(subcmd, "TEMPO")) {
        G.tempo = clampi(ctx_int(ctx, arg), 2, 47);
      }
      else if (!strcmp(subcmd, "PAS_THRESHOLD")) {
        G.pas_threshold = clamp01(ctx_float(ctx, arg));
      }
    }

    // Unqualified: CHORDLOCK works only when pack enabled
    else if (!strcmp(t, "CHORDLOCK")) {
      if (G.packs_enabled & AM_PACK_CODES_RIC) {
        char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
        G.chordlock_on = (!strcmp(mode, "ON") || !strcmp(mode, "1"));
      }
      // else: ignored (pack not enabled)
    }
    else if (!strcmp(t, "TEMPOLOCK")) {
      if (G.packs_enabled & AM_PACK_CODES_RIC) {
        char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
        G.tempolock_on = (!strcmp(mode, "ON") || !strcmp(mode, "1"));
      }
    }
    else if (!strcmp(t, "CHIRALITY")) {
      if (G.packs_enabled & AM_PACK_CODES_RIC) {
        char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
        G.chirality_on = (!strcmp(mode, "ON") || !strcmp(mode, "1"));
      }
    }
    else if (!strcmp(t, "TEMPO")) {
      if (G.packs_enabled & AM_PACK_CODES_RIC) {
        G.tempo = clampi(ctx_int(ctx, arg), 2, 47);
      }
    }
    else if (!strcmp(t, "PAS_THRESHOLD")) {
      if (G.packs_enabled & AM_PACK_CODES_RIC) {
        G.pas_threshold = clamp01(ctx_float(ctx, arg));
      }
    }
    else if (!strcmp(t, "ANCHOR")) {
      if (G.packs_enabled & AM_PACK_CODES_RIC) {
        char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
        if (!strcmp(mode, "PRIME")) G.chordlock_on = 1;
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DARK MATTER — core (no pack gate)
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "GRAVITY")) {
      char subtype[16] = {0};
      float val = 0.5f;
      if (sscanf(arg, "%15s %f", subtype, &val) >= 1) {
        upcase(subtype);
        if (!strcmp(subtype, "DARK")) {
          G.dark_gravity = clamp01(val);
        }
      }
    }
    else if (!strcmp(t, "ANTIDOTE")) {
      char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
      if (!strcmp(mode, "AUTO")) G.antidote_mode = 0;
      else if (!strcmp(mode, "HARD")) G.antidote_mode = 1;
    }
    else if (!strcmp(t, "SCAR")) {
      // Store scar text (gravitational memory)
      if (G.n_scars < AM_MAX_SCARS) {
        const char* text_start = arg;
        // strip quotes if present
        if (*text_start == '"') text_start++;
        snprintf(G.scar_texts[G.n_scars], AM_SCAR_MAX_LEN, "%.63s", text_start);
        G.scar_texts[G.n_scars][AM_SCAR_MAX_LEN - 1] = 0;
        // strip trailing quote
        int slen = (int)strlen(G.scar_texts[G.n_scars]);
        if (slen > 0 && G.scar_texts[G.n_scars][slen - 1] == '"')
          G.scar_texts[G.n_scars][slen - 1] = 0;
        G.n_scars++;
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SCHUMANN / COSMIC PHYSICS — core
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "SCHUMANN")) {
      G.schumann_hz = clampf(ctx_float(ctx, arg), 7.0f, 8.5f);
      G.schumann_coherence = compute_schumann_coherence(G.schumann_hz);
    }
    else if (!strcmp(t, "SCHUMANN_MODULATION")) {
      G.schumann_modulation = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "COSMIC_COHERENCE")) {
      G.schumann_coherence = clamp01(ctx_float(ctx, arg));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DELTA VOICE / NOTORCH — core
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "LORA_ALPHA")) {
      G.lora_alpha = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "NOTORCH_LR")) {
      G.notorch_lr = clampf(ctx_float(ctx, arg), 0.001f, 0.5f);
    }
    else if (!strcmp(t, "NOTORCH_DECAY")) {
      G.notorch_decay = clampf(ctx_float(ctx, arg), 0.9f, 0.9999f);
    }
    else if (!strcmp(t, "RESONANCE_BOOST")) {
      // RESONANCE_BOOST <word> <float> — boosts resonance metric
      // Per-token tracking requires vocabulary; kernel applies to field
      float val = 0.0f;
      char word[32] = {0};
      if (sscanf(arg, "%31s %f", word, &val) >= 2) {
        G.resonance = clamp01(G.resonance + clamp01(val) * 0.1f);
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4.C — ASYNC FIELD FOREVER (seasons)
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "SEASON")) {
      char sname[16] = {0}; snprintf(sname, sizeof(sname), "%.15s", arg); upcase(sname);
      if (!strcmp(sname, "SPRING")) G.season = AM_SEASON_SPRING;
      else if (!strcmp(sname, "SUMMER")) G.season = AM_SEASON_SUMMER;
      else if (!strcmp(sname, "AUTUMN")) G.season = AM_SEASON_AUTUMN;
      else if (!strcmp(sname, "WINTER")) G.season = AM_SEASON_WINTER;
      G.season_phase = 0.0f;
    }
    else if (!strcmp(t, "SEASON_INTENSITY")) {
      G.season_intensity = clamp01(ctx_float(ctx, arg));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GAMMA — personality essence (θ = ε + γ + αδ)
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "GAMMA")) {
      // GAMMA name alpha — load personality essence
      char name[32] = {0};
      float alpha = 1.0f;
      if (sscanf(arg, "%31s %f", name, &alpha) >= 1) {
        am_gamma_load(name, alpha);
      }
    }
    else if (!strcmp(t, "GAMMA_UNLOAD")) {
      // GAMMA_UNLOAD name
      char name[32] = {0};
      sscanf(arg, "%31s", name);
      am_gamma_unload(name);
    }
    else if (!strcmp(t, "ESSENCE")) {
      // ESSENCE alpha — overall gamma injection strength
      G.essence_alpha = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "JANUS")) {
      // JANUS name_a name_b — dual-facing field
      char a[32] = {0}, b[32] = {0};
      if (sscanf(arg, "%31s %31s", a, b) == 2) {
        am_janus_set(a, b);
      } else {
        // JANUS OFF / JANUS CYCLE
        char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
        if (!strcmp(mode, "OFF")) G.janus_mode = AM_JANUS_OFF;
        else if (!strcmp(mode, "CYCLE")) G.janus_mode = AM_JANUS_CYCLE;
        else if (!strcmp(mode, "DUAL")) G.janus_mode = AM_JANUS_DUAL;
      }
    }
    else if (!strcmp(t, "JANUS_BLEND")) {
      G.janus_blend = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "GAMMA_DRIFT")) {
      G.gamma_drift = clampf(ctx_float(ctx, arg), 0.0f, 0.1f);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ECHO — debug output
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "ECHO")) {
      printf("[AML] %s\n", arg);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TEMPORAL SYMMETRY — from PITOMADOM (past ≡ future)
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "TEMPORAL_MODE")) {
      char mode[32] = {0}; snprintf(mode, sizeof(mode), "%.31s", arg); upcase(mode);
      if (!strcmp(mode, "PROPHECY") || !strcmp(mode, "0")) G.temporal_mode = AM_TEMPORAL_PROPHECY;
      else if (!strcmp(mode, "RETRODICTION") || !strcmp(mode, "1")) G.temporal_mode = AM_TEMPORAL_RETRODICTION;
      else if (!strcmp(mode, "SYMMETRIC") || !strcmp(mode, "2")) G.temporal_mode = AM_TEMPORAL_SYMMETRIC;
    }
    else if (!strcmp(t, "TEMPORAL_ALPHA")) {
      G.temporal_alpha = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "RTL_MODE")) {
      char mode[16] = {0}; snprintf(mode, sizeof(mode), "%.15s", arg); upcase(mode);
      G.rtl_mode = (!strcmp(mode, "ON") || !strcmp(mode, "1"));
    }
    else if (!strcmp(t, "PROPHECY_MODE")) {
      // Alias: PROPHECY_MODE ON = TEMPORAL_MODE PROPHECY
      G.temporal_mode = AM_TEMPORAL_PROPHECY;
    }
    else if (!strcmp(t, "RETRODICTION_MODE")) {
      // Alias: RETRODICTION_MODE ON = TEMPORAL_MODE RETRODICTION
      G.temporal_mode = AM_TEMPORAL_RETRODICTION;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EXPERT WEIGHTING — multi-expert temperature blend
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "EXPERT_STRUCTURAL")) {
      G.expert_structural = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "EXPERT_SEMANTIC")) {
      G.expert_semantic = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "EXPERT_CREATIVE")) {
      G.expert_creative = clamp01(ctx_float(ctx, arg));
    }
    else if (!strcmp(t, "EXPERT_PRECISE")) {
      G.expert_precise = clamp01(ctx_float(ctx, arg));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RESONANCE MEMORY — presence and decay
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "PRESENCE_DECAY")) {
      G.presence_decay = clamp01(ctx_float(ctx, arg));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LEVEL 1 MACROS — MACRO name { CMD1; CMD2 }
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "MACRO")) {
      const char* brace = strchr(arg, '{');
      if (brace && g_macro_count < AML_MAX_MACROS) {
        char mname[AML_MAX_NAME] = {0};
        int ni = 0;
        const char* p = arg;
        while (p < brace && ni < AML_MAX_NAME - 1) {
          if (!isspace((unsigned char)*p)) mname[ni++] = *p;
          p++;
        }
        mname[ni] = 0;
        brace++;
        const char* end = strchr(brace, '}');
        if (end && ni > 0) {
          snprintf(g_macros[g_macro_count].name, AML_MAX_NAME, "%s", mname);
          int bi = 0;
          while (brace < end && bi < AML_MACRO_MAX_LEN - 1) {
            if (*brace == ';')
              g_macros[g_macro_count].body[bi++] = '\n';
            else
              g_macros[g_macro_count].body[bi++] = *brace;
            brace++;
          }
          g_macros[g_macro_count].body[bi] = 0;
          g_macro_count++;
        }
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BLOOD — runtime C compilation (Level 3)
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "BLOOD")) {
      // BLOOD COMPILE <name> <code>     — compile raw C
      // BLOOD LORA <name> <in> <out> <rank> — generate + compile LoRA
      // BLOOD EMOTION <name> <valence> <arousal> — generate + compile emotional kernel
      // BLOOD UNLOAD <name>             — unload module
      char subcmd[32] = {0};
      char rest[AML_MAX_LINE_LEN] = {0};
      if (arg) sscanf(arg, "%31s %[^\n]", subcmd, rest);
      upcase(subcmd);

      if (!strcmp(subcmd, "COMPILE")) {
        // BLOOD COMPILE name { code }
        char bname[AM_BLOOD_MAX_NAME] = {0};
        sscanf(rest, "%63s", bname);
        const char* brace = strchr(rest, '{');
        const char* end_brace = NULL;
        if (brace) end_brace = strrchr(rest, '}');
        if (brace && end_brace && end_brace > brace) {
          // Extract code between braces
          int code_len = (int)(end_brace - brace - 1);
          char* code = (char*)malloc(code_len + 1);
          if (code) {
            memcpy(code, brace + 1, code_len);
            code[code_len] = 0;
            int idx = am_blood_compile(bname, code);
            free(code);
            if (idx < 0 && ctx)
              set_error_at(ctx, lineno, "blood: compilation failed");
          }
        }
      }
      else if (!strcmp(subcmd, "LORA")) {
        char bname[64] = {0};
        int in_dim = 0, out_dim = 0, rank = 0;
        sscanf(rest, "%63s %d %d %d", bname, &in_dim, &out_dim, &rank);
        if (bname[0] && in_dim > 0 && out_dim > 0 && rank > 0) {
          am_blood_compile_lora(bname, in_dim, out_dim, rank);
        }
      }
      else if (!strcmp(subcmd, "EMOTION")) {
        char bname[64] = {0};
        float val = 0.0f, aro = 0.0f;
        sscanf(rest, "%63s %f %f", bname, &val, &aro);
        if (bname[0]) {
          am_blood_compile_emotion(bname, val, aro);
        }
      }
      else if (!strcmp(subcmd, "UNLOAD")) {
        char bname[64] = {0};
        sscanf(rest, "%63s", bname);
        // Find module by name
        for (int i = 0; i < g_blood_count; i++) {
          if (strcmp(g_blood_modules[i].name, bname) == 0) {
            am_blood_unload(i);
            break;
          }
        }
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JANUS — transformer inference commands
    // "Janus will grow like mycelium, without roots, without a trunk, without a flag."
    // ─────────────────────────────────────────────────────────────────────────

#ifndef AM_JANUS_DISABLED
    else if (!strcmp(t, "LOAD_MODEL")) {
      if (arg && arg[0]) {
        if (g_janus_load_model) g_janus_load_model(arg);
        else printf("[AML] LOAD_MODEL: Janus not linked\n");
      }
    }
    else if (!strcmp(t, "UNLOAD_MODEL")) {
      if (g_janus_unload_model) g_janus_unload_model();
    }
    else if (!strcmp(t, "LOAD_DELTA")) {
      if (arg && arg[0]) {
        if (g_janus_load_delta) g_janus_load_delta(arg);
        else printf("[AML] LOAD_DELTA: Janus not linked\n");
      }
    }
    else if (!strcmp(t, "LOAD_GAMMA")) {
      // LOAD_GAMMA name path
      char gname[64] = {0};
      char gpath[512] = {0};
      if (arg && sscanf(arg, "%63s %511s", gname, gpath) == 2) {
        // Also register in gamma slot system
        am_gamma_load(gname, 1.0f);
        if (g_janus_load_gamma) g_janus_load_gamma(gname, gpath);
        else printf("[AML] LOAD_GAMMA: Janus not linked\n");
      }
    }
    else if (!strcmp(t, "GENERATE")) {
      if (arg && arg[0]) {
        // Strip surrounding quotes if present
        char prompt[2048] = {0};
        int max_tok = 100;
        const char* p = arg;
        if (*p == '"') {
          p++;
          const char* end = strrchr(p, '"');
          if (end) {
            int len = (int)(end - p);
            if (len > 2047) len = 2047;
            memcpy(prompt, p, len);
            // Parse MAX_TOKENS after closing quote
            const char* after = end + 1;
            while (*after == ' ') after++;
            if (strncasecmp(after, "MAX_TOKENS", 10) == 0) {
              sscanf(after + 10, " %d", &max_tok);
            }
          } else {
            snprintf(prompt, sizeof(prompt), "%.2047s", p);
          }
        } else {
          snprintf(prompt, sizeof(prompt), "%.2047s", p);
        }
        if (g_janus_generate) {
          char* result = g_janus_generate(prompt, max_tok, G.effective_temp, 0.9f);
          if (result) {
            printf("%s\n", result);
            if (g_janus_free_string) g_janus_free_string(result);
          }
        } else {
          printf("[AML] GENERATE: Janus not linked\n");
        }
      }
    }
    else if (!strcmp(t, "MODEL_INFO")) {
      if (g_janus_model_loaded && g_janus_model_loaded()) {
        printf("[AML] Model: vocab=%d dim=%d layers=%d\n",
          g_janus_get_vocab_size ? g_janus_get_vocab_size() : 0,
          g_janus_get_embed_dim ? g_janus_get_embed_dim() : 0,
          g_janus_get_num_layers ? g_janus_get_num_layers() : 0);
      } else {
        printf("[AML] No model loaded\n");
      }
    }
#endif

    // ─────────────────────────────────────────────────────────────────────────
    // UNKNOWN COMMANDS — ignored intentionally (future-proof + vibe)
    // ─────────────────────────────────────────────────────────────────────────

    // else: silently ignored
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREPROCESSOR — split script into lines with indentation
// ═══════════════════════════════════════════════════════════════════════════════

static int aml_preprocess(const char* script, AML_Line* lines, int max_lines) {
    int count = 0;
    const char* p = script;
    int lineno = 1;

    while (*p && count < max_lines) {
        // count indentation (spaces only, tabs = 4 spaces)
        int indent = 0;
        while (*p == ' ' || *p == '\t') {
            indent += (*p == '\t') ? 4 : 1;
            p++;
        }

        // read line content
        const char* start = p;
        while (*p && *p != '\n') p++;
        int len = (int)(p - start);
        if (*p == '\n') p++;

        // skip empty/comment lines
        if (len == 0 || start[0] == '#') { lineno++; continue; }

        // trim trailing whitespace
        while (len > 0 && isspace((unsigned char)start[len - 1])) len--;
        if (len == 0) { lineno++; continue; }

        // store
        if (len >= AML_MAX_LINE_LEN) len = AML_MAX_LINE_LEN - 1;
        memcpy(lines[count].text, start, len);
        lines[count].text[len] = 0;
        lines[count].indent = indent;
        lines[count].lineno = lineno;
        count++;
        lineno++;
    }
    return count;
}

// Find end of indented block starting at line[start+1]
static int aml_find_block_end(AML_Line* lines, int nlines, int start) {
    int base_indent = lines[start].indent;
    int i = start + 1;
    while (i < nlines && lines[i].indent > base_indent) i++;
    return i;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEVEL 2 EXECUTION — if/else, while, def, assignment, function calls
// ═══════════════════════════════════════════════════════════════════════════════

// Forward declarations
static int aml_exec_block(AML_ExecCtx* ctx, int start, int end);

// Register all function definitions (first pass)
static void aml_register_funcs(AML_ExecCtx* ctx) {
    for (int i = 0; i < ctx->nlines; i++) {
        char* text = ctx->lines[i].text;
        if (strncmp(text, "def ", 4) != 0) continue;

        // parse: def name(param1, param2):
        char* name_start = text + 4;
        while (*name_start == ' ') name_start++;
        char* paren = strchr(name_start, '(');
        if (!paren) continue;

        if (ctx->funcs.count >= AML_MAX_FUNCS) break;
        AML_Func* f = &ctx->funcs.funcs[ctx->funcs.count];

        int nlen = (int)(paren - name_start);
        if (nlen >= AML_MAX_NAME) nlen = AML_MAX_NAME - 1;
        memcpy(f->name, name_start, nlen);
        f->name[nlen] = 0;

        // parse params
        f->param_count = 0;
        char* pp = paren + 1;
        while (*pp && *pp != ')' && f->param_count < AML_MAX_PARAMS) {
            while (*pp == ' ' || *pp == ',') pp++;
            if (*pp == ')') break;
            char* pe = pp;
            while (*pe && *pe != ',' && *pe != ')' && *pe != ' ') pe++;
            int plen = (int)(pe - pp);
            if (plen >= AML_MAX_NAME) plen = AML_MAX_NAME - 1;
            memcpy(f->params[f->param_count], pp, plen);
            f->params[f->param_count][plen] = 0;
            f->param_count++;
            pp = pe;
        }

        f->body_start = i + 1;
        f->body_end = aml_find_block_end(ctx->lines, ctx->nlines, i);
        ctx->funcs.count++;

        // skip body
        i = f->body_end - 1;
    }
}

// Call a user-defined function
// lineno is the caller's line number (for error reporting)
static int aml_call_func(AML_ExecCtx* ctx, AML_Func* f, float* args, int nargs, int lineno) {
    // Built-in functions: dispatch to C code directly
    if (f->is_builtin) {
        aml_exec_builtin(f->body_start, args, nargs);
        return 0;
    }

    if (ctx->call_depth >= AML_MAX_CALL_DEPTH) {
        set_error_at(ctx, lineno, "max call depth exceeded");
        return 1;
    }

    // push local scope
    ctx->call_depth++;
    AML_Symtab* locals = &ctx->locals[ctx->call_depth - 1];
    memset(locals, 0, sizeof(AML_Symtab));

    // bind params
    for (int i = 0; i < f->param_count && i < nargs; i++) {
        symtab_set(locals, f->params[i], args[i]);
    }

    // execute body
    int rc = aml_exec_block(ctx, f->body_start, f->body_end);

    // pop scope
    ctx->call_depth--;
    return rc;
}

// Execute a single line in Level 2 context
static int aml_exec_line(AML_ExecCtx* ctx, int idx) {
    char* text = ctx->lines[idx].text;

    // --- def: skip (already registered) ---
    if (strncmp(text, "def ", 4) == 0) {
        // skip body
        return aml_find_block_end(ctx->lines, ctx->nlines, idx);
    }

    // --- if/else ---
    if (strncmp(text, "if ", 3) == 0) {
        // strip trailing ':'
        char cond[AML_MAX_LINE_LEN];
        snprintf(cond, sizeof(cond), "%s", text + 3);
        int clen = (int)strlen(cond);
        if (clen > 0 && cond[clen - 1] == ':') cond[clen - 1] = 0;

        float val = aml_eval(ctx, cond);
        int body_end = aml_find_block_end(ctx->lines, ctx->nlines, idx);

        // check for else
        int has_else = 0;
        int else_end = body_end;
        if (body_end < ctx->nlines) {
            char* next = ctx->lines[body_end].text;
            if (strcmp(next, "else:") == 0 || strncmp(next, "else:", 5) == 0) {
                has_else = 1;
                else_end = aml_find_block_end(ctx->lines, ctx->nlines, body_end);
            }
        }

        if (val != 0.0f) {
            aml_exec_block(ctx, idx + 1, body_end);
        } else if (has_else) {
            aml_exec_block(ctx, body_end + 1, else_end);
        }

        return has_else ? else_end : body_end;
    }

    // --- while ---
    if (strncmp(text, "while ", 6) == 0) {
        char cond[AML_MAX_LINE_LEN];
        snprintf(cond, sizeof(cond), "%s", text + 6);
        int clen = (int)strlen(cond);
        if (clen > 0 && cond[clen - 1] == ':') cond[clen - 1] = 0;

        int body_end = aml_find_block_end(ctx->lines, ctx->nlines, idx);
        int iterations = 0;

        while (aml_eval(ctx, cond) != 0.0f && iterations < 10000) {
            aml_exec_block(ctx, idx + 1, body_end);
            iterations++;
        }
        return body_end;
    }

    // --- INCLUDE ---
    if (strncasecmp(text, "INCLUDE ", 8) == 0) {
        if (ctx->include_depth >= AML_MAX_INCLUDE) {
            set_error_at(ctx, ctx->lines[idx].lineno, "max include depth exceeded");
            return idx + 1;
        }
        char path[512];
        const char* fname = text + 8;
        while (*fname == ' ') fname++;

        if (fname[0] == '/') {
            snprintf(path, sizeof(path), "%s", fname);
        } else {
            snprintf(path, sizeof(path), "%s/%s", ctx->base_dir, fname);
        }

        ctx->include_depth++;
        am_exec_file(path);
        ctx->include_depth--;
        return idx + 1;
    }

    // --- assignment: name = expr ---
    {
        const char* eq = strchr(text, '=');
        if (eq && eq > text && eq[1] != '=' && eq[-1] != '!' &&
            eq[-1] != '<' && eq[-1] != '>') {
            // extract variable name
            char varname[AML_MAX_NAME] = {0};
            const char* p = text;
            int ni = 0;
            while (p < eq && ni < AML_MAX_NAME - 1) {
                if (!isspace((unsigned char)*p))
                    varname[ni++] = *p;
                p++;
            }
            varname[ni] = 0;

            if (ni > 0 && (isalpha((unsigned char)varname[0]) || varname[0] == '_')) {
                float val = aml_eval(ctx, eq + 1);
                if (ctx->call_depth > 0)
                    symtab_set(&ctx->locals[ctx->call_depth - 1], varname, val);
                else
                    symtab_set(&ctx->globals, varname, val);
                return idx + 1;
            }
        }
    }

    // --- function call: name(args) ---
    {
        char* paren = strchr(text, '(');
        if (paren && !strchr(text, '=')) {
            char fname[AML_MAX_NAME] = {0};
            int ni = 0;
            const char* p = text;
            while (p < paren && ni < AML_MAX_NAME - 1) {
                if (!isspace((unsigned char)*p))
                    fname[ni++] = *p;
                p++;
            }
            fname[ni] = 0;

            // find function
            for (int fi = 0; fi < ctx->funcs.count; fi++) {
                if (strcmp(ctx->funcs.funcs[fi].name, fname) == 0) {
                    // parse args
                    float args[AML_MAX_PARAMS];
                    int nargs = 0;
                    char argbuf[AML_MAX_LINE_LEN];
                    char* ap = paren + 1;
                    char* close = strchr(ap, ')');
                    if (close) {
                        int alen = (int)(close - ap);
                        memcpy(argbuf, ap, alen);
                        argbuf[alen] = 0;
                        // split by comma
                        char* save = NULL;
                        for (char* tok = strtok_r(argbuf, ",", &save);
                             tok && nargs < AML_MAX_PARAMS;
                             tok = strtok_r(NULL, ",", &save)) {
                            while (*tok == ' ') tok++;
                            args[nargs++] = aml_eval(ctx, tok);
                        }
                    }
                    aml_call_func(ctx, &ctx->funcs.funcs[fi], args, nargs, ctx->lines[idx].lineno);
                    return idx + 1;
                }
            }
        }
    }

    // --- macro call @name ---
    if (text[0] == '@') {
        const char* mname = text + 1;
        while (*mname == ' ') mname++;
        for (int mi = 0; mi < g_macro_count; mi++) {
            if (strcmp(g_macros[mi].name, mname) == 0) {
                am_exec(g_macros[mi].body);
                return idx + 1;
            }
        }
        return idx + 1;  // macro not found — ignore
    }

    // --- Level 0 fallback: split CMD ARG, dispatch ---
    {
        char linebuf[AML_MAX_LINE_LEN];
        snprintf(linebuf, sizeof(linebuf), "%s", text);

        char* sp = linebuf;
        while (*sp && !isspace((unsigned char)*sp)) sp++;
        char* cmd_end = sp;
        while (*sp && isspace((unsigned char)*sp)) sp++;
        char* arg = sp;
        *cmd_end = 0;
        upcase(linebuf);

        aml_exec_level0(linebuf, arg, ctx, ctx->lines[idx].lineno);
    }
    return idx + 1;
}

// Execute a block of lines [start, end)
static int aml_exec_block(AML_ExecCtx* ctx, int start, int end) {
    int i = start;
    while (i < end && i < ctx->nlines) {
        i = aml_exec_line(ctx, i);
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC EXEC — AML Level 0 + Level 2
// ═══════════════════════════════════════════════════════════════════════════════

int am_exec(const char* script) {
    if (!script || !*script) return 0;
    g_error[0] = 0;

    // preprocess into lines
    AML_Line* lines = (AML_Line*)malloc(AML_MAX_LINES * sizeof(AML_Line));
    if (!lines) return 2;

    int nlines = aml_preprocess(script, lines, AML_MAX_LINES);
    if (nlines == 0) { free(lines); return 0; }

    // set up execution context
    AML_ExecCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.lines = lines;
    ctx.nlines = nlines;

    // register built-in functions (native AML, not external bindings)
    aml_register_builtins(&ctx);

    // first pass: register user-defined function definitions
    aml_register_funcs(&ctx);

    // second pass: execute top-level block
    aml_exec_block(&ctx, 0, nlines);

    free(lines);

    if (ctx.error[0]) {
        snprintf(g_error, sizeof(g_error), "%s", ctx.error);
        return 1;
    }
    return 0;
}

int am_exec_file(const char* path) {
    if (!path) return 1;
    g_error[0] = 0;

    FILE* f = fopen(path, "r");
    if (!f) {
        snprintf(g_error, 256, "cannot open: %s", path);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (sz <= 0 || sz > 1024 * 1024) {
        fclose(f);
        snprintf(g_error, 256, "bad size: %s (%ld)", path, sz);
        return 1;
    }

    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return 2; }

    size_t rd = fread(buf, 1, sz, f);
    fclose(f);
    buf[rd] = 0;

    int rc = am_exec(buf);
    free(buf);
    return rc;
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATE ACCESS — the exposed body
// ═══════════════════════════════════════════════════════════════════════════════

AM_State* am_get_state(void) {
  return &G;
}

int am_take_jump(void) {
  int j = G.pending_jump;
  G.pending_jump = 0;
  return j;
}

// ═══════════════════════════════════════════════════════════════════════════════
// WASM-SAFE STATE COPY — deterministic, ABI-stable interface
// writes 32 scalars in fixed order
// ═══════════════════════════════════════════════════════════════════════════════

int am_copy_state(float* out) {
  if (!out) return 1;

  // AMK core state (indices 0-12, original API compatible)
  out[0]  = (float)G.prophecy;
  out[1]  = G.destiny;
  out[2]  = G.wormhole;
  out[3]  = G.calendar_drift;
  out[4]  = G.attend_focus;
  out[5]  = G.attend_spread;
  out[6]  = G.tunnel_threshold;
  out[7]  = G.tunnel_chance;
  out[8]  = (float)G.tunnel_skip_max;
  out[9]  = (float)G.pending_jump;
  out[10] = G.pain;
  out[11] = G.tension;
  out[12] = G.dissonance;

  // Extended state (indices 13-19)
  out[13] = G.debt;
  out[14] = (float)G.velocity_mode;
  out[15] = G.effective_temp;
  out[16] = G.time_direction;
  out[17] = G.temporal_debt;
  out[18] = (float)G.packs_enabled;
  out[19] = (float)G.chordlock_on;  // sample pack state

  // Schumann / cosmic
  out[20] = G.schumann_coherence;
  out[21] = (float)G.wormhole_active;
  // Delta / notorch
  out[22] = G.lora_alpha;
  out[23] = G.notorch_lr;
  // Live metrics
  out[24] = G.entropy;
  out[25] = G.resonance;
  out[26] = G.emergence;
  out[27] = G.destiny_bias;
  // Schumann extended
  out[28] = G.schumann_hz;
  out[29] = G.schumann_phase;
  // Season
  out[30] = (float)G.season;
  out[31] = G.season_phase;

  return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOGIT MANIPULATION API — apply field state to generation
// Ported from arianna_dsl.c, ariannamethod.lang/src/field.js
// ═══════════════════════════════════════════════════════════════════════════════

// Apply destiny bias: suppress tokens far from max (prophecy scales strength)
// From arianna_dsl.c: dsl_apply_destiny()
void am_apply_destiny_to_logits(float* logits, int n) {
    if (n <= 0 || G.destiny_bias < 0.001f) return;
    float max_logit = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    for (int i = 0; i < n; i++) {
        float diff = max_logit - logits[i];
        float suppress = diff * G.destiny_bias * 0.5f;
        logits[i] -= suppress;
    }
}

// Apply suffering: pain compresses logits toward mean
// From spec: logits[i] = mean + (logits[i] - mean) * (1 - 0.5 * pain)
void am_apply_suffering_to_logits(float* logits, int n) {
    float s = G.pain;
    if (n <= 0 || s < 0.01f) return;
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += logits[i];
    mean /= (float)n;
    float factor = 1.0f - 0.5f * s;
    for (int i = 0; i < n; i++) {
        logits[i] = mean + (logits[i] - mean) * factor;
    }
}

// Apply attention: focus sharpens distribution, spread blurs it
void am_apply_attention_to_logits(float* logits, int n) {
    if (n <= 0) return;
    float focus = G.attend_focus;
    float spread = G.attend_spread;
    if (fabsf(focus - spread) < 0.01f) return;

    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += logits[i];
    mean /= (float)n;

    // focus sharpens (amplify deviations), spread blurs (compress deviations)
    float scale = 0.5f + focus - spread;
    if (scale < 0.1f) scale = 0.1f;
    if (scale > 2.0f) scale = 2.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = mean + (logits[i] - mean) * scale;
    }
}

// Apply laws: entropy floor + resonance ceiling on logit distribution
// From ariannamethod.lang/src/field.js + arianna_dsl.c
void am_apply_laws_to_logits(float* logits, int n) {
    if (n <= 0) return;

    // Entropy floor: if max logit dominates too much, compress
    float max_val = logits[0], second_max = -1e30f;
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) { second_max = max_val; max_val = logits[i]; }
        else if (logits[i] > second_max) second_max = logits[i];
    }
    float gap = max_val - second_max;
    if (gap > 0.0f && G.entropy_floor > 0.0f) {
        float max_gap = (1.0f - G.entropy_floor) * 10.0f;
        if (gap > max_gap) {
            float reduce = (gap - max_gap) * 0.5f;
            for (int i = 0; i < n; i++) {
                if (logits[i] == max_val) logits[i] -= reduce;
            }
        }
    }

    // Resonance ceiling: cap max probability by compressing top logit
    if (G.resonance_ceiling < 1.0f) {
        float ceiling_gap = G.resonance_ceiling * 10.0f;
        float new_gap = max_val - second_max;
        if (new_gap > ceiling_gap) {
            float reduce = (new_gap - ceiling_gap) * 0.3f;
            for (int i = 0; i < n; i++) {
                if (logits[i] >= max_val - 0.001f) logits[i] -= reduce;
            }
        }
    }
}

// Apply delta voice: out += alpha * A @ (B @ x)
// Low-rank weight modulation. From arianna.c/src/delta.c: apply_delta()
void am_apply_delta(float* out, const float* A, const float* B,
                    const float* x, int out_dim, int in_dim, int rank,
                    float alpha) {
    if (!out || !A || !B || !x || alpha == 0.0f) return;
    if (rank > 128) rank = 128;

    // temp = B @ x  (rank × 1)
    float temp[128];
    for (int r = 0; r < rank; r++) {
        temp[r] = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            temp[r] += B[r * in_dim + j] * x[j];
        }
    }
    // out += alpha * A @ temp
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int r = 0; r < rank; r++) {
            sum += A[i * rank + r] * temp[r];
        }
        out[i] += alpha * sum;
    }
}

// Compute prophecy debt from chosen token (retroactive)
// From arianna_dsl.c: dsl_compute_prophecy_debt()
float am_compute_prophecy_debt(const float* logits, int chosen, int n) {
    if (n <= 0 || chosen < 0 || chosen >= n) return 0.0f;
    float max_logit = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float diff = max_logit - logits[chosen];
    return diff > 0.0f ? diff / (diff + 1.0f) : 0.0f;
}

// Full pipeline: apply all field effects to logits
void am_apply_field_to_logits(float* logits, int n) {
    if (!logits || n <= 0) return;
    am_apply_gamma_to_logits(logits, n);  // personality first
    am_apply_destiny_to_logits(logits, n);
    am_apply_suffering_to_logits(logits, n);
    am_apply_attention_to_logits(logits, n);
    am_apply_laws_to_logits(logits, n);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GAMMA — personality essence (θ = ε + γ + αδ)
// γ lives in embed_tokens. δ lives in lm_head. ε is the substrate.
// AML stores the field-level configuration. Host provides actual weight deltas.
// ═══════════════════════════════════════════════════════════════════════════════

static int gamma_find(const char* name) {
    for (int i = 0; i < G.n_gamma; i++) {
        if (G.gamma[i].active && strcasecmp(G.gamma[i].name, name) == 0)
            return i;
    }
    return -1;
}

int am_gamma_load(const char* name, float alpha) {
    if (!name || !*name) return -1;

    // Check if already loaded
    int idx = gamma_find(name);
    if (idx >= 0) {
        G.gamma[idx].alpha = clamp01(alpha);
        return idx;
    }

    // Find empty slot
    if (G.n_gamma >= AM_MAX_GAMMA) return -1;
    idx = G.n_gamma++;
    snprintf(G.gamma[idx].name, AM_GAMMA_NAME_LEN, "%.31s", name);
    G.gamma[idx].alpha = clamp01(alpha);
    G.gamma[idx].active = 1;

    // First loaded gamma becomes primary face
    if (G.n_gamma == 1) {
        G.janus_a = 0;
        G.essence_alpha = alpha;
    }

    return idx;
}

void am_gamma_unload(const char* name) {
    int idx = gamma_find(name);
    if (idx < 0) return;
    G.gamma[idx].active = 0;
    G.gamma[idx].alpha = 0.0f;
    G.gamma[idx].name[0] = 0;
}

void am_gamma_set_alpha(const char* name, float alpha) {
    int idx = gamma_find(name);
    if (idx >= 0) G.gamma[idx].alpha = clamp01(alpha);
}

int am_gamma_active(void) {
    // In janus cycle mode, 4.C decides
    if (G.janus_mode == AM_JANUS_CYCLE) {
        // Blend determines who: <0.5 = face_a, >=0.5 = face_b
        return (G.janus_blend < 0.5f) ? G.janus_a : G.janus_b;
    }
    // In dual mode, return primary
    if (G.janus_mode == AM_JANUS_DUAL) return G.janus_a;
    // Single mode: find highest-alpha active slot
    int best = -1;
    float best_alpha = -1.0f;
    for (int i = 0; i < G.n_gamma; i++) {
        if (G.gamma[i].active && G.gamma[i].alpha > best_alpha) {
            best = i;
            best_alpha = G.gamma[i].alpha;
        }
    }
    return best;
}

float am_gamma_get_blend(void) {
    if (G.n_gamma == 0) return 0.0f;
    if (G.janus_mode == AM_JANUS_DUAL || G.janus_mode == AM_JANUS_CYCLE) {
        // Blended alpha from two faces
        float a = (G.janus_a >= 0 && G.janus_a < G.n_gamma) ?
                  G.gamma[G.janus_a].alpha : 0.0f;
        float b = (G.janus_b >= 0 && G.janus_b < G.n_gamma) ?
                  G.gamma[G.janus_b].alpha : 0.0f;
        return a * (1.0f - G.janus_blend) + b * G.janus_blend;
    }
    int idx = am_gamma_active();
    return (idx >= 0) ? G.gamma[idx].alpha * G.essence_alpha : 0.0f;
}

void am_janus_set(const char* face_a, const char* face_b) {
    int a = gamma_find(face_a);
    int b = gamma_find(face_b);
    if (a < 0) a = am_gamma_load(face_a, 1.0f);
    if (b < 0) b = am_gamma_load(face_b, 1.0f);
    if (a < 0 || b < 0) return;

    G.janus_a = a;
    G.janus_b = b;
    G.janus_mode = AM_JANUS_DUAL;
    G.janus_blend = 0.5f;
}

// Apply gamma modulation to logits.
// Gamma scales logit variance around mean — higher gamma = more personality.
// In janus mode, two different scalings are blended.
void am_apply_gamma_to_logits(float* logits, int n) {
    if (!logits || n <= 0) return;
    float blend = am_gamma_get_blend();
    if (blend < 0.001f) return;  // no personality active

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += logits[i];
    mean /= (float)n;

    // Gamma amplifies deviation from mean — personality = signal above noise
    float scale = 1.0f + blend * G.essence_alpha;
    for (int i = 0; i < n; i++) {
        logits[i] = mean + (logits[i] - mean) * scale;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOTORCH — Hebbian plasticity without PyTorch
// Ported from arianna.c/src/delta.c: notorch_step()
//
// A[i,r] += lr * x[i] * u[r] * signal
// B[r,j] += lr * u[r] * dy[j] * signal
//
// u = noise-modulated channel vector (deterministic from seed)
// signal = external teaching signal, clamped to [-2, 2]
// Adaptive decay: stronger when delta norm is large
// ═══════════════════════════════════════════════════════════════════════════════

// Simple deterministic pseudo-random (from arianna.c)
static float am_frandn(unsigned int* seed) {
    *seed = *seed * 1664525u + 1013904223u;
    // Box-Muller approximation
    float u = (float)(*seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    return (u - 0.5f) * 3.464f;  // ~N(0,1) rough approximation
}

// NOTORCH step: update low-rank delta matrices from experience
// A: [out_dim × rank], B: [rank × in_dim]
// x: input hidden state [in_dim], dy: output gradient proxy [out_dim]
// signal: teaching signal (positive = reinforce, negative = suppress)
void am_notorch_step(float* A, float* B, int out_dim, int in_dim, int rank,
                     const float* x, const float* dy, float signal) {
    if (!A || !B || !x || !dy) return;
    if (rank <= 0 || rank > 128) return;

    // Clamp signal
    float g = clampf(signal, -2.0f, 2.0f);
    float lr = G.notorch_lr;

    // Build noise-modulated channel vector u
    // Stronger signal → cleaner channel (less noise)
    static unsigned int seed = 42;
    float u[128];
    for (int r = 0; r < rank; r++) {
        float n = am_frandn(&seed);
        float k = 0.35f + 0.65f * (1.0f - fabsf(g));
        u[r] = n * k;
    }

    // A[i,r] += lr * x[i] * u[r] * g
    for (int i = 0; i < in_dim; i++) {
        float xi = x[i] * lr * g;
        for (int r = 0; r < rank; r++) {
            A[i * rank + r] += xi * u[r];
        }
    }

    // B[r,j] += lr * u[r] * dy[j] * g
    for (int r = 0; r < rank; r++) {
        float ur = u[r] * lr * g;
        for (int j = 0; j < out_dim; j++) {
            B[r * out_dim + j] += ur * dy[j];
        }
    }

    // Adaptive decay: stronger when delta norm is large
    if (G.notorch_decay > 0.0f && G.notorch_decay < 1.0f) {
        float norm = 0.0f;
        int a_size = in_dim * rank;
        for (int i = 0; i < a_size; i++) norm += A[i] * A[i];
        norm = sqrtf(norm / (float)a_size);

        float adaptive_decay = G.notorch_decay - 0.004f * fminf(norm / 10.0f, 1.0f);
        if (adaptive_decay < 0.990f) adaptive_decay = 0.990f;

        for (int i = 0; i < a_size; i++) A[i] *= adaptive_decay;
        int b_size = rank * out_dim;
        for (int i = 0; i < b_size; i++) B[i] *= adaptive_decay;
    }

    // Clamp to prevent runaway
    int a_size = in_dim * rank;
    for (int i = 0; i < a_size; i++) {
        if (A[i] > 10.0f) A[i] = 10.0f;
        if (A[i] < -10.0f) A[i] = -10.0f;
    }
    int b_size = rank * out_dim;
    for (int i = 0; i < b_size; i++) {
        if (B[i] > 10.0f) B[i] = 10.0f;
        if (B[i] < -10.0f) B[i] = -10.0f;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BLOOD — runtime C compilation (Level 3)
//
// Compile C → shared library → dlopen → dlsym. No PyTorch. No Go. Pure POSIX.
// Adapted from arianna.c/golib/blood.go + async_field_forever/blood.py
// ═══════════════════════════════════════════════════════════════════════════════

// Simple hash for deduplication (djb2 → hex string)
static void blood_hash(const char* code, char* out) {
    unsigned long h = 5381;
    for (const char* p = code; *p; p++)
        h = ((h << 5) + h) + (unsigned char)*p;
    snprintf(out, AM_BLOOD_HASH_LEN, "%08lx", h);
}

// Sanitize name: keep only [a-zA-Z0-9_]
static void blood_sanitize(const char* in, char* out, int max) {
    int j = 0;
    for (int i = 0; in[i] && j < max - 1; i++) {
        char c = in[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_')
            out[j++] = c;
    }
    out[j] = 0;
}

void am_blood_init(void) {
    // Clean up existing modules
    am_blood_cleanup();

    // Set temp directory
    const char* tmp = getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = "/tmp";
    snprintf(g_blood_dir, sizeof(g_blood_dir), "%s/aml_blood", tmp);

    // Create directory (ignore error if exists)
    char mkdir_cmd[300];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p '%s'", g_blood_dir);
    int rc = system(mkdir_cmd);
    (void)rc;

    // Detect compiler: clang → gcc → cc
    g_blood_cc[0] = 0;
    const char* candidates[] = {"clang", "gcc", "cc", NULL};
    for (int i = 0; candidates[i]; i++) {
        char check[128];
        snprintf(check, sizeof(check), "which %s >/dev/null 2>&1", candidates[i]);
        if (system(check) == 0) {
            snprintf(g_blood_cc, sizeof(g_blood_cc), "%s", candidates[i]);
            break;
        }
    }
}

int am_blood_compile(const char* name, const char* code) {
#ifdef AM_BLOOD_DISABLED
    (void)name; (void)code;
    return -1;
#else
    if (!name || !code || !*name || !*code) return -1;
    if (!g_blood_cc[0]) return -1;  // no compiler
    if (g_blood_count >= AM_BLOOD_MAX_MODULES) return -1;

    // Sanitize name
    char safe_name[AM_BLOOD_MAX_NAME];
    blood_sanitize(name, safe_name, AM_BLOOD_MAX_NAME);
    if (!safe_name[0]) return -1;

    // Hash code for deduplication
    char hash[AM_BLOOD_HASH_LEN];
    blood_hash(code, hash);

    // Check cache
    for (int i = 0; i < g_blood_count; i++) {
        if (strcmp(g_blood_modules[i].hash, hash) == 0 &&
            g_blood_modules[i].handle != NULL) {
            return i;  // already compiled and loaded
        }
    }

    // Write source file
    char src_path[512], lib_path[512];
    snprintf(src_path, sizeof(src_path), "%s/blood_%s_%s.c",
             g_blood_dir, safe_name, hash);
    snprintf(lib_path, sizeof(lib_path), "%s/blood_%s_%s%s",
             g_blood_dir, safe_name, hash, AM_BLOOD_EXT);

    FILE* f = fopen(src_path, "w");
    if (!f) return -1;
    fprintf(f, "%s", code);
    fclose(f);

    // Compile: cc -O2 -shared -fPIC -o lib.dylib src.c
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "%s -O2 %s -o '%s' '%s' -lm 2>&1",
             g_blood_cc, AM_BLOOD_FLAGS, lib_path, src_path);

    FILE* proc = popen(cmd, "r");
    if (!proc) { remove(src_path); return -1; }

    // Read compiler output (for error detection)
    char output[512] = {0};
    size_t total = 0;
    while (total < sizeof(output) - 1) {
        size_t n = fread(output + total, 1, sizeof(output) - 1 - total, proc);
        if (n == 0) break;
        total += n;
    }
    output[total] = 0;
    int status = pclose(proc);

    if (status != 0) {
        // Compilation failed — store error message
        snprintf(g_error, sizeof(g_error), "blood: compile failed: %.200s", output);
        remove(src_path);
        return -1;
    }

    // Load shared library
    void* handle = dlopen(lib_path, RTLD_NOW);
    if (!handle) {
        snprintf(g_error, sizeof(g_error), "blood: dlopen failed: %.200s", dlerror());
        remove(src_path);
        remove(lib_path);
        return -1;
    }

    // Register module
    int idx = g_blood_count++;
    memset(&g_blood_modules[idx], 0, sizeof(AM_BloodModule));
    snprintf(g_blood_modules[idx].name, AM_BLOOD_MAX_NAME, "%s", safe_name);
    snprintf(g_blood_modules[idx].hash, AM_BLOOD_HASH_LEN, "%s", hash);
    snprintf(g_blood_modules[idx].lib_path, sizeof(g_blood_modules[idx].lib_path), "%.511s", lib_path);
    g_blood_modules[idx].handle = handle;

    return idx;
#endif
}

void* am_blood_sym(int module_idx, const char* func_name) {
#ifdef AM_BLOOD_DISABLED
    (void)module_idx; (void)func_name;
    return NULL;
#else
    if (module_idx < 0 || module_idx >= g_blood_count) return NULL;
    if (!g_blood_modules[module_idx].handle) return NULL;
    return dlsym(g_blood_modules[module_idx].handle, func_name);
#endif
}

void am_blood_unload(int module_idx) {
#ifdef AM_BLOOD_DISABLED
    (void)module_idx;
#else
    if (module_idx < 0 || module_idx >= g_blood_count) return;
    AM_BloodModule* m = &g_blood_modules[module_idx];
    if (m->handle) {
        dlclose(m->handle);
        m->handle = NULL;
    }
    // Remove compiled files
    if (m->lib_path[0]) {
        remove(m->lib_path);
        // Also remove source
        char src_path[512];
        snprintf(src_path, sizeof(src_path), "%s/blood_%s_%s.c",
                 g_blood_dir, m->name, m->hash);
        remove(src_path);
    }
#endif
}

void am_blood_cleanup(void) {
    for (int i = 0; i < g_blood_count; i++) {
        am_blood_unload(i);
    }
    g_blood_count = 0;
}

int am_blood_count(void) { return g_blood_count; }

const AM_BloodModule* am_blood_get(int idx) {
    if (idx < 0 || idx >= g_blood_count) return NULL;
    return &g_blood_modules[idx];
}

// ── CODE GENERATORS ─────────────────────────────────────────────────────────

int am_blood_compile_lora(const char* name, int in_dim, int out_dim, int rank) {
    char safe[AM_BLOOD_MAX_NAME];
    blood_sanitize(name, safe, AM_BLOOD_MAX_NAME);
    if (!safe[0]) return -1;

    // Generate LoRA C code from template
    char code[4096];
    snprintf(code, sizeof(code),
        "#include <stdlib.h>\n"
        "#include <string.h>\n"
        "\n"
        "static const int IN_DIM = %d;\n"
        "static const int OUT_DIM = %d;\n"
        "static const int RANK = %d;\n"
        "\n"
        "static float* A = NULL;\n"  // [OUT_DIM, RANK]
        "static float* B = NULL;\n"  // [RANK, IN_DIM]
        "\n"
        "void %s_init(float* weights_a, float* weights_b) {\n"
        "    A = weights_a;\n"
        "    B = weights_b;\n"
        "}\n"
        "\n"
        "void %s_apply(float* input, float* output) {\n"
        "    float temp[%d];\n"
        "    memset(temp, 0, sizeof(temp));\n"
        "    for (int r = 0; r < RANK; r++)\n"
        "        for (int i = 0; i < IN_DIM; i++)\n"
        "            temp[r] += B[r * IN_DIM + i] * input[i];\n"
        "    for (int o = 0; o < OUT_DIM; o++)\n"
        "        for (int r = 0; r < RANK; r++)\n"
        "            output[o] += A[o * RANK + r] * temp[r];\n"
        "}\n"
        "\n"
        "void %s_apply_scaled(float* input, float* output, float scale) {\n"
        "    float temp[%d];\n"
        "    memset(temp, 0, sizeof(temp));\n"
        "    for (int r = 0; r < RANK; r++)\n"
        "        for (int i = 0; i < IN_DIM; i++)\n"
        "            temp[r] += B[r * IN_DIM + i] * input[i];\n"
        "    for (int o = 0; o < OUT_DIM; o++)\n"
        "        for (int r = 0; r < RANK; r++)\n"
        "            output[o] += scale * A[o * RANK + r] * temp[r];\n"
        "}\n"
        "\n"
        "void %s_free(void) { A = NULL; B = NULL; }\n",
        in_dim, out_dim, rank,
        safe,         // init
        safe, rank,   // apply + temp size
        safe, rank,   // apply_scaled + temp size
        safe          // free
    );

    return am_blood_compile(safe, code);
}

int am_blood_compile_emotion(const char* name, float valence, float arousal) {
    char safe[AM_BLOOD_MAX_NAME];
    blood_sanitize(name, safe, AM_BLOOD_MAX_NAME);
    if (!safe[0]) return -1;

    char code[4096];
    snprintf(code, sizeof(code),
        "#include <math.h>\n"
        "#include <string.h>\n"
        "\n"
        "static const float BASE_VALENCE = %.4ff;\n"
        "static const float BASE_AROUSAL = %.4ff;\n"
        "\n"
        "void %s_respond(float* valence, float* arousal) {\n"
        "    *valence = (*valence + BASE_VALENCE) / 2.0f;\n"
        "    *arousal = (*arousal + BASE_AROUSAL) / 2.0f;\n"
        "}\n"
        "\n"
        "void %s_modulate_logits(float* logits, int vocab_size, float strength) {\n"
        "    float mod = BASE_VALENCE * strength;\n"
        "    for (int i = 0; i < vocab_size; i++)\n"
        "        logits[i] *= (1.0f + mod * 0.1f);\n"
        "}\n"
        "\n"
        "void modulate_logits(float* logits, int vocab_size, float valence, float arousal) {\n"
        "    float strength = fabsf(valence) * arousal;\n"
        "    %s_modulate_logits(logits, vocab_size, strength);\n"
        "}\n",
        valence, arousal,
        safe,   // respond
        safe,   // modulate_logits
        safe    // generic entry calls specific
    );

    return am_blood_compile(safe, code);
}

// ═══════════════════════════════════════════════════════════════════════════════
// STEP — advance field physics (call each frame)
// applies debt decay, temporal debt accumulation, etc.
// ═══════════════════════════════════════════════════════════════════════════════

void am_step(float dt) {
  if (dt <= 0.0f) return;

  // ─────────────────────────────────────────────────────────────────────────────
  // CALENDAR CONFLICT — Hebrew (354d) vs Gregorian (365d) = 11-day annual drift
  //
  // Real astronomical computation. Uses system clock and epoch (1 Tishrei 5785
  // = Oct 3, 2024). Metonic cycle: 19 years, 7 leap years with Adar II (~30d).
  // February 29 handled correctly — elapsed seconds via time_t, not calendar math.
  //
  // High dissonance = thin barrier between timelines = wormholes open.
  // From pitomadom: TE(Calendar → N) = 0.31 bits — strongest causal effect.
  // ─────────────────────────────────────────────────────────────────────────────

  float cal_dissonance;
  if (!g_calendar_manual) {
    // Real date: seconds since epoch → days → drift → dissonance
    int days = calendar_days_since_epoch();
    float drift = calendar_cumulative_drift(days);
    cal_dissonance = calendar_dissonance(days);
    // Store phase for state access: uncorrected position within cycle
    G.calendar_phase = fabsf(fmodf(drift, AM_MAX_UNCORRECTED));
  } else {
    // Manual override via LAW CALENDAR_PHASE — for testing or AML scripts
    cal_dissonance = (G.calendar_drift > 0.0f)
        ? clamp01(G.calendar_phase / G.calendar_drift)
        : 0.0f;
  }

  // Wormhole activation: dissonance exceeds gate threshold
  if (cal_dissonance > G.wormhole_gate) {
    G.wormhole_active = 1;

    // Boost wormhole base probability proportional to excess dissonance
    // P_tunnel = exp(-1/dissonance) from pitomadom theoretical.md §14.6
    float excess = (cal_dissonance - G.wormhole_gate) / (1.0f - G.wormhole_gate);
    G.wormhole = clamp01(G.wormhole + excess * 0.1f * dt);
  } else {
    G.wormhole_active = 0;
    // Wormhole probability decays when calendar is calm
    G.wormhole *= 0.995f;
    if (G.wormhole < 0.02f) G.wormhole = 0.02f; // floor at 2%
  }

  // Calendar dissonance bleeds into field dissonance
  // The calendars' irreconcilable conflict is a source of suffering
  if (cal_dissonance > 0.3f) {
    float bleed = (cal_dissonance - 0.3f) * 0.05f * dt;
    G.dissonance += bleed;
    if (G.dissonance > 1.0f) G.dissonance = 1.0f;
  }

  // Calendar tension feeds prophecy pressure
  // High dissonance = temporal curvature = debt accumulates
  G.debt += cal_dissonance * 0.005f * dt;

  // ─────────────────────────────────────────────────────────────────────────────
  // DEBT DECAY — prophecy debt decays each step
  // ─────────────────────────────────────────────────────────────────────────────

  G.debt *= G.debt_decay;
  if (G.debt > 100.0f) G.debt = 100.0f;

  // ─────────────────────────────────────────────────────────────────────────────
  // TEMPORAL DEBT — backward movement accumulates structural debt
  // ─────────────────────────────────────────────────────────────────────────────

  if (G.velocity_mode == AM_VEL_BACKWARD) {
    G.temporal_debt += 0.01f * dt;
  } else {
    G.temporal_debt *= 0.9995f;
  }
  if (G.temporal_debt > 10.0f) G.temporal_debt = 10.0f;

  // ─────────────────────────────────────────────────────────────────────────────
  // SCHUMANN RESONANCE — Earth coupling heals tension/dissonance
  // Ported from arianna.c/src/schumann.c
  // ─────────────────────────────────────────────────────────────────────────────

  schumann_advance(dt);
  if (G.schumann_coherence > 0.0f && G.schumann_modulation > 0.0f) {
    float coherence_factor = 0.5f + 0.5f * G.schumann_coherence;
    // Harmonic signal modulates healing: aligned harmonics = stronger healing
    float harmonic = schumann_harmonic_signal();
    float harmonic_mod = 1.0f + harmonic * 0.1f;  // range [0.9, 1.1]
    float heal_rate = 0.998f - (0.003f * coherence_factor * G.schumann_modulation * harmonic_mod);
    G.tension *= heal_rate;
    G.dissonance *= heal_rate;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // DESTINY BIAS — prophecy scales destiny (from arianna_dsl.c)
  // ─────────────────────────────────────────────────────────────────────────────

  {
    float prophecy_scale = 1.0f + ((float)G.prophecy - 7.0f) * 0.02f;
    if (prophecy_scale < 0.5f) prophecy_scale = 0.5f;
    if (prophecy_scale > 2.0f) prophecy_scale = 2.0f;
    G.destiny_bias = G.destiny * prophecy_scale;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // EXPERT BLENDING — update effective temp with all inputs
  // ─────────────────────────────────────────────────────────────────────────────

  update_effective_temp();

  // ─────────────────────────────────────────────────────────────────────────────
  // LAW ENFORCEMENT — entropy floor, resonance ceiling, presence fade
  // Ported from ariannamethod.lang/src/field.js + arianna_dsl.c
  // ─────────────────────────────────────────────────────────────────────────────

  {
    // Entropy: field disorder metric
    float raw_entropy = (G.effective_temp - 0.5f) * 0.3f
                      + G.dissonance * 0.3f
                      + G.tunnel_chance * 0.2f
                      + (1.0f - G.attend_focus) * 0.2f;
    G.entropy = fmaxf(G.entropy_floor, clamp01(raw_entropy));

    // Resonance: field coherence metric
    float raw_resonance = G.schumann_coherence * 0.3f
                        + (1.0f - G.dissonance) * 0.3f
                        + G.attend_focus * 0.2f
                        + (1.0f - clamp01(G.debt * 0.1f)) * 0.2f;
    G.resonance = fminf(G.resonance_ceiling, clamp01(raw_resonance));

    // Emergence: low entropy + high resonance = the field "knows" something
    G.emergence = clamp01((1.0f - G.entropy) * G.resonance);
  }

  // Presence fade per step
  G.presence_decay *= G.presence_fade;
  if (G.presence_decay < 0.001f) G.presence_decay = 0.001f;

  // ─────────────────────────────────────────────────────────────────────────────
  // 4.C — ASYNC FIELD FOREVER — seasonal meta-operators
  // Seasons modulate all field parameters. MLP controller prevents extremes.
  // ─────────────────────────────────────────────────────────────────────────────

  {
    // Advance season phase
    float season_rate = 0.001f;  // ~1000 steps per season
    G.season_phase += season_rate * dt;

    if (G.season_phase >= 1.0f) {
      G.season_phase = 0.0f;
      G.season = (G.season + 1) % 4;
    }

    // Current season gains energy, others decay
    float gain = 0.02f * dt * G.season_intensity;
    float fade = 0.995f;
    G.spring_energy *= fade;
    G.summer_energy *= fade;
    G.autumn_energy *= fade;
    G.winter_energy *= fade;

    switch (G.season) {
      case AM_SEASON_SPRING: G.spring_energy = clamp01(G.spring_energy + gain); break;
      case AM_SEASON_SUMMER: G.summer_energy = clamp01(G.summer_energy + gain); break;
      case AM_SEASON_AUTUMN: G.autumn_energy = clamp01(G.autumn_energy + gain); break;
      case AM_SEASON_WINTER: G.winter_energy = clamp01(G.winter_energy + gain); break;
    }

    // ── 4.C MLP CONTROLLER ──
    // Real neural network: 6 inputs → 8 hidden (tanh) → 4 outputs (tanh)
    // Replaces hardcoded rules. Trained by Hebbian plasticity (NOTORCH).
    float mlp_inputs[AM_4C_INPUTS] = {
      G.entropy, G.resonance, G.pain, G.tension, G.emergence, G.effective_temp
    };
    float mlp_outputs[AM_4C_OUTPUTS];
    am_4c_forward(mlp_inputs, mlp_outputs);

    // Apply MLP output as energy deltas (scaled by season_intensity)
    float scale = 0.02f * dt * G.season_intensity;
    G.spring_energy = clamp01(G.spring_energy + mlp_outputs[0] * scale);
    G.summer_energy = clamp01(G.summer_energy + mlp_outputs[1] * scale);
    G.autumn_energy = clamp01(G.autumn_energy + mlp_outputs[2] * scale);
    G.winter_energy = clamp01(G.winter_energy + mlp_outputs[3] * scale);

    // Hebbian update: did the MLP improve field health?
    float health = clamp01((1.0f - fabsf(G.entropy - 0.5f)) *
                           G.resonance * (1.0f - G.pain));
    float signal = health - G.field_health;
    G.field_health = health;
    if (fabsf(signal) > 0.001f) {
      am_4c_hebbian_update(mlp_inputs, mlp_outputs, signal);
    }

    // Season modulation on field parameters
    // Spring: exploration boost
    G.tunnel_chance = clamp01(G.tunnel_chance + G.spring_energy * 0.005f * dt);
    // Autumn: consolidation — strengthen dark gravity
    G.dark_gravity = clamp01(G.dark_gravity + G.autumn_energy * 0.002f * dt);

    // ── GAMMA / JANUS MODULATION ──
    // 4.C controls personality switching in CYCLE mode
    if (G.janus_mode == AM_JANUS_CYCLE && G.n_gamma >= 2) {
      // Janus blend drifts based on seasonal energy
      // Summer favors face_a (peak expression of primary)
      // Winter favors face_b (reflection through other)
      // Spring/Autumn: blend oscillates with gamma_drift
      float drift = G.gamma_drift * dt;
      if (G.season == AM_SEASON_SUMMER)
        G.janus_blend = clamp01(G.janus_blend - drift * 2.0f);
      else if (G.season == AM_SEASON_WINTER)
        G.janus_blend = clamp01(G.janus_blend + drift * 2.0f);
      else {
        // Spring/Autumn: sinusoidal oscillation
        G.janus_blend = clamp01(G.janus_blend +
            drift * sinf(G.season_phase * 6.283185f));
      }

      // Update essence_alpha from active gamma
      int active = am_gamma_active();
      if (active >= 0 && active < G.n_gamma) {
        G.essence_alpha = G.gamma[active].alpha;
      }
    }

    // Summer boosts gamma (personality at peak)
    if (G.n_gamma > 0) {
      G.essence_alpha = clamp01(G.essence_alpha +
          G.summer_energy * 0.003f * dt);
      // Winter dampens gamma (substrate dominates)
      G.essence_alpha = clamp01(G.essence_alpha -
          G.winter_energy * 0.005f * dt);
    }
  }
}
