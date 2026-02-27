// ariannamethod.c — AML: Arianna Method Language
// Reference implementation. THE KERNEL: movement IS language.
//
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Source of truth: github.com/ariannamethod/ariannamethod.ai
// Embed (copy) into your project or link.
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
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef AM_BLOOD_DISABLED
#include <dlfcn.h>   // for dlopen, dlsym, dlclose (Blood compiler)
#endif

// Lilith I/O — named pipes (FIFO) for data infrastructure
#ifndef AM_IO_DISABLED
#include <fcntl.h>     // for open(), O_RDONLY, O_WRONLY, O_NONBLOCK
#include <unistd.h>    // for read(), write(), close(), unlink()
#include <sys/stat.h>  // for mkfifo()
#include <errno.h>     // for EAGAIN, ENXIO
#endif

// Async — pthreads for SPAWN/AWAIT
#ifndef AM_ASYNC_DISABLED
#include <pthread.h>
#endif

// Platform detection for Blood compiler
#ifdef __APPLE__
  #define AM_BLOOD_EXT ".dylib"
  #define AM_BLOOD_FLAGS "-dynamiclib -fPIC"
#else
  #define AM_BLOOD_EXT ".so"
  #define AM_BLOOD_FLAGS "-shared -fPIC"
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// BLAS ACCELERATION — optional hardware-accelerated matmul for Delta Voice
// and NOTORCH Hebbian plasticity.
//
// Compile with -DUSE_BLAS to enable:
//   macOS:  -DUSE_BLAS -DACCELERATE -framework Accelerate  (Apple AMX/Neural Engine)
//   Linux:  -DUSE_BLAS -lopenblas                           (OpenBLAS)
//
// Without USE_BLAS: pure scalar C loops (portable, correct, slower on large dims)
// With USE_BLAS: cblas_sgemv for delta, cblas_sger for notorch
//
// Evolved in molequla (github.com/ariannamethod/molequla), ported back to core.
// ═══════════════════════════════════════════════════════════════════════════════
#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

// See ariannamethod.h for struct definitions and pack flags

static AM_State G;

// Blood compiler globals (used by Level 0 dispatch + Blood API)
static AM_BloodModule g_blood_modules[AM_BLOOD_MAX_MODULES];
static int g_blood_count = 0;
static char g_blood_dir[256] = {0};
static char g_blood_cc[64] = {0};

// Lilith I/O globals (named pipes for data infrastructure)
#ifndef AM_IO_DISABLED
static AM_Pipe g_pipes[AM_MAX_PIPES];
static int g_pipe_count = 0;
static float g_pipe_last_value = 0.0f;
static char g_pipe_read_buf[AM_PIPE_BUF_SIZE] = {0};
#endif

// Async — SPAWN/AWAIT/CHANNEL globals
#ifndef AM_ASYNC_DISABLED
static AM_SpawnSlot   g_spawns[AM_MAX_SPAWNS];
static int            g_spawn_count = 0;
static pthread_t      g_spawn_threads[AM_MAX_SPAWNS];
static pthread_mutex_t g_spawn_mutex = PTHREAD_MUTEX_INITIALIZER;

static AM_ChannelSlot g_channels[AM_MAX_CHANNELS];
static int            g_channel_count = 0;
static pthread_mutex_t g_channel_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_channel_cond = PTHREAD_COND_INITIALIZER;
#endif

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

// Forward declarations for tape cleanup (defined after tape section)
static void am_tape_destroy(void);

// Forward declarations for async cleanup (defined after async section)
#ifndef AM_ASYNC_DISABLED
static void am_spawn_reset(void);
static void am_channel_reset(void);
#endif

// Forward declarations for persistent globals (defined after am_init)
static int g_persistent_enabled;
static AML_Symtab g_persistent_globals;
void am_persistent_clear(void);

void am_init(void) {
  // Clean up tape from previous session
  am_tape_destroy();

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

  // persistent globals — clear on full init
  am_persistent_clear();
  g_persistent_enabled = 0;

  // lilith I/O
#ifndef AM_IO_DISABLED
  am_pipe_close_all();
  g_pipe_last_value = 0.0f;
  g_pipe_read_buf[0] = 0;
#endif

  // async — SPAWN/AWAIT/CHANNEL
#ifndef AM_ASYNC_DISABLED
  am_spawn_await_all();   // join any running threads first
  am_spawn_reset();
  am_channel_reset();
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTENT GLOBALS — survive across am_exec() calls
// ═══════════════════════════════════════════════════════════════════════════════

// Forward declarations for symtab functions (defined later in file)
static float*   symtab_get(AML_Symtab* tab, const char* name);
static AML_Var* symtab_get_var(AML_Symtab* tab, const char* name);
static int      symtab_set(AML_Symtab* tab, const char* name, float value);
static int      symtab_set_array(AML_Symtab* tab, const char* name, AM_Array* arr);

void am_persistent_mode(int enable) {
    if (!enable && g_persistent_enabled) {
        // Turning off — free all persistent arrays
        am_persistent_clear();
    }
    g_persistent_enabled = enable;
}

void am_persistent_clear(void) {
    for (int i = 0; i < g_persistent_globals.count; i++) {
        if (g_persistent_globals.vars[i].type == AML_TYPE_ARRAY &&
            g_persistent_globals.vars[i].array) {
            am_array_free(g_persistent_globals.vars[i].array);
            g_persistent_globals.vars[i].array = NULL;
        }
    }
    g_persistent_globals.count = 0;
}

// Restore persistent globals into execution context
static void persistent_restore(AML_Symtab* dst) {
    if (!g_persistent_enabled) return;
    for (int i = 0; i < g_persistent_globals.count; i++) {
        AML_Var* pv = &g_persistent_globals.vars[i];
        if (pv->type == AML_TYPE_ARRAY && pv->array) {
            // Clone array: exec ctx gets its own copy, persistent keeps original
            AM_Array* clone = am_array_new(pv->array->len);
            if (clone) {
                memcpy(clone->data, pv->array->data,
                       pv->array->len * sizeof(float));
                clone->rows = pv->array->rows;
                clone->cols = pv->array->cols;
                symtab_set_array(dst, pv->name, clone);
            }
        } else {
            symtab_set(dst, pv->name, pv->value);
        }
    }
}

// Save execution context globals back to persistent storage.
// Two-phase approach:
//   Phase 1: Update variables that already exist in persistent
//   Phase 2: Add NEW variables that aren't in persistent yet
// This means the first am_exec creates persistent vars, and subsequent calls
// update them. Intermediates created by later scripts get saved once (unavoidable)
// but since they have fixed names, the count stabilizes.
static void persistent_save(AML_Symtab* src) {
    static int save_call_count = 0;
    save_call_count++;
    if (!g_persistent_enabled) return;

    // Phase 1: Update existing persistent variables from exec ctx
    for (int i = 0; i < g_persistent_globals.count; i++) {
        AML_Var* pv = &g_persistent_globals.vars[i];
        AML_Var* sv = NULL;
        for (int j = 0; j < src->count; j++) {
            if (strcmp(src->vars[j].name, pv->name) == 0) {
                sv = &src->vars[j];
                break;
            }
        }
        if (!sv) continue;

        if (sv->type == AML_TYPE_ARRAY && sv->array) {
            AM_Array* clone = am_array_new(sv->array->len);
            if (clone) {
                memcpy(clone->data, sv->array->data,
                       sv->array->len * sizeof(float));
                clone->rows = sv->array->rows;
                clone->cols = sv->array->cols;
                if (pv->type == AML_TYPE_ARRAY && pv->array) {
                    am_array_free(pv->array);
                }
                pv->type = AML_TYPE_ARRAY;
                pv->array = clone;
                pv->value = 0;
            }
        } else {
            if (pv->type == AML_TYPE_ARRAY && pv->array) {
                am_array_free(pv->array);
                pv->array = NULL;
            }
            pv->type = AML_TYPE_FLOAT;
            pv->value = sv->value;
        }
    }

    // Phase 2: Add NEW variables from exec ctx that aren't in persistent yet
    for (int j = 0; j < src->count; j++) {
        AML_Var* sv = &src->vars[j];
        int found = 0;
        for (int i = 0; i < g_persistent_globals.count; i++) {
            if (strcmp(g_persistent_globals.vars[i].name, sv->name) == 0) {
                found = 1;
                break;
            }
        }
        if (found) continue;

        if (sv->type == AML_TYPE_ARRAY && sv->array) {
            AM_Array* clone = am_array_new(sv->array->len);
            if (clone) {
                memcpy(clone->data, sv->array->data,
                       sv->array->len * sizeof(float));
                clone->rows = sv->array->rows;
                clone->cols = sv->array->cols;
                symtab_set_array(&g_persistent_globals, sv->name, clone);
            }
        } else {
            symtab_set(&g_persistent_globals, sv->name, sv->value);
        }
    // DIAGNOSTIC: print persistent storage stats every 50 calls
    if (save_call_count % 50 == 0) {
        long total_bytes = 0;
        int n_arrays = 0;
        for (int i = 0; i < g_persistent_globals.count; i++) {
            if (g_persistent_globals.vars[i].type == AML_TYPE_ARRAY && 
                g_persistent_globals.vars[i].array) {
                total_bytes += g_persistent_globals.vars[i].array->len * sizeof(float);
                n_arrays++;
            }
        }
        fprintf(stderr, "[DIAG] persistent_save #%d: %d vars (%d arrays), %ld bytes (%.1f MB), src: %d vars\n",
            save_call_count, g_persistent_globals.count, n_arrays, total_bytes, total_bytes / 1048576.0, src->count);
    }
    }
}

int am_set_var_array(const char* name, const float* data, int len) {
    if (!name || !data || len <= 0 || len > AM_MAX_ARRAY_SIZE) return 1;
    // Force persistent mode on
    g_persistent_enabled = 1;
    AM_Array* arr = am_array_new(len);
    if (!arr) return 2;
    memcpy(arr->data, data, len * sizeof(float));
    return symtab_set_array(&g_persistent_globals, name, arr);
}

int am_set_var_matrix(const char* name, const float* data, int rows, int cols) {
    if (!name || !data || rows <= 0 || cols <= 0) return 1;
    int len = rows * cols;
    if (len > AM_MAX_ARRAY_SIZE) return 1;
    g_persistent_enabled = 1;
    AM_Array* arr = am_array_new(len);
    if (!arr) return 2;
    memcpy(arr->data, data, len * sizeof(float));
    arr->rows = rows;
    arr->cols = cols;
    return symtab_set_array(&g_persistent_globals, name, arr);
}

const float* am_get_var_array(const char* name, int* len) {
    if (!name) return NULL;
    AML_Var* v = symtab_get_var(&g_persistent_globals, name);
    if (!v || v->type != AML_TYPE_ARRAY || !v->array) return NULL;
    if (len) *len = v->array->len;
    return v->array->data;
}

float am_get_var_float(const char* name) {
    if (!name) return 0.0f;
    AML_Var* v = symtab_get_var(&g_persistent_globals, name);
    if (!v) return 0.0f;
    if (v->type == AML_TYPE_FLOAT) return v->value;
    // Array with 1 element → treat as scalar
    if (v->type == AML_TYPE_ARRAY && v->array && v->array->len >= 1)
        return v->array->data[0];
    return 0.0f;
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

// ═══════════════════════════════════════════════════════════════════════════════
// ARRAY MEMORY MANAGEMENT (v4.0)
// ═══════════════════════════════════════════════════════════════════════════════

AM_Array* am_array_new(int len) {
    if (len <= 0 || len > AM_MAX_ARRAY_SIZE) return NULL;
    AM_Array* arr = (AM_Array*)malloc(sizeof(AM_Array));
    if (!arr) return NULL;
    arr->data = (float*)calloc(len, sizeof(float));
    if (!arr->data) { free(arr); return NULL; }
    arr->len = len;
    arr->refcount = 1;
    arr->rows = 0;
    arr->cols = 0;
    return arr;
}

// Create a 2D matrix (flat array with shape tracking)
static AM_Array* am_matrix_new(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    int total = rows * cols;
    if (total > AM_MAX_ARRAY_SIZE) return NULL;
    AM_Array* arr = am_array_new(total);
    if (!arr) return NULL;
    arr->rows = rows;
    arr->cols = cols;
    return arr;
}

void am_array_free(AM_Array* arr) {
    if (!arr) return;
    arr->refcount--;
    if (arr->refcount <= 0) {
        free(arr->data);
        free(arr);
    }
}

AM_Array* am_array_ref(AM_Array* arr) {
    if (arr) arr->refcount++;
    return arr;
}

// Clone an array (deep copy, preserves shape)
static AM_Array* am_array_clone(const AM_Array* src) {
    if (!src) return NULL;
    AM_Array* dst = am_array_new(src->len);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, src->len * sizeof(float));
    dst->rows = src->rows;
    dst->cols = src->cols;
    return dst;
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOGRAD TAPE (v4.0 Phase 3) — reverse-mode automatic differentiation
// ═══════════════════════════════════════════════════════════════════════════════

static AM_Tape g_tape = {0};

void am_tape_start(void) {
    // Clear any existing tape state
    am_tape_clear();
    g_tape.active = 1;
}

void am_tape_clear(void) {
    // Free non-param outputs and all grads
    for (int i = 0; i < g_tape.count; i++) {
        if (g_tape.entries[i].output) {  // Free ALL outputs including params (refcount handles safety)
            am_array_free(g_tape.entries[i].output);
        }
        if (g_tape.entries[i].grad) {
            am_array_free(g_tape.entries[i].grad);
            g_tape.entries[i].grad = NULL;
        }
    }
    g_tape.count = 0;
    g_tape.n_params = 0;  // Reset param count so adam states get reused (not leaked)
    g_tape.active = 0;
    // NOTE: adam states and n_params survive clear — they persist across steps
}

// Full tape reset — frees ALL resources including params and adam states
static void am_tape_destroy(void) {
    // Free all tape entries including params
    for (int i = 0; i < g_tape.count; i++) {
        if (g_tape.entries[i].output) {
            am_array_free(g_tape.entries[i].output);
            g_tape.entries[i].output = NULL;
        }
        if (g_tape.entries[i].grad) {
            am_array_free(g_tape.entries[i].grad);
            g_tape.entries[i].grad = NULL;
        }
    }
    // Free adam states
    for (int i = 0; i < g_tape.n_params; i++) {
        if (g_tape.adam[i].m) { am_array_free(g_tape.adam[i].m); g_tape.adam[i].m = NULL; }
        if (g_tape.adam[i].v) { am_array_free(g_tape.adam[i].v); g_tape.adam[i].v = NULL; }
        g_tape.adam[i].t = 0;
    }
    memset(&g_tape, 0, sizeof(g_tape));
}

int am_tape_is_active(void) { return g_tape.active; }
AM_Tape* am_tape_get(void) { return &g_tape; }

// Record a computation on the tape. Returns entry index.
int am_tape_record(AM_Array* output, int op, int p1, int p2, float aux) {
    if (!g_tape.active || g_tape.count >= AM_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    AM_TapeEntry* e = &g_tape.entries[idx];
    e->output = output;
    am_array_ref(output); // tape owns a reference
    e->grad = NULL;
    e->op = op;
    e->parent1 = p1;
    e->parent2 = p2;
    e->parent3 = -1;
    e->aux = aux;
    e->aux2 = 0;
    e->is_param = 0;
    g_tape.count++;
    return idx;
}

// Record with 3 parents + 2 aux values (for seq ops like causal_attention)
int am_tape_record3(AM_Array* output, int op, int p1, int p2, int p3, float aux, float aux2) {
    if (!g_tape.active || g_tape.count >= AM_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    AM_TapeEntry* e = &g_tape.entries[idx];
    e->output = output;
    am_array_ref(output);
    e->grad = NULL;
    e->op = op;
    e->parent1 = p1;
    e->parent2 = p2;
    e->parent3 = p3;
    e->aux = aux;
    e->aux2 = aux2;
    e->is_param = 0;
    g_tape.count++;
    return idx;
}

// Record a parameter (trainable weight). Returns entry index.
// Adam state is allocated on first call for this param.
int am_tape_record_param(AM_Array* param) {
    if (!g_tape.active || g_tape.count >= AM_TAPE_MAX_ENTRIES) return -1;
    int idx = g_tape.count;
    AM_TapeEntry* e = &g_tape.entries[idx];
    e->output = param;
    am_array_ref(param);
    e->grad = NULL;
    e->op = AM_OP_NONE;
    e->parent1 = -1;
    e->parent2 = -1;
    e->parent3 = -1;
    e->aux = 0;
    e->aux2 = 0;
    e->is_param = 1;

    // Register for Adam if not already registered
    // Use pointer comparison to find existing registration
    int found = -1;
    for (int i = 0; i < g_tape.n_params; i++) {
        // adam[i] corresponds to the i-th unique param
        // We need a way to match — use the data pointer
        // (params share the same data pointer across tape recordings)
        if (g_tape.adam[i].m && g_tape.adam[i].m->len == param->len) {
            // Already has an adam state of the right size — reuse index
            // NOTE: this is a simplification. In practice we'd track by pointer.
        }
    }
    if (found < 0 && g_tape.n_params < AM_TAPE_MAX_PARAMS) {
        int pi = g_tape.n_params;
        if (!g_tape.adam[pi].m) {
            g_tape.adam[pi].m = am_array_new(param->len);
            g_tape.adam[pi].v = am_array_new(param->len);
            g_tape.adam[pi].t = 0;
        }
        g_tape.n_params++;
    }

    g_tape.count++;
    return idx;
}

// Accumulate gradient into an entry
static void tape_acc_grad(int idx, const float* grad, int len) {
    if (idx < 0 || idx >= g_tape.count) return;
    AM_TapeEntry* e = &g_tape.entries[idx];
    if (!e->grad) {
        e->grad = am_array_new(len);
        if (!e->grad) return;
    }
    int n = e->grad->len < len ? e->grad->len : len;
    for (int i = 0; i < n; i++) e->grad->data[i] += grad[i];
}

// Backward pass: propagate gradients from loss to all parents
void am_tape_backward(int loss_idx) {
    if (loss_idx < 0 || loss_idx >= g_tape.count) return;

    // Initialize loss gradient to 1.0
    AM_TapeEntry* loss = &g_tape.entries[loss_idx];
    if (!loss->grad) {
        loss->grad = am_array_new(loss->output->len);
    }
    for (int i = 0; i < loss->grad->len; i++) loss->grad->data[i] = 1.0f;

    // Reverse topological order (entries are already in forward order)
    for (int idx = loss_idx; idx >= 0; idx--) {
        AM_TapeEntry* e = &g_tape.entries[idx];
        if (!e->grad) continue;
        float* dout = e->grad->data;
        int out_len = e->output->len;

        switch (e->op) {
        case AM_OP_ADD: {
            // y = a + b → da += dout, db += dout
            if (e->parent1 >= 0) tape_acc_grad(e->parent1, dout, out_len);
            if (e->parent2 >= 0) tape_acc_grad(e->parent2, dout, out_len);
            break;
        }
        case AM_OP_MUL: {
            // y = a * b → da += dout * b, db += dout * a
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                AM_TapeEntry* pa = &g_tape.entries[e->parent1];
                AM_TapeEntry* pb = &g_tape.entries[e->parent2];
                float* ga = (float*)calloc(out_len, sizeof(float));
                float* gb = (float*)calloc(out_len, sizeof(float));
                if (ga && gb) {
                    for (int i = 0; i < out_len; i++) {
                        ga[i] = dout[i] * pb->output->data[i];
                        gb[i] = dout[i] * pa->output->data[i];
                    }
                    tape_acc_grad(e->parent1, ga, out_len);
                    tape_acc_grad(e->parent2, gb, out_len);
                }
                free(ga); free(gb);
            }
            break;
        }
        case AM_OP_SCALE: {
            // y = a * scalar → da += dout * scalar
            if (e->parent1 >= 0) {
                float* ga = (float*)calloc(out_len, sizeof(float));
                if (ga) {
                    for (int i = 0; i < out_len; i++) ga[i] = dout[i] * e->aux;
                    tape_acc_grad(e->parent1, ga, out_len);
                }
                free(ga);
            }
            break;
        }
        case AM_OP_MATVEC: {
            // y = W @ x → dW += dout ⊗ x, dx += W^T @ dout
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                AM_TapeEntry* pw = &g_tape.entries[e->parent1]; // W
                AM_TapeEntry* px = &g_tape.entries[e->parent2]; // x
                int rows = pw->output->rows;
                int cols = pw->output->cols;
                if (rows > 0 && cols > 0) {
                    // dW: outer product dout ⊗ x (rows × cols)
                    float* dw = (float*)calloc(rows * cols, sizeof(float));
                    if (dw) {
                        for (int i = 0; i < rows; i++)
                            for (int j = 0; j < cols; j++)
                                dw[i * cols + j] = dout[i] * px->output->data[j];
                        tape_acc_grad(e->parent1, dw, rows * cols);
                    }
                    free(dw);
                    // dx: W^T @ dout
                    float* dx = (float*)calloc(cols, sizeof(float));
                    if (dx) {
                        for (int j = 0; j < cols; j++)
                            for (int i = 0; i < rows; i++)
                                dx[j] += pw->output->data[i * cols + j] * dout[i];
                        tape_acc_grad(e->parent2, dx, cols);
                    }
                    free(dx);
                }
            }
            break;
        }
        case AM_OP_SILU: {
            // y = x * sigmoid(x) → dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            if (e->parent1 >= 0) {
                AM_TapeEntry* px = &g_tape.entries[e->parent1];
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++) {
                        float x = px->output->data[i];
                        float sig = 1.0f / (1.0f + expf(-x));
                        gx[i] = dout[i] * sig * (1.0f + x * (1.0f - sig));
                    }
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }
        case AM_OP_SOFTMAX: {
            // y = softmax(x) → Jacobian: diag(y) - y⊗y
            // dsoftmax_i = y_i * (dout_i - sum(dout * y))
            if (e->parent1 >= 0) {
                float dot_dy = 0;
                for (int i = 0; i < out_len; i++)
                    dot_dy += dout[i] * e->output->data[i];
                float* gx = (float*)calloc(out_len, sizeof(float));
                if (gx) {
                    for (int i = 0; i < out_len; i++)
                        gx[i] = e->output->data[i] * (dout[i] - dot_dy);
                    tape_acc_grad(e->parent1, gx, out_len);
                }
                free(gx);
            }
            break;
        }
        case AM_OP_RMSNORM: {
            // y = x / rms, rms = sqrt(mean(x^2) + eps)
            // Simplified gradient: similar to LayerNorm but without mean subtraction
            if (e->parent1 >= 0) {
                AM_TapeEntry* px = &g_tape.entries[e->parent1];
                int n = out_len;
                float ss = 0;
                for (int i = 0; i < n; i++) ss += px->output->data[i] * px->output->data[i];
                float rms = sqrtf(ss / n + 1e-6f);
                float rms3 = rms * rms * rms;
                float sum_dout_x = 0;
                for (int i = 0; i < n; i++)
                    sum_dout_x += dout[i] * px->output->data[i];
                float* gx = (float*)calloc(n, sizeof(float));
                if (gx) {
                    for (int i = 0; i < n; i++)
                        gx[i] = (dout[i] / rms) - (px->output->data[i] * sum_dout_x / (n * rms3));
                    tape_acc_grad(e->parent1, gx, n);
                }
                free(gx);
            }
            break;
        }
        case AM_OP_CROSS_ENT: {
            // loss = -log(softmax(logits)[target])
            // d_logits = softmax(logits) - one_hot(target)
            if (e->parent1 >= 0) {
                AM_TapeEntry* pl = &g_tape.entries[e->parent1]; // logits
                int n = pl->output->len;
                int target = (int)e->aux;
                // Compute softmax of logits
                float mx = pl->output->data[0];
                for (int i = 1; i < n; i++)
                    if (pl->output->data[i] > mx) mx = pl->output->data[i];
                float* sm = (float*)calloc(n, sizeof(float));
                if (sm) {
                    float sum = 0;
                    for (int i = 0; i < n; i++) {
                        sm[i] = expf(pl->output->data[i] - mx);
                        sum += sm[i];
                    }
                    for (int i = 0; i < n; i++) sm[i] /= sum;
                    // gradient = softmax - one_hot
                    if (target >= 0 && target < n) sm[target] -= 1.0f;
                    // Scale by dout (which is 1.0 for loss)
                    for (int i = 0; i < n; i++) sm[i] *= dout[0];
                    tape_acc_grad(e->parent1, sm, n);
                }
                free(sm);
            }
            break;
        }
        case AM_OP_EMB_LOOKUP: {
            // y = wte[token_id, :] → d_wte[token_id, :] += dout
            if (e->parent1 >= 0) {
                AM_TapeEntry* pw = &g_tape.entries[e->parent1]; // wte
                int token_id = (int)e->aux;
                int cols = pw->output->cols;
                if (cols > 0 && token_id >= 0 && token_id < pw->output->rows) {
                    // Need full-size gradient for wte
                    float* gw = (float*)calloc(pw->output->len, sizeof(float));
                    if (gw) {
                        for (int i = 0; i < cols && i < out_len; i++)
                            gw[token_id * cols + i] = dout[i];
                        tape_acc_grad(e->parent1, gw, pw->output->len);
                    }
                    free(gw);
                }
            }
            break;
        }
        // ── Phase 5: sequence-level backward ──

        case AM_OP_SEQ_EMBED: {
            // h[t*D+d] = wte[tok*D+d] + wpe[pos*D+d]
            // d_wte[tok*D+d] += dout[t*D+d], d_wpe[pos*D+d] += dout[t*D+d]
            if (e->parent1 >= 0 && e->parent3 >= 0) {
                AM_TapeEntry* pwte = &g_tape.entries[e->parent1];
                AM_TapeEntry* pwpe = &g_tape.entries[e->parent2];
                AM_TapeEntry* ptok = &g_tape.entries[e->parent3]; // tokens array
                int T = (int)e->aux;
                int D = (int)e->aux2;
                float* dwte = (float*)calloc(pwte->output->len, sizeof(float));
                float* dwpe = (float*)calloc(pwpe->output->len, sizeof(float));
                if (dwte && dwpe) {
                    for (int t = 0; t < T; t++) {
                        int tok = (int)ptok->output->data[t];
                        if (tok < 0) tok = 0;
                        if (tok >= pwte->output->rows) tok = pwte->output->rows - 1;
                        int pos = t < pwpe->output->rows ? t : pwpe->output->rows - 1;
                        for (int d = 0; d < D; d++) {
                            dwte[tok * D + d] += dout[t * D + d];
                            dwpe[pos * D + d] += dout[t * D + d];
                        }
                    }
                    tape_acc_grad(e->parent1, dwte, pwte->output->len);
                    tape_acc_grad(e->parent2, dwpe, pwpe->output->len);
                }
                free(dwte); free(dwpe);
            }
            break;
        }

        case AM_OP_SEQ_MATVEC: {
            // Y[t*out+i] = sum_j W[i*in+j] * X[t*in+j]
            // dW[i*in+j] += sum_t dout[t*out+i] * X[t*in+j]
            // dX[t*in+j] += sum_i W[i*in+j] * dout[t*out+i]
            if (e->parent1 >= 0 && e->parent2 >= 0) {
                AM_TapeEntry* pw = &g_tape.entries[e->parent1]; // W
                AM_TapeEntry* px = &g_tape.entries[e->parent2]; // X
                int T = (int)e->aux;
                int out_d = pw->output->rows;
                int in_d = pw->output->cols;
                float* dw = (float*)calloc(pw->output->len, sizeof(float));
                float* dx = (float*)calloc(px->output->len, sizeof(float));
                if (dw && dx) {
                    float* Wd = pw->output->data;
                    float* Xd = px->output->data;
                    // dX: each position t independent → parallelize over t
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(static) if(T > 16)
                    #endif
                    for (int t = 0; t < T; t++) {
                        float* dout_t = dout + t * out_d;
                        // dX_t += W^T @ dout_t
                        for (int j = 0; j < in_d; j++)
                            for (int i = 0; i < out_d; i++)
                                dx[t * in_d + j] += Wd[i * in_d + j] * dout_t[i];
                    }
                    // dW: accumulates across T → can't trivially parallelize outer loop
                    for (int t = 0; t < T; t++) {
                        float* dout_t = dout + t * out_d;
                        float* x_t = Xd + t * in_d;
                        for (int i = 0; i < out_d; i++)
                            for (int j = 0; j < in_d; j++)
                                dw[i * in_d + j] += dout_t[i] * x_t[j];
                    }
                    tape_acc_grad(e->parent1, dw, pw->output->len);
                    tape_acc_grad(e->parent2, dx, px->output->len);
                }
                free(dw); free(dx);
            }
            break;
        }

        case AM_OP_SEQ_RMSNORM: {
            // For each position t: y_t = x_t / rms_t where rms_t = sqrt(mean(x_t^2) + eps)
            if (e->parent1 >= 0) {
                AM_TapeEntry* px = &g_tape.entries[e->parent1];
                int T = (int)e->aux;
                int D = (int)e->aux2;
                float* gx = (float*)calloc(T * D, sizeof(float));
                if (gx) {
                    float* Xrn = px->output->data;
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(static) if(T > 32)
                    #endif
                    for (int t = 0; t < T; t++) {
                        float* x_t = Xrn + t * D;
                        float* dout_t = dout + t * D;
                        float ss = 0;
                        for (int d = 0; d < D; d++) ss += x_t[d] * x_t[d];
                        float rms = sqrtf(ss / D + 1e-6f);
                        float rms3 = rms * rms * rms;
                        float sum_dx = 0;
                        for (int d = 0; d < D; d++) sum_dx += dout_t[d] * x_t[d];
                        for (int d = 0; d < D; d++)
                            gx[t * D + d] = (dout_t[d] / rms) - (x_t[d] * sum_dx / (D * rms3));
                    }
                    tape_acc_grad(e->parent1, gx, T * D);
                }
                free(gx);
            }
            break;
        }

        case AM_OP_CAUSAL_ATTN: {
            // Causal self-attention backward
            // Forward: for each i: scores_j = q_i·k_j/sqrt(D), attn = softmax(scores), out_i = sum attn_j * v_j
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                AM_TapeEntry* pq = &g_tape.entries[e->parent1]; // Q
                AM_TapeEntry* pk = &g_tape.entries[e->parent2]; // K
                AM_TapeEntry* pv = &g_tape.entries[e->parent3]; // V
                int T = (int)e->aux;
                int D = (int)e->aux2;
                float sc = 1.0f / sqrtf((float)D);
                float* dq = (float*)calloc(T * D, sizeof(float));
                float* dk = (float*)calloc(T * D, sizeof(float));
                float* dv = (float*)calloc(T * D, sizeof(float));
                if (dq && dk && dv) {
                    for (int i = 0; i < T; i++) {
                        float* qi = pq->output->data + i * D;
                        float* dout_i = dout + i * D;
                        // Recompute attention weights for position i
                        float* scores = (float*)calloc(i + 1, sizeof(float));
                        float* attn = (float*)calloc(i + 1, sizeof(float));
                        if (!scores || !attn) { free(scores); free(attn); continue; }
                        float mx = -1e30f;
                        for (int j = 0; j <= i; j++) {
                            float* kj = pk->output->data + j * D;
                            float dot = 0;
                            for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                            scores[j] = dot * sc;
                            if (scores[j] > mx) mx = scores[j];
                        }
                        float sm = 0;
                        for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                        if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;

                        // d_attn[j] = dout_i · v_j
                        float* d_attn = (float*)calloc(i + 1, sizeof(float));
                        if (d_attn) {
                            for (int j = 0; j <= i; j++) {
                                float* vj = pv->output->data + j * D;
                                for (int d = 0; d < D; d++) d_attn[j] += dout_i[d] * vj[d];
                            }
                            // dv[j] += attn[j] * dout_i
                            for (int j = 0; j <= i; j++) {
                                float* dvj = dv + j * D;
                                for (int d = 0; d < D; d++) dvj[d] += attn[j] * dout_i[d];
                            }
                            // softmax backward: dscore[j] = attn[j] * (d_attn[j] - sum(d_attn * attn))
                            float dot_da = 0;
                            for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                            for (int j = 0; j <= i; j++) {
                                float ds = attn[j] * (d_attn[j] - dot_da) * sc;
                                // dq_i += ds * k_j, dk_j += ds * q_i
                                float* kj = pk->output->data + j * D;
                                for (int d = 0; d < D; d++) {
                                    dq[i * D + d] += ds * kj[d];
                                    dk[j * D + d] += ds * qi[d];
                                }
                            }
                        }
                        free(scores); free(attn); free(d_attn);
                    }
                    tape_acc_grad(e->parent1, dq, T * D);
                    tape_acc_grad(e->parent2, dk, T * D);
                    tape_acc_grad(e->parent3, dv, T * D);
                }
                free(dq); free(dk); free(dv);
            }
            break;
        }

        case AM_OP_MH_CAUSAL_ATTN: {
            // Multi-head causal self-attention backward
            // aux = T, aux2 = head_dim. D recovered from output->len / T.
            if (e->parent1 >= 0 && e->parent2 >= 0 && e->parent3 >= 0) {
                AM_TapeEntry* pq = &g_tape.entries[e->parent1]; // Q
                AM_TapeEntry* pk = &g_tape.entries[e->parent2]; // K
                AM_TapeEntry* pv = &g_tape.entries[e->parent3]; // V
                int T = (int)e->aux;
                int head_dim = (int)e->aux2;
                int D = e->output->len / T;
                int n_heads = D / head_dim;
                float sc = 1.0f / sqrtf((float)head_dim);
                float* dq = (float*)calloc(T * D, sizeof(float));
                float* dk = (float*)calloc(T * D, sizeof(float));
                float* dv = (float*)calloc(T * D, sizeof(float));
                if (dq && dk && dv) {
                    for (int h = 0; h < n_heads; h++) {
                        int ho = h * head_dim;
                        for (int i = 0; i < T; i++) {
                            float* qi = pq->output->data + i * D + ho;
                            float* dout_i = dout + i * D + ho;
                            float* scores = (float*)calloc(i + 1, sizeof(float));
                            float* attn = (float*)calloc(i + 1, sizeof(float));
                            if (!scores || !attn) { free(scores); free(attn); continue; }
                            float mx = -1e30f;
                            for (int j = 0; j <= i; j++) {
                                float* kj = pk->output->data + j * D + ho;
                                float dot = 0;
                                for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                                scores[j] = dot * sc;
                                if (scores[j] > mx) mx = scores[j];
                            }
                            float sm = 0;
                            for (int j = 0; j <= i; j++) { attn[j] = expf(scores[j] - mx); sm += attn[j]; }
                            if (sm > 0) for (int j = 0; j <= i; j++) attn[j] /= sm;
                            float* d_attn = (float*)calloc(i + 1, sizeof(float));
                            if (d_attn) {
                                for (int j = 0; j <= i; j++) {
                                    float* vj = pv->output->data + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) d_attn[j] += dout_i[d] * vj[d];
                                }
                                for (int j = 0; j <= i; j++) {
                                    float* dvj = dv + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) dvj[d] += attn[j] * dout_i[d];
                                }
                                float dot_da = 0;
                                for (int j = 0; j <= i; j++) dot_da += d_attn[j] * attn[j];
                                for (int j = 0; j <= i; j++) {
                                    float ds = attn[j] * (d_attn[j] - dot_da) * sc;
                                    float* kj = pk->output->data + j * D + ho;
                                    for (int d = 0; d < head_dim; d++) {
                                        dq[i * D + ho + d] += ds * kj[d];
                                        dk[j * D + ho + d] += ds * qi[d];
                                    }
                                }
                            }
                            free(scores); free(attn); free(d_attn);
                        }
                    }
                    tape_acc_grad(e->parent1, dq, T * D);
                    tape_acc_grad(e->parent2, dk, T * D);
                    tape_acc_grad(e->parent3, dv, T * D);
                }
                free(dq); free(dk); free(dv);
            }
            break;
        }

        case AM_OP_SEQ_CROSSENT: {
            // loss = mean over t of -log(softmax(logits_t)[target_t])
            // d_logits[t*V+j] = (softmax[j] - one_hot[target]) / T
            if (e->parent1 >= 0) {
                AM_TapeEntry* pl = &g_tape.entries[e->parent1]; // logits
                AM_TapeEntry* pt = &g_tape.entries[e->parent2]; // targets
                int T = (int)e->aux;
                int V = (int)e->aux2;
                float* dl = (float*)calloc(T * V, sizeof(float));
                if (dl && pt) {
                    for (int t = 0; t < T; t++) {
                        float* logits_t = pl->output->data + t * V;
                        int target = (int)pt->output->data[t];
                        if (target < 0 || target >= V) target = 0;
                        float mx = logits_t[0];
                        for (int j = 1; j < V; j++)
                            if (logits_t[j] > mx) mx = logits_t[j];
                        float sum = 0;
                        for (int j = 0; j < V; j++) {
                            dl[t * V + j] = expf(logits_t[j] - mx);
                            sum += dl[t * V + j];
                        }
                        for (int j = 0; j < V; j++) dl[t * V + j] /= sum;
                        dl[t * V + target] -= 1.0f;
                        // Scale by dout[0] / T
                        float s = dout[0] / T;
                        for (int j = 0; j < V; j++) dl[t * V + j] *= s;
                    }
                    tape_acc_grad(e->parent1, dl, T * V);
                }
                free(dl);
            }
            break;
        }

        default:
            break;
        }
    }
}

// Adam optimizer step: update all parameters using their accumulated gradients
void am_tape_adam_step(float lr) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    int param_idx = 0;

    for (int i = 0; i < g_tape.count && param_idx < g_tape.n_params; i++) {
        AM_TapeEntry* e = &g_tape.entries[i];
        if (!e->is_param || !e->grad) continue;

        AM_AdamState* as = &g_tape.adam[param_idx];
        if (!as->m || !as->v) { param_idx++; continue; }

        as->t++;
        int n = e->output->len;
        if (as->m->len < n) n = as->m->len;

        for (int j = 0; j < n; j++) {
            float g = e->grad->data[j];
            as->m->data[j] = beta1 * as->m->data[j] + (1.0f - beta1) * g;
            as->v->data[j] = beta2 * as->v->data[j] + (1.0f - beta2) * g * g;
            float m_hat = as->m->data[j] / (1.0f - powf(beta1, (float)as->t));
            float v_hat = as->v->data[j] / (1.0f - powf(beta2, (float)as->t));
            e->output->data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
        param_idx++;
    }
}

// Find tape entry index by array pointer (-1 if not found)
static int tape_find_entry(AM_Array* arr) {
    if (!arr) return -1;
    for (int i = g_tape.count - 1; i >= 0; i--) {
        if (g_tape.entries[i].output && g_tape.entries[i].output->data == arr->data)
            return i;
    }
    return -1;
}

// Ensure array is on tape. If not found, record as a non-trainable leaf.
// Returns entry index.
static int tape_ensure_entry(AM_Array* arr) {
    if (!arr || !g_tape.active) return -1;
    int idx = tape_find_entry(arr);
    if (idx >= 0) return idx;
    // Record as leaf (OP_NONE, not a param — just for backward data access)
    return am_tape_record(arr, AM_OP_NONE, -1, -1, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASYNC — SPAWN/AWAIT/CHANNEL (v4.0 Phase 4)
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef AM_ASYNC_DISABLED

// Thread argument: holds the AML script to execute
typedef struct {
    char* script;       // heap-allocated AML script text
    int   slot_idx;     // index into g_spawns
} AM_SpawnArg;

// Thread entry point: runs an AML script in its own context
static void* am_spawn_thread_fn(void* arg) {
    AM_SpawnArg* sa = (AM_SpawnArg*)arg;

    // Execute the script (am_exec creates its own AML_ExecCtx)
    int rc = am_exec(sa->script);

    // Mark slot as done
    pthread_mutex_lock(&g_spawn_mutex);
    if (sa->slot_idx >= 0 && sa->slot_idx < AM_MAX_SPAWNS) {
        g_spawns[sa->slot_idx].active = 0;
        g_spawns[sa->slot_idx].result = rc;
    }
    pthread_mutex_unlock(&g_spawn_mutex);

    free(sa->script);
    free(sa);
    return NULL;
}

// Launch a spawn: create thread running the given AML script
int am_spawn_launch(const char* name, const char* script) {
    if (g_spawn_count >= AM_MAX_SPAWNS) return -1;

    int idx = g_spawn_count;
    snprintf(g_spawns[idx].name, AM_SPAWN_NAME_LEN, "%s", name);
    g_spawns[idx].active = 1;
    g_spawns[idx].joined = 0;
    g_spawns[idx].result = 0;

    AM_SpawnArg* arg = (AM_SpawnArg*)malloc(sizeof(AM_SpawnArg));
    if (!arg) return -1;
    arg->script = strdup(script);
    if (!arg->script) { free(arg); return -1; }
    arg->slot_idx = idx;

    int err = pthread_create(&g_spawn_threads[idx], NULL, am_spawn_thread_fn, arg);
    if (err != 0) {
        free(arg->script);
        free(arg);
        g_spawns[idx].active = 0;
        return -1;
    }

    g_spawn_count++;
    return idx;
}

// Await a specific spawn by name. Returns result code.
int am_spawn_await(const char* name) {
    for (int i = 0; i < g_spawn_count; i++) {
        if (strcmp(g_spawns[i].name, name) == 0 && !g_spawns[i].joined) {
            pthread_join(g_spawn_threads[i], NULL);
            g_spawns[i].joined = 1;
            return g_spawns[i].result;
        }
    }
    return -1;
}

// Await all spawns
void am_spawn_await_all(void) {
    for (int i = 0; i < g_spawn_count; i++) {
        if (!g_spawns[i].joined) {
            pthread_join(g_spawn_threads[i], NULL);
            g_spawns[i].joined = 1;
        }
    }
}

int am_spawn_count(void) {
    int n = 0;
    for (int i = 0; i < g_spawn_count; i++)
        if (g_spawns[i].active) n++;
    return n;
}

// Reset spawn state (called from am_init)
static void am_spawn_reset(void) {
    am_spawn_await_all();
    g_spawn_count = 0;
    memset(g_spawns, 0, sizeof(g_spawns));
}

// --- CHANNEL ---

// Find channel by name (-1 if not found)
static int channel_find(const char* name) {
    for (int i = 0; i < g_channel_count; i++)
        if (g_channels[i].active && strcmp(g_channels[i].name, name) == 0)
            return i;
    return -1;
}

// Create a channel with given capacity
int am_channel_create(const char* name, int capacity) {
    if (g_channel_count >= AM_MAX_CHANNELS || capacity <= 0) return -1;
    if (capacity > AM_CHANNEL_BUF) capacity = AM_CHANNEL_BUF;

    int idx = g_channel_count;
    memset(&g_channels[idx], 0, sizeof(AM_ChannelSlot));
    snprintf(g_channels[idx].name, AM_SPAWN_NAME_LEN, "%s", name);
    g_channels[idx].capacity = capacity;
    g_channels[idx].active = 1;
    g_channel_count++;
    return idx;
}

// Write a float to a channel (blocking if full)
int am_channel_write(const char* name, float value) {
    pthread_mutex_lock(&g_channel_mutex);
    int idx = channel_find(name);
    if (idx < 0) { pthread_mutex_unlock(&g_channel_mutex); return -1; }

    // Wait until not full (with timeout to prevent deadlock)
    int tries = 0;
    while (g_channels[idx].count >= g_channels[idx].capacity && tries < 1000) {
        pthread_mutex_unlock(&g_channel_mutex);
        struct timespec ts = {0, 1000000}; // 1ms
        nanosleep(&ts, NULL);
        pthread_mutex_lock(&g_channel_mutex);
        tries++;
    }
    if (g_channels[idx].count >= g_channels[idx].capacity) {
        pthread_mutex_unlock(&g_channel_mutex);
        return -1; // channel full, timeout
    }

    g_channels[idx].data[g_channels[idx].tail] = value;
    g_channels[idx].tail = (g_channels[idx].tail + 1) % g_channels[idx].capacity;
    g_channels[idx].count++;
    pthread_cond_broadcast(&g_channel_cond);
    pthread_mutex_unlock(&g_channel_mutex);
    return 0;
}

// Read a float from a channel (blocking if empty, with timeout)
int am_channel_read(const char* name, float* out) {
    pthread_mutex_lock(&g_channel_mutex);
    int idx = channel_find(name);
    if (idx < 0) { pthread_mutex_unlock(&g_channel_mutex); return -1; }

    // Wait until not empty (with timeout)
    int tries = 0;
    while (g_channels[idx].count == 0 && tries < 1000) {
        pthread_mutex_unlock(&g_channel_mutex);
        struct timespec ts = {0, 1000000}; // 1ms
        nanosleep(&ts, NULL);
        pthread_mutex_lock(&g_channel_mutex);
        tries++;
    }
    if (g_channels[idx].count == 0) {
        pthread_mutex_unlock(&g_channel_mutex);
        return -1; // channel empty, timeout
    }

    *out = g_channels[idx].data[g_channels[idx].head];
    g_channels[idx].head = (g_channels[idx].head + 1) % g_channels[idx].capacity;
    g_channels[idx].count--;
    pthread_cond_broadcast(&g_channel_cond);
    pthread_mutex_unlock(&g_channel_mutex);
    return 0;
}

int am_channel_count(void) {
    int n = 0;
    for (int i = 0; i < g_channel_count; i++)
        if (g_channels[i].active) n++;
    return n;
}

void am_channel_close_all(void) {
    pthread_mutex_lock(&g_channel_mutex);
    for (int i = 0; i < g_channel_count; i++)
        g_channels[i].active = 0;
    g_channel_count = 0;
    pthread_mutex_unlock(&g_channel_mutex);
}

// Reset channels (called from am_init)
static void am_channel_reset(void) {
    am_channel_close_all();
}

#endif // AM_ASYNC_DISABLED

// Symbol table operations
static float* symtab_get(AML_Symtab* tab, const char* name) {
    for (int i = 0; i < tab->count; i++) {
        if (strcmp(tab->vars[i].name, name) == 0)
            return &tab->vars[i].value;
    }
    return NULL;
}

// Get full variable record (needed for array access)
static AML_Var* symtab_get_var(AML_Symtab* tab, const char* name) {
    for (int i = 0; i < tab->count; i++) {
        if (strcmp(tab->vars[i].name, name) == 0)
            return &tab->vars[i];
    }
    return NULL;
}

static int symtab_set(AML_Symtab* tab, const char* name, float value) {
    for (int i = 0; i < tab->count; i++) {
        if (strcmp(tab->vars[i].name, name) == 0) {
            // If overwriting an array with a float, free the array
            if (tab->vars[i].type == AML_TYPE_ARRAY && tab->vars[i].array) {
                am_array_free(tab->vars[i].array);
                tab->vars[i].array = NULL;
            }
            tab->vars[i].type = AML_TYPE_FLOAT;
            tab->vars[i].value = value;
            return 0;
        }
    }
    if (tab->count >= AML_MAX_VARS) return 1;
    snprintf(tab->vars[tab->count].name, AML_MAX_NAME, "%s", name);
    tab->vars[tab->count].type = AML_TYPE_FLOAT;
    tab->vars[tab->count].value = value;
    tab->vars[tab->count].array = NULL;
    tab->count++;
    return 0;
}

// Set an array variable (takes ownership of arr's refcount)
static int symtab_set_array(AML_Symtab* tab, const char* name, AM_Array* arr) {
    for (int i = 0; i < tab->count; i++) {
        if (strcmp(tab->vars[i].name, name) == 0) {
            // Free old array if any
            if (tab->vars[i].type == AML_TYPE_ARRAY && tab->vars[i].array) {
                am_array_free(tab->vars[i].array);
            }
            tab->vars[i].type = AML_TYPE_ARRAY;
            tab->vars[i].value = 0;
            tab->vars[i].array = arr;
            return 0;
        }
    }
    if (tab->count >= AML_MAX_VARS) return 1;
    snprintf(tab->vars[tab->count].name, AML_MAX_NAME, "%s", name);
    tab->vars[tab->count].type = AML_TYPE_ARRAY;
    tab->vars[tab->count].value = 0;
    tab->vars[tab->count].array = arr;
    tab->count++;
    return 0;
}

// Free all arrays in a symbol table (for scope cleanup)
static void symtab_clear_arrays(AML_Symtab* tab) {
    for (int i = 0; i < tab->count; i++) {
        if (tab->vars[i].type == AML_TYPE_ARRAY && tab->vars[i].array) {
            am_array_free(tab->vars[i].array);
            tab->vars[i].array = NULL;
        }
    }
}

// Resolve full variable (AML_Var*): locals → globals
static AML_Var* resolve_var_full(AML_ExecCtx* ctx, const char* name) {
    if (ctx->call_depth > 0) {
        AML_Var* v = symtab_get_var(&ctx->locals[ctx->call_depth - 1], name);
        if (v) return v;
    }
    return symtab_get_var(&ctx->globals, name);
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

// Forward declarations for user function calls from expressions
static int aml_call_func(AML_ExecCtx* ctx, AML_Func* f, float* args, int nargs, int lineno);

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

        // v4.0: array indexing — name[index]
        if (*e->p == '[') {
            e->p++;
            float idx_f = expr_or(e);
            expr_skip_ws(e);
            if (*e->p == ']') e->p++;
            int idx = (int)idx_f;

            if (e->ctx) {
                AML_Var* var = resolve_var_full(e->ctx, name);
                if (var && var->type == AML_TYPE_ARRAY && var->array) {
                    if (idx >= 0 && idx < var->array->len)
                        return var->array->data[idx];
                }
            }
            return 0;
        }

        // function call
        if (*e->p == '(') {
            // v4.0: array scalar-returning builtins need raw arg names
            // Parse them BEFORE evaluating args as expressions
            if (e->ctx && (strcasecmp(name, "len") == 0 ||
                           strcasecmp(name, "sum") == 0 ||
                           strcasecmp(name, "dot") == 0 ||
                           strcasecmp(name, "rows") == 0 ||
                           strcasecmp(name, "cols") == 0)) {
                // Parse argument names (identifiers), not evaluated expressions
                e->p++; // skip '('
                char arg_names[AML_MAX_PARAMS][AML_MAX_NAME];
                int n_arg_names = 0;
                expr_skip_ws(e);
                while (*e->p != ')' && *e->p && n_arg_names < AML_MAX_PARAMS) {
                    int ai = 0;
                    while ((isalnum((unsigned char)*e->p) || *e->p == '_') && ai < AML_MAX_NAME - 1)
                        arg_names[n_arg_names][ai++] = *e->p++;
                    arg_names[n_arg_names][ai] = 0;
                    n_arg_names++;
                    expr_skip_ws(e);
                    if (*e->p == ',') { e->p++; expr_skip_ws(e); }
                }
                if (*e->p == ')') e->p++;

                if (strcasecmp(name, "len") == 0 && n_arg_names >= 1) {
                    AML_Var* v = resolve_var_full(e->ctx, arg_names[0]);
                    if (v && v->type == AML_TYPE_ARRAY && v->array)
                        return (float)v->array->len;
                    return 0;
                }
                if (strcasecmp(name, "sum") == 0 && n_arg_names >= 1) {
                    AML_Var* v = resolve_var_full(e->ctx, arg_names[0]);
                    if (v && v->type == AML_TYPE_ARRAY && v->array) {
                        float s = 0;
                        for (int j = 0; j < v->array->len; j++) s += v->array->data[j];
                        return s;
                    }
                    return 0;
                }
                if (strcasecmp(name, "dot") == 0 && n_arg_names >= 2) {
                    AML_Var* va = resolve_var_full(e->ctx, arg_names[0]);
                    AML_Var* vb = resolve_var_full(e->ctx, arg_names[1]);
                    if (va && va->type == AML_TYPE_ARRAY && va->array &&
                        vb && vb->type == AML_TYPE_ARRAY && vb->array) {
                        int n = va->array->len < vb->array->len ? va->array->len : vb->array->len;
                        float d = 0;
                        for (int j = 0; j < n; j++) d += va->array->data[j] * vb->array->data[j];
                        return d;
                    }
                    return 0;
                }
                // Phase 2: rows(M), cols(M)
                if (strcasecmp(name, "rows") == 0 && n_arg_names >= 1) {
                    AML_Var* v = resolve_var_full(e->ctx, arg_names[0]);
                    if (v && v->type == AML_TYPE_ARRAY && v->array)
                        return (float)v->array->rows;
                    return 0;
                }
                if (strcasecmp(name, "cols") == 0 && n_arg_names >= 1) {
                    AML_Var* v = resolve_var_full(e->ctx, arg_names[0]);
                    if (v && v->type == AML_TYPE_ARRAY && v->array)
                        return (float)v->array->cols;
                    return 0;
                }
                return 0;
            }

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

            // look up user-defined function — v4.0: actually call it and return value
            if (e->ctx) {
                for (int fi = 0; fi < e->ctx->funcs.count; fi++) {
                    if (strcmp(e->ctx->funcs.funcs[fi].name, name) == 0) {
                        AML_Func* fn = &e->ctx->funcs.funcs[fi];
                        aml_call_func(e->ctx, fn, args, nargs, 0);
                        if (e->ctx->has_return) {
                            float rv = e->ctx->return_value;
                            // Only reset has_return for scalar returns.
                            // Array returns stay flagged so the assignment handler
                            // can pick them up from ctx->return_array.
                            if (!e->ctx->return_array) {
                                e->ctx->has_return = 0;
                            }
                            return rv;
                        }
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
    // LILITH I/O — named pipes for data infrastructure
    // "Та, которая была до Евы."
    // ─────────────────────────────────────────────────────────────────────────

#ifndef AM_IO_DISABLED
    else if (!strcmp(t, "PIPE")) {
      // PIPE CREATE <path>               — create FIFO at path
      // PIPE OPEN <name> <path> <mode>   — open pipe (mode: READ or WRITE)
      // PIPE WRITE <name> <message>      — write to pipe
      // PIPE READ <name>                 — read from pipe (non-blocking)
      // PIPE CLOSE <name>                — close pipe
      // PIPE CLOSE_ALL                   — close all pipes
      // PIPE LIST                        — list open pipes
      char subcmd[32] = {0};
      char rest[AML_MAX_LINE_LEN] = {0};
      if (arg) sscanf(arg, "%31s %[^\n]", subcmd, rest);
      upcase(subcmd);

      if (!strcmp(subcmd, "CREATE")) {
        // PIPE CREATE /tmp/lilith_idx1
        char path[AM_PIPE_PATH_LEN] = {0};
        sscanf(rest, "%255s", path);
        if (path[0]) {
          am_pipe_create(path);
        } else if (ctx) {
          set_error_at(ctx, lineno, "PIPE CREATE: path required");
        }
      }
      else if (!strcmp(subcmd, "OPEN")) {
        // PIPE OPEN idx1_cmd /tmp/lilith_idx1_cmd WRITE
        char pname[AM_PIPE_NAME_LEN] = {0};
        char path[AM_PIPE_PATH_LEN] = {0};
        char mode_str[16] = {0};
        if (sscanf(rest, "%31s %255s %15s", pname, path, mode_str) >= 2) {
          upcase(mode_str);
          int mode = AM_PIPE_MODE_READ;
          if (!strcmp(mode_str, "WRITE") || !strcmp(mode_str, "W"))
            mode = AM_PIPE_MODE_WRITE;
          int idx = am_pipe_open(pname, path, mode);
          if (idx < 0 && ctx)
            set_error_at(ctx, lineno, "PIPE OPEN failed");
        } else if (ctx) {
          set_error_at(ctx, lineno, "PIPE OPEN: name and path required");
        }
      }
      else if (!strcmp(subcmd, "WRITE")) {
        // PIPE WRITE idx1_cmd "FETCH r/philosophy"
        char pname[AM_PIPE_NAME_LEN] = {0};
        char msg[AM_PIPE_BUF_SIZE] = {0};
        // Parse: first token = name, rest = message (strip quotes)
        char* space = strchr(rest, ' ');
        if (space) {
          int nlen = (int)(space - rest);
          if (nlen >= AM_PIPE_NAME_LEN) nlen = AM_PIPE_NAME_LEN - 1;
          memcpy(pname, rest, nlen);
          pname[nlen] = 0;
          // Skip space, strip surrounding quotes
          const char* mp = space + 1;
          while (*mp == ' ') mp++;
          if (*mp == '"') {
            mp++;
            const char* end = strrchr(mp, '"');
            if (end) {
              int mlen = (int)(end - mp);
              if (mlen >= AM_PIPE_BUF_SIZE) mlen = AM_PIPE_BUF_SIZE - 1;
              memcpy(msg, mp, mlen);
              msg[mlen] = 0;
            } else {
              snprintf(msg, sizeof(msg), "%s", mp);
            }
          } else {
            snprintf(msg, sizeof(msg), "%s", mp);
          }
          am_pipe_write(pname, msg);
        }
      }
      else if (!strcmp(subcmd, "READ")) {
        // PIPE READ idx1_rsp
        char pname[AM_PIPE_NAME_LEN] = {0};
        sscanf(rest, "%31s", pname);
        if (pname[0]) {
          int n = am_pipe_read(pname, g_pipe_read_buf, AM_PIPE_BUF_SIZE);
          if (n > 0) {
            printf("[LILITH] %s: %s\n", pname, g_pipe_read_buf);
            // Store numeric value in AML variable _pipe_value if ctx exists
            if (ctx && ctx->call_depth > 0) {
              symtab_set(&ctx->locals[ctx->call_depth - 1],
                         "_pipe_value", g_pipe_last_value);
            } else if (ctx) {
              symtab_set(&ctx->globals, "_pipe_value", g_pipe_last_value);
            }
          }
        }
      }
      else if (!strcmp(subcmd, "CLOSE")) {
        char pname[AM_PIPE_NAME_LEN] = {0};
        sscanf(rest, "%31s", pname);
        upcase(pname);
        if (!strcmp(pname, "ALL") || !strcmp(rest, "ALL")) {
          am_pipe_close_all();
        } else {
          // Restore original case for name lookup
          sscanf(rest, "%31s", pname);
          am_pipe_close(pname);
        }
      }
      else if (!strcmp(subcmd, "LIST")) {
        printf("[LILITH] pipes (%d open):\n", am_pipe_count());
        for (int i = 0; i < g_pipe_count; i++) {
          if (g_pipes[i].active) {
            printf("[LILITH]   %s → %s (%s)\n",
                   g_pipes[i].name, g_pipes[i].path,
                   g_pipes[i].mode == AM_PIPE_MODE_READ ? "READ" : "WRITE");
          }
        }
      }
    }

    else if (!strcmp(t, "INDEX")) {
      // INDEX <id> <subcmd> [args]  — high-level INDEX node management
      // Sugar over PIPE commands. Uses convention:
      //   pipe name = "idx<id>_cmd" (write) / "idx<id>_rsp" (read)
      //   pipe path = "/tmp/lilith_idx<id>_cmd" / "/tmp/lilith_idx<id>_rsp"
      char id_str[8] = {0};
      char subcmd2[32] = {0};
      char irest[AML_MAX_LINE_LEN] = {0};
      if (arg) sscanf(arg, "%7s %31s %[^\n]", id_str, subcmd2, irest);
      upcase(subcmd2);

      if (id_str[0]) {
        // Build pipe names from INDEX id
        char cmd_name[AM_PIPE_NAME_LEN];
        char rsp_name[AM_PIPE_NAME_LEN];
        char cmd_path[AM_PIPE_PATH_LEN];
        char rsp_path[AM_PIPE_PATH_LEN];
        snprintf(cmd_name, sizeof(cmd_name), "idx%s_cmd", id_str);
        snprintf(rsp_name, sizeof(rsp_name), "idx%s_rsp", id_str);
        snprintf(cmd_path, sizeof(cmd_path), "/tmp/lilith_idx%s_cmd", id_str);
        snprintf(rsp_path, sizeof(rsp_path), "/tmp/lilith_idx%s_rsp", id_str);

        if (!strcmp(subcmd2, "INIT")) {
          // INDEX 1 INIT — create pipes and open them
          am_pipe_create(cmd_path);
          am_pipe_create(rsp_path);
          am_pipe_open(cmd_name, cmd_path, AM_PIPE_MODE_WRITE);
          am_pipe_open(rsp_name, rsp_path, AM_PIPE_MODE_READ);
          printf("[LILITH] INDEX %s initialized\n", id_str);
        }
        else if (!strcmp(subcmd2, "FETCH")) {
          // INDEX 1 FETCH r/philosophy — tell index node to fetch
          char fetch_cmd[AM_PIPE_BUF_SIZE];
          snprintf(fetch_cmd, sizeof(fetch_cmd), "FETCH %s", irest);
          am_pipe_write(cmd_name, fetch_cmd);
        }
        else if (!strcmp(subcmd2, "STATUS")) {
          // INDEX 1 STATUS — request + read status
          am_pipe_write(cmd_name, "STATUS");
          // Try reading response (might not be immediate)
          int n = am_pipe_read(rsp_name, g_pipe_read_buf, AM_PIPE_BUF_SIZE);
          if (n > 0) {
            printf("[LILITH] INDEX %s status: %s\n", id_str, g_pipe_read_buf);
          } else {
            printf("[LILITH] INDEX %s: no response yet\n", id_str);
          }
        }
        else if (!strcmp(subcmd2, "STOP")) {
          am_pipe_write(cmd_name, "STOP");
        }
        else if (!strcmp(subcmd2, "CLOSE")) {
          am_pipe_close(cmd_name);
          am_pipe_close(rsp_name);
        }
      }
    }
#endif // AM_IO_DISABLED

    // ─────────────────────────────────────────────────────────────────────────
    // TAPE — autograd (v4.0 Phase 3)
    // ─────────────────────────────────────────────────────────────────────────

    else if (!strcmp(t, "TAPE")) {
      char subcmd[32] = {0};
      char rest[AML_MAX_LINE_LEN] = {0};
      if (arg) sscanf(arg, "%31s %[^\n]", subcmd, rest);
      upcase(subcmd);

      if (!strcmp(subcmd, "START")) {
        am_tape_start();
      }
      else if (!strcmp(subcmd, "CLEAR")) {
        am_tape_clear();
      }
      else if (!strcmp(subcmd, "BACKWARD")) {
        // TAPE BACKWARD <var_name> — backprop from loss variable
        char vname[AML_MAX_NAME] = {0};
        sscanf(rest, "%31s", vname);
        if (vname[0] && ctx) {
          AML_Var* v = resolve_var_full(ctx, vname);
          if (v && v->type == AML_TYPE_ARRAY && v->array) {
            int tidx = tape_find_entry(v->array);
            if (tidx >= 0) am_tape_backward(tidx);
          }
        }
      }
      else if (!strcmp(subcmd, "ADAM_STEP") || !strcmp(subcmd, "ADAM")) {
        // TAPE ADAM_STEP <lr> or TAPE ADAM <lr>
        float lr = 0.001f;
        if (rest[0]) lr = ctx_float(ctx, rest);
        am_tape_adam_step(lr);
      }
      else if (!strcmp(subcmd, "PARAM")) {
        // TAPE PARAM <var_name> — register variable as trainable parameter
        char vname[AML_MAX_NAME] = {0};
        sscanf(rest, "%31s", vname);
        if (vname[0] && ctx) {
          AML_Var* v = resolve_var_full(ctx, vname);
          if (v && v->type == AML_TYPE_ARRAY && v->array) {
            am_tape_record_param(v->array);
          }
        }
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ASYNC — SPAWN/AWAIT/CHANNEL (v4.0 Phase 4)
    // ─────────────────────────────────────────────────────────────────────────

#ifndef AM_ASYNC_DISABLED
    else if (!strcmp(t, "AWAIT")) {
      // AWAIT name1 name2 ... — wait for spawned threads
      if (arg && *arg) {
        char names[AML_MAX_LINE_LEN];
        snprintf(names, sizeof(names), "%s", arg);
        char* save = NULL;
        char* tok = strtok_r(names, " \t", &save);
        while (tok) {
          am_spawn_await(tok);
          tok = strtok_r(NULL, " \t", &save);
        }
      } else {
        am_spawn_await_all();
      }
    }

    else if (!strcmp(t, "CHANNEL")) {
      char subcmd[32] = {0};
      char rest[AML_MAX_LINE_LEN] = {0};
      if (arg) sscanf(arg, "%31s %[^\n]", subcmd, rest);
      upcase(subcmd);

      if (!strcmp(subcmd, "CREATE")) {
        // CHANNEL CREATE name capacity
        char chname[AM_SPAWN_NAME_LEN] = {0};
        int cap = AM_CHANNEL_BUF;
        sscanf(rest, "%31s %d", chname, &cap);
        if (cap <= 0) cap = AM_CHANNEL_BUF;
        if (cap > AM_CHANNEL_BUF) cap = AM_CHANNEL_BUF;
        if (chname[0]) am_channel_create(chname, cap);
      }
      else if (!strcmp(subcmd, "WRITE")) {
        // CHANNEL WRITE name value_expr
        char chname[AM_SPAWN_NAME_LEN] = {0};
        char vexpr[AML_MAX_LINE_LEN] = {0};
        sscanf(rest, "%31s %[^\n]", chname, vexpr);
        if (chname[0] && vexpr[0] && ctx) {
          float val = ctx_float(ctx, vexpr);
          am_channel_write(chname, val);
        }
      }
      else if (!strcmp(subcmd, "READ")) {
        // CHANNEL READ name var_name
        char chname[AM_SPAWN_NAME_LEN] = {0};
        char vname[AML_MAX_NAME] = {0};
        sscanf(rest, "%31s %31s", chname, vname);
        if (chname[0] && vname[0] && ctx) {
          float out = 0;
          if (am_channel_read(chname, &out) == 0) {
            int d = ctx->call_depth > 0 ? ctx->call_depth - 1 : 0;
            symtab_set(&ctx->locals[d], vname, out);
          }
        }
      }
      else if (!strcmp(subcmd, "CLOSE")) {
        // CHANNEL CLOSE name
        char chname[AM_SPAWN_NAME_LEN] = {0};
        sscanf(rest, "%31s", chname);
        if (chname[0]) {
          // close specific channel by zeroing it
          for (int ci = 0; ci < g_channel_count; ci++) {
            if (g_channels[ci].active && strcmp(g_channels[ci].name, chname) == 0) {
              g_channels[ci].active = 0;
              break;
            }
          }
        } else {
          am_channel_close_all();
        }
      }
    }
#endif // AM_ASYNC_DISABLED

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
// v4.0: supports return values via ctx->has_return / return_value / return_array
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

    // Save caller's return state (nested calls must not clobber it)
    int saved_has_return = ctx->has_return;
    float saved_return_value = ctx->return_value;
    AM_Array* saved_return_array = ctx->return_array;
    int saved_return_type = ctx->return_type;

    // push local scope
    ctx->call_depth++;
    AML_Symtab* locals = &ctx->locals[ctx->call_depth - 1];
    memset(locals, 0, sizeof(AML_Symtab));

    // bind params
    for (int i = 0; i < f->param_count && i < nargs; i++) {
        symtab_set(locals, f->params[i], args[i]);
    }

    // reset return state for this function
    ctx->has_return = 0;
    ctx->return_value = 0;
    ctx->return_array = NULL;

    // execute body
    int rc = aml_exec_block(ctx, f->body_start, f->body_end);

    // v4.0: clean up local arrays on scope exit
    // BUT: if we're returning an array, bump its refcount first
    if (ctx->has_return && ctx->return_array) {
        am_array_ref(ctx->return_array);
    }
    symtab_clear_arrays(locals);

    // pop scope
    ctx->call_depth--;

    // If this function didn't return, restore caller's return state
    if (!ctx->has_return) {
        ctx->has_return = saved_has_return;
        ctx->return_value = saved_return_value;
        ctx->return_array = saved_return_array;
        ctx->return_type = saved_return_type;
    }
    // If this function DID return, has_return/return_value stay set for caller to read

    return rc;
}

// ═══════════════════════════════════════════════════════════════════════════════
// v4.0: ARRAY HELPER — try to parse RHS as array-producing expression
// Returns newly allocated AM_Array*, or NULL if not an array expression.
// Handles: zeros(n), randn(n, std), [1.0, 2.0, 3.0], add(a,b), mul(a,b),
//          scale(a, s), and user function calls that return arrays.
// ═══════════════════════════════════════════════════════════════════════════════

static AM_Array* aml_try_array_expr(AML_ExecCtx* ctx, const char* rhs) {
    // skip whitespace
    while (*rhs == ' ') rhs++;

    // --- array literal: [1.0, 2.0, 3.0] ---
    if (*rhs == '[') {
        rhs++;
        float vals[256];
        int count = 0;
        while (*rhs && *rhs != ']' && count < 256) {
            while (*rhs == ' ' || *rhs == ',') rhs++;
            if (*rhs == ']') break;
            char* end;
            vals[count++] = strtof(rhs, &end);
            rhs = end;
        }
        if (count > 0) {
            AM_Array* arr = am_array_new(count);
            if (arr) memcpy(arr->data, vals, count * sizeof(float));
            return arr;
        }
        return NULL;
    }

    // --- function call: name(args) ---
    char fname[AML_MAX_NAME] = {0};
    int fi = 0;
    while ((isalnum((unsigned char)rhs[fi]) || rhs[fi] == '_') && fi < AML_MAX_NAME - 1) {
        fname[fi] = rhs[fi]; fi++;
    }
    fname[fi] = 0;
    const char* after_name = rhs + fi;
    while (*after_name == ' ') after_name++;
    if (*after_name != '(') return NULL;

    // Parse arguments as raw text tokens (needed for variable names)
    const char* ap = after_name + 1;
    char arg_strs[AML_MAX_PARAMS][AML_MAX_NAME];
    int nargs = 0;
    while (*ap && *ap != ')' && nargs < AML_MAX_PARAMS) {
        while (*ap == ' ' || *ap == ',') ap++;
        if (*ap == ')') break;
        int ai = 0;
        // Capture the whole argument expression (may be a number or identifier)
        int paren_depth = 0;
        while (*ap && (paren_depth > 0 || (*ap != ',' && *ap != ')')) && ai < AML_MAX_NAME - 1) {
            if (*ap == '(') paren_depth++;
            if (*ap == ')') { if (paren_depth > 0) paren_depth--; else break; }
            arg_strs[nargs][ai++] = *ap++;
        }
        // Trim trailing spaces
        while (ai > 0 && arg_strs[nargs][ai-1] == ' ') ai--;
        arg_strs[nargs][ai] = 0;
        nargs++;
    }

    // zeros(n) — create zero-initialized array
    if (strcasecmp(fname, "zeros") == 0 && nargs >= 1) {
        int n = (int)aml_eval(ctx, arg_strs[0]);
        if (n > 0 && n <= AM_MAX_ARRAY_SIZE) return am_array_new(n);
        return NULL;
    }

    // randn(n, std) — random normal initialization
    if (strcasecmp(fname, "randn") == 0 && nargs >= 1) {
        int n = (int)aml_eval(ctx, arg_strs[0]);
        float std = (nargs >= 2) ? aml_eval(ctx, arg_strs[1]) : 1.0f;
        if (n <= 0 || n > AM_MAX_ARRAY_SIZE) return NULL;
        AM_Array* arr = am_array_new(n);
        if (!arr) return NULL;
        // Box-Muller transform for normal distribution
        for (int j = 0; j < n; j += 2) {
            float u1 = ((float)rand() / (float)RAND_MAX) * 0.9998f + 0.0001f;
            float u2 = ((float)rand() / (float)RAND_MAX);
            float r = sqrtf(-2.0f * logf(u1));
            arr->data[j] = r * cosf(2.0f * 3.14159265f * u2) * std;
            if (j + 1 < n)
                arr->data[j + 1] = r * sinf(2.0f * 3.14159265f * u2) * std;
        }
        return arr;
    }

    // add(a, b) — element-wise addition
    if (strcasecmp(fname, "add") == 0 && nargs >= 2) {
        AML_Var* va = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vb = resolve_var_full(ctx, arg_strs[1]);
        if (va && va->type == AML_TYPE_ARRAY && va->array &&
            vb && vb->type == AML_TYPE_ARRAY && vb->array) {
            int n = va->array->len < vb->array->len ? va->array->len : vb->array->len;
            AM_Array* arr = am_array_new(n);
            if (!arr) return NULL;
            for (int j = 0; j < n; j++)
                arr->data[j] = va->array->data[j] + vb->array->data[j];
            if (am_tape_is_active())
                am_tape_record(arr, AM_OP_ADD, tape_ensure_entry(va->array), tape_ensure_entry(vb->array), 0);
            return arr;
        }
        return NULL;
    }

    // mul(a, b) — element-wise multiplication
    if (strcasecmp(fname, "mul") == 0 && nargs >= 2) {
        AML_Var* va = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vb = resolve_var_full(ctx, arg_strs[1]);
        if (va && va->type == AML_TYPE_ARRAY && va->array &&
            vb && vb->type == AML_TYPE_ARRAY && vb->array) {
            int n = va->array->len < vb->array->len ? va->array->len : vb->array->len;
            AM_Array* arr = am_array_new(n);
            if (!arr) return NULL;
            for (int j = 0; j < n; j++)
                arr->data[j] = va->array->data[j] * vb->array->data[j];
            if (am_tape_is_active())
                am_tape_record(arr, AM_OP_MUL, tape_ensure_entry(va->array), tape_ensure_entry(vb->array), 0);
            return arr;
        }
        return NULL;
    }

    // scale(a, scalar) — scalar multiplication
    if (strcasecmp(fname, "scale") == 0 && nargs >= 2) {
        AML_Var* va = resolve_var_full(ctx, arg_strs[0]);
        float scalar = aml_eval(ctx, arg_strs[1]);
        if (va && va->type == AML_TYPE_ARRAY && va->array) {
            AM_Array* arr = am_array_new(va->array->len);
            if (!arr) return NULL;
            for (int j = 0; j < va->array->len; j++)
                arr->data[j] = va->array->data[j] * scalar;
            if (am_tape_is_active())
                am_tape_record(arr, AM_OP_SCALE, tape_ensure_entry(va->array), -1, scalar);
            return arr;
        }
        return NULL;
    }

    // ── Phase 2: Matrix/Tensor operations ──

    // matrix(rows, cols, std) — create matrix with random normal init
    if (strcasecmp(fname, "matrix") == 0 && nargs >= 2) {
        int rows = (int)aml_eval(ctx, arg_strs[0]);
        int cols = (int)aml_eval(ctx, arg_strs[1]);
        float std = (nargs >= 3) ? aml_eval(ctx, arg_strs[2]) : 0.08f;
        AM_Array* arr = am_matrix_new(rows, cols);
        if (!arr) return NULL;
        for (int j = 0; j < arr->len; j += 2) {
            float u1 = ((float)rand() / (float)RAND_MAX) * 0.9998f + 0.0001f;
            float u2 = ((float)rand() / (float)RAND_MAX);
            float r = sqrtf(-2.0f * logf(u1));
            arr->data[j] = r * cosf(2.0f * 3.14159265f * u2) * std;
            if (j + 1 < arr->len)
                arr->data[j + 1] = r * sinf(2.0f * 3.14159265f * u2) * std;
        }
        return arr;
    }

    // matrix_zeros(rows, cols) — create zero-initialized matrix
    if (strcasecmp(fname, "matrix_zeros") == 0 && nargs >= 2) {
        int rows = (int)aml_eval(ctx, arg_strs[0]);
        int cols = (int)aml_eval(ctx, arg_strs[1]);
        return am_matrix_new(rows, cols);
    }

    // matvec(W, x) — matrix × vector → vector
    if (strcasecmp(fname, "matvec") == 0 && nargs >= 2) {
        AML_Var* vw = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vx = resolve_var_full(ctx, arg_strs[1]);
        if (vw && vw->type == AML_TYPE_ARRAY && vw->array && vw->array->rows > 0 &&
            vx && vx->type == AML_TYPE_ARRAY && vx->array) {
            int rows = vw->array->rows;
            int cols = vw->array->cols;
            if (cols != vx->array->len) return NULL;
            AM_Array* out = am_array_new(rows);
            if (!out) return NULL;
#ifdef USE_BLAS
            cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols,
                        1.0f, vw->array->data, cols, vx->array->data, 1,
                        0.0f, out->data, 1);
#else
            for (int i = 0; i < rows; i++) {
                float s = 0;
                for (int j = 0; j < cols; j++)
                    s += vw->array->data[i * cols + j] * vx->array->data[j];
                out->data[i] = s;
            }
#endif
            if (am_tape_is_active())
                am_tape_record(out, AM_OP_MATVEC, tape_ensure_entry(vw->array), tape_ensure_entry(vx->array), 0);
            return out;
        }
        return NULL;
    }

    // matmul(A, B) — matrix × matrix → matrix
    if (strcasecmp(fname, "matmul") == 0 && nargs >= 2) {
        AML_Var* va = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vb = resolve_var_full(ctx, arg_strs[1]);
        if (va && va->type == AML_TYPE_ARRAY && va->array && va->array->rows > 0 &&
            vb && vb->type == AML_TYPE_ARRAY && vb->array && vb->array->rows > 0) {
            int m = va->array->rows, k = va->array->cols;
            int k2 = vb->array->rows, n = vb->array->cols;
            if (k != k2) return NULL;
            AM_Array* out = am_matrix_new(m, n);
            if (!out) return NULL;
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0f,
                        va->array->data, k, vb->array->data, n,
                        0.0f, out->data, n);
#else
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) {
                    float s = 0;
                    for (int p = 0; p < k; p++)
                        s += va->array->data[i * k + p] * vb->array->data[p * n + j];
                    out->data[i * n + j] = s;
                }
#endif
            return out;
        }
        return NULL;
    }

    // softmax(x) — softmax over 1D array
    if (strcasecmp(fname, "softmax") == 0 && nargs >= 1) {
        AML_Var* vx = resolve_var_full(ctx, arg_strs[0]);
        if (vx && vx->type == AML_TYPE_ARRAY && vx->array) {
            int n = vx->array->len;
            AM_Array* out = am_array_new(n);
            if (!out) return NULL;
            // Find max for numerical stability
            float mx = vx->array->data[0];
            for (int j = 1; j < n; j++)
                if (vx->array->data[j] > mx) mx = vx->array->data[j];
            float sum = 0;
            for (int j = 0; j < n; j++) {
                out->data[j] = expf(vx->array->data[j] - mx);
                sum += out->data[j];
            }
            if (sum > 0) for (int j = 0; j < n; j++) out->data[j] /= sum;
            if (am_tape_is_active())
                am_tape_record(out, AM_OP_SOFTMAX, tape_ensure_entry(vx->array), -1, 0);
            return out;
        }
        return NULL;
    }

    // rmsnorm(x) — RMS normalization
    if (strcasecmp(fname, "rmsnorm") == 0 && nargs >= 1) {
        AML_Var* vx = resolve_var_full(ctx, arg_strs[0]);
        if (vx && vx->type == AML_TYPE_ARRAY && vx->array) {
            int n = vx->array->len;
            AM_Array* out = am_array_new(n);
            if (!out) return NULL;
            float ss = 0;
            for (int j = 0; j < n; j++) ss += vx->array->data[j] * vx->array->data[j];
            float rms = sqrtf(ss / n + 1e-6f);
            for (int j = 0; j < n; j++) out->data[j] = vx->array->data[j] / rms;
            if (am_tape_is_active())
                am_tape_record(out, AM_OP_RMSNORM, tape_ensure_entry(vx->array), -1, 0);
            return out;
        }
        return NULL;
    }

    // silu(x) — SiLU/Swish activation: x * sigmoid(x)
    if (strcasecmp(fname, "silu") == 0 && nargs >= 1) {
        AML_Var* vx = resolve_var_full(ctx, arg_strs[0]);
        AM_Array* input_arr = NULL;
        int owns_input = 0;  // 1 if we created input_arr via recursive call
        if (vx && vx->type == AML_TYPE_ARRAY && vx->array) {
            input_arr = vx->array;
        } else {
            // Try recursive evaluation: silu(seq_matvec(...)) etc.
            input_arr = aml_try_array_expr(ctx, arg_strs[0]);
            if (input_arr) owns_input = 1;
        }
        if (input_arr) {
            int n = input_arr->len;
            AM_Array* out = am_array_new(n);
            if (!out) { if (owns_input) am_array_free(input_arr); return NULL; }
            for (int j = 0; j < n; j++) {
                float x = input_arr->data[j];
                out->data[j] = x / (1.0f + expf(-x));
            }
            if (am_tape_is_active())
                am_tape_record(out, AM_OP_SILU, tape_ensure_entry(input_arr), -1, 0);
            // Don't free input_arr even if owns_input — tape may reference it
            return out;
        }
        return NULL;
    }

    // relu(x) — ReLU activation
    if (strcasecmp(fname, "relu") == 0 && nargs >= 1) {
        AML_Var* vx = resolve_var_full(ctx, arg_strs[0]);
        AM_Array* input_arr = NULL;
        if (vx && vx->type == AML_TYPE_ARRAY && vx->array) {
            input_arr = vx->array;
        } else {
            input_arr = aml_try_array_expr(ctx, arg_strs[0]);
        }
        if (input_arr) {
            int n = input_arr->len;
            AM_Array* out = am_array_new(n);
            if (!out) return NULL;
            for (int j = 0; j < n; j++)
                out->data[j] = input_arr->data[j] > 0 ? input_arr->data[j] : 0;
            return out;
        }
        return NULL;
    }

    // ── Phase 3: Autograd operations ──

    // cross_entropy(logits, target_idx) — cross-entropy loss (returns 1-element array)
    if (strcasecmp(fname, "cross_entropy") == 0 && nargs >= 2) {
        AML_Var* vl = resolve_var_full(ctx, arg_strs[0]);
        int target = (int)aml_eval(ctx, arg_strs[1]);
        if (vl && vl->type == AML_TYPE_ARRAY && vl->array) {
            int n = vl->array->len;
            if (target < 0 || target >= n) return NULL;
            // Compute softmax
            float mx = vl->array->data[0];
            for (int j = 1; j < n; j++)
                if (vl->array->data[j] > mx) mx = vl->array->data[j];
            float sum = 0;
            for (int j = 0; j < n; j++)
                sum += expf(vl->array->data[j] - mx);
            float log_softmax = vl->array->data[target] - mx - logf(sum);
            AM_Array* out = am_array_new(1);
            if (!out) return NULL;
            out->data[0] = -log_softmax;
            if (am_tape_is_active())
                am_tape_record(out, AM_OP_CROSS_ENT, tape_ensure_entry(vl->array), -1, (float)target);
            return out;
        }
        return NULL;
    }

    // embedding_lookup(wte, token_id) — extract row from embedding matrix (alias for row with tape)
    if (strcasecmp(fname, "embedding_lookup") == 0 && nargs >= 2) {
        AML_Var* vm = resolve_var_full(ctx, arg_strs[0]);
        int token_id = (int)aml_eval(ctx, arg_strs[1]);
        if (vm && vm->type == AML_TYPE_ARRAY && vm->array && vm->array->rows > 0) {
            if (token_id < 0 || token_id >= vm->array->rows) return NULL;
            int cols = vm->array->cols;
            AM_Array* out = am_array_new(cols);
            if (!out) return NULL;
            memcpy(out->data, vm->array->data + token_id * cols, cols * sizeof(float));
            if (am_tape_is_active())
                am_tape_record(out, AM_OP_EMB_LOOKUP, tape_ensure_entry(vm->array), -1, (float)token_id);
            return out;
        }
        return NULL;
    }

    // row(M, i) — extract row i from matrix M as 1D array
    if (strcasecmp(fname, "row") == 0 && nargs >= 2) {
        AML_Var* vm = resolve_var_full(ctx, arg_strs[0]);
        int ri = (int)aml_eval(ctx, arg_strs[1]);
        if (vm && vm->type == AML_TYPE_ARRAY && vm->array && vm->array->rows > 0) {
            if (ri < 0 || ri >= vm->array->rows) return NULL;
            int cols = vm->array->cols;
            AM_Array* out = am_array_new(cols);
            if (!out) return NULL;
            memcpy(out->data, vm->array->data + ri * cols, cols * sizeof(float));
            return out;
        }
        return NULL;
    }

    // ── Phase 5: Sequence-level transformer operations ──

    // seq_embed(wte, wpe, tokens, T) — embed a sequence of T tokens
    // wte: matrix[vocab_size × D], wpe: matrix[seq_len × D], tokens: array[T] of float token IDs
    // Returns: array[T*D] — concatenated embeddings for each position
    if (strcasecmp(fname, "seq_embed") == 0 && nargs >= 4) {
        AML_Var* vwte = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vwpe = resolve_var_full(ctx, arg_strs[1]);
        AML_Var* vtok = resolve_var_full(ctx, arg_strs[2]);
        int T = (int)aml_eval(ctx, arg_strs[3]);
        if (vwte && vwte->type == AML_TYPE_ARRAY && vwte->array && vwte->array->rows > 0 &&
            vwpe && vwpe->type == AML_TYPE_ARRAY && vwpe->array && vwpe->array->rows > 0 &&
            vtok && vtok->type == AML_TYPE_ARRAY && vtok->array && T > 0) {
            int D = vwte->array->cols;
            if (vwpe->array->cols != D) return NULL;
            if (T > vtok->array->len) T = vtok->array->len;
            AM_Array* out = am_array_new(T * D);
            if (!out) return NULL;
            for (int t = 0; t < T; t++) {
                int tok = (int)vtok->array->data[t];
                if (tok < 0) tok = 0;
                if (tok >= vwte->array->rows) tok = vwte->array->rows - 1;
                int pos = t < vwpe->array->rows ? t : vwpe->array->rows - 1;
                for (int d = 0; d < D; d++)
                    out->data[t * D + d] = vwte->array->data[tok * D + d] + vwpe->array->data[pos * D + d];
            }
            if (am_tape_is_active())
                am_tape_record3(out, AM_OP_SEQ_EMBED,
                    tape_ensure_entry(vwte->array), tape_ensure_entry(vwpe->array),
                    tape_ensure_entry(vtok->array), (float)T, (float)D);
            return out;
        }
        return NULL;
    }

    // seq_matvec(W, X, T) — apply W to each of T vectors in X
    // W: matrix[out_dim × in_dim], X: array[T*in_dim]
    // Returns: array[T*out_dim]
    if (strcasecmp(fname, "seq_matvec") == 0 && nargs >= 3) {
        AML_Var* vw = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vx = resolve_var_full(ctx, arg_strs[1]);
        int T = (int)aml_eval(ctx, arg_strs[2]);
        if (vw && vw->type == AML_TYPE_ARRAY && vw->array && vw->array->rows > 0 &&
            vx && vx->type == AML_TYPE_ARRAY && vx->array && T > 0) {
            int out_dim = vw->array->rows;
            int in_dim = vw->array->cols;
            if (T * in_dim > vx->array->len) return NULL;
            AM_Array* out = am_array_new(T * out_dim);
            if (!out) return NULL;
            float* W = vw->array->data;
            float* X = vx->array->data;
            float* Y = out->data;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(T * out_dim > 4096)
            #endif
            for (int t = 0; t < T; t++) {
                float* x_t = X + t * in_dim;
                float* y_t = Y + t * out_dim;
                for (int i = 0; i < out_dim; i++) {
                    float s = 0;
                    for (int j = 0; j < in_dim; j++)
                        s += W[i * in_dim + j] * x_t[j];
                    y_t[i] = s;
                }
            }
            if (am_tape_is_active())
                am_tape_record3(out, AM_OP_SEQ_MATVEC,
                    tape_ensure_entry(vw->array), tape_ensure_entry(vx->array), -1,
                    (float)T, 0);
            return out;
        }
        return NULL;
    }

    // seq_rmsnorm(X, T, D) — RMSNorm each D-sized chunk of X independently
    // X: array[T*D], returns array[T*D]
    if (strcasecmp(fname, "seq_rmsnorm") == 0 && nargs >= 3) {
        AML_Var* vx = resolve_var_full(ctx, arg_strs[0]);
        int T = (int)aml_eval(ctx, arg_strs[1]);
        int D = (int)aml_eval(ctx, arg_strs[2]);
        if (vx && vx->type == AML_TYPE_ARRAY && vx->array && T > 0 && D > 0) {
            if (T * D > vx->array->len) return NULL;
            AM_Array* out = am_array_new(T * D);
            if (!out) return NULL;
            float* Xr = vx->array->data;
            float* Or = out->data;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(T > 32)
            #endif
            for (int t = 0; t < T; t++) {
                float* x_t = Xr + t * D;
                float* o_t = Or + t * D;
                float ss = 0;
                for (int d = 0; d < D; d++) ss += x_t[d] * x_t[d];
                float rms = sqrtf(ss / D + 1e-6f);
                for (int d = 0; d < D; d++) o_t[d] = x_t[d] / rms;
            }
            if (am_tape_is_active())
                am_tape_record3(out, AM_OP_SEQ_RMSNORM,
                    tape_ensure_entry(vx->array), -1, -1, (float)T, (float)D);
            return out;
        }
        return NULL;
    }

    // causal_attention(Q, K, V, T, D) — single-head causal self-attention
    // Q, K, V: array[T*D], returns array[T*D]
    // For each position i, attends to positions 0..i with softmax
    if (strcasecmp(fname, "causal_attention") == 0 && nargs >= 5) {
        AML_Var* vq = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vk = resolve_var_full(ctx, arg_strs[1]);
        AML_Var* vv = resolve_var_full(ctx, arg_strs[2]);
        int T = (int)aml_eval(ctx, arg_strs[3]);
        int D = (int)aml_eval(ctx, arg_strs[4]);
        if (vq && vq->type == AML_TYPE_ARRAY && vq->array &&
            vk && vk->type == AML_TYPE_ARRAY && vk->array &&
            vv && vv->type == AML_TYPE_ARRAY && vv->array && T > 0 && D > 0) {
            if (T * D > vq->array->len || T * D > vk->array->len || T * D > vv->array->len)
                return NULL;
            float scale = 1.0f / sqrtf((float)D);
            AM_Array* out = am_array_new(T * D);
            if (!out) return NULL;
            // For each query position
            for (int i = 0; i < T; i++) {
                float* qi = vq->array->data + i * D;
                // Compute attention scores for positions 0..i
                float* scores = (float*)calloc(i + 1, sizeof(float));
                if (!scores) { am_array_free(out); return NULL; }
                float mx = -1e30f;
                for (int j = 0; j <= i; j++) {
                    float* kj = vk->array->data + j * D;
                    float dot = 0;
                    for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                    scores[j] = dot * scale;
                    if (scores[j] > mx) mx = scores[j];
                }
                // Softmax
                float sum = 0;
                for (int j = 0; j <= i; j++) {
                    scores[j] = expf(scores[j] - mx);
                    sum += scores[j];
                }
                if (sum > 0) for (int j = 0; j <= i; j++) scores[j] /= sum;
                // Weighted sum of V
                float* oi = out->data + i * D;
                for (int d = 0; d < D; d++) oi[d] = 0;
                for (int j = 0; j <= i; j++) {
                    float* vj = vv->array->data + j * D;
                    for (int d = 0; d < D; d++) oi[d] += scores[j] * vj[d];
                }
                free(scores);
            }
            if (am_tape_is_active())
                am_tape_record3(out, AM_OP_CAUSAL_ATTN,
                    tape_ensure_entry(vq->array), tape_ensure_entry(vk->array),
                    tape_ensure_entry(vv->array), (float)T, (float)D);
            return out;
        }
        return NULL;
    }

    // multi_head_attention(Q, K, V, T, D, n_heads) — multi-head causal self-attention
    // Q, K, V: array[T*D], splits D into n_heads heads of head_dim = D/n_heads
    // Returns: array[T*D]
    if (strcasecmp(fname, "multi_head_attention") == 0 && nargs >= 6) {
        AML_Var* vq = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vk = resolve_var_full(ctx, arg_strs[1]);
        AML_Var* vv = resolve_var_full(ctx, arg_strs[2]);
        int T = (int)aml_eval(ctx, arg_strs[3]);
        int D = (int)aml_eval(ctx, arg_strs[4]);
        int n_heads = (int)aml_eval(ctx, arg_strs[5]);
        if (vq && vq->type == AML_TYPE_ARRAY && vq->array &&
            vk && vk->type == AML_TYPE_ARRAY && vk->array &&
            vv && vv->type == AML_TYPE_ARRAY && vv->array &&
            T > 0 && D > 0 && n_heads > 0 && (D % n_heads) == 0) {
            if (T * D > vq->array->len || T * D > vk->array->len || T * D > vv->array->len)
                return NULL;
            int head_dim = D / n_heads;
            float scale = 1.0f / sqrtf((float)head_dim);
            AM_Array* out = am_array_new(T * D);
            if (!out) return NULL;
            float* Qd = vq->array->data;
            float* Kd = vk->array->data;
            float* Vd = vv->array->data;
            float* Od = out->data;
            #ifdef _OPENMP
            #pragma omp parallel if(n_heads >= 2)
            {
            float* scores_buf = (float*)malloc(T * sizeof(float));
            #pragma omp for schedule(static) collapse(2)
            #else
            float* scores_buf = (float*)malloc(T * sizeof(float));
            #endif
            for (int h = 0; h < n_heads; h++) {
                for (int i = 0; i < T; i++) {
                    int ho = h * head_dim;
                    float* qi = Qd + i * D + ho;
                    float mx = -1e30f;
                    for (int j = 0; j <= i; j++) {
                        float* kj = Kd + j * D + ho;
                        float dot = 0;
                        for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                        scores_buf[j] = dot * scale;
                        if (scores_buf[j] > mx) mx = scores_buf[j];
                    }
                    float sum = 0;
                    for (int j = 0; j <= i; j++) {
                        scores_buf[j] = expf(scores_buf[j] - mx);
                        sum += scores_buf[j];
                    }
                    if (sum > 0) for (int j = 0; j <= i; j++) scores_buf[j] /= sum;
                    float* oi = Od + i * D + ho;
                    for (int d = 0; d < head_dim; d++) oi[d] = 0;
                    for (int j = 0; j <= i; j++) {
                        float* vj = Vd + j * D + ho;
                        for (int d = 0; d < head_dim; d++) oi[d] += scores_buf[j] * vj[d];
                    }
                }
            }
            #ifdef _OPENMP
            free(scores_buf);
            }
            #else
            free(scores_buf);
            #endif
            if (am_tape_is_active())
                am_tape_record3(out, AM_OP_MH_CAUSAL_ATTN,
                    tape_ensure_entry(vq->array), tape_ensure_entry(vk->array),
                    tape_ensure_entry(vv->array), (float)T, (float)head_dim);
            return out;
        }
        return NULL;
    }

    // seq_cross_entropy(logits, targets, T, V) — cross-entropy over T positions
    // logits: array[T*V], targets: array[T] of float token IDs
    // Returns: array[1] (mean loss over T positions)
    if (strcasecmp(fname, "seq_cross_entropy") == 0 && nargs >= 4) {
        AML_Var* vl = resolve_var_full(ctx, arg_strs[0]);
        AML_Var* vt = resolve_var_full(ctx, arg_strs[1]);
        int T = (int)aml_eval(ctx, arg_strs[2]);
        int V = (int)aml_eval(ctx, arg_strs[3]);
        if (vl && vl->type == AML_TYPE_ARRAY && vl->array &&
            vt && vt->type == AML_TYPE_ARRAY && vt->array && T > 0 && V > 0) {
            if (T * V > vl->array->len || T > vt->array->len) return NULL;
            AM_Array* out = am_array_new(1);
            if (!out) return NULL;
            float total_loss = 0;
            for (int t = 0; t < T; t++) {
                float* logits_t = vl->array->data + t * V;
                int target = (int)vt->array->data[t];
                if (target < 0 || target >= V) target = 0;
                // Softmax + log-loss
                float mx = logits_t[0];
                for (int j = 1; j < V; j++)
                    if (logits_t[j] > mx) mx = logits_t[j];
                float sum = 0;
                for (int j = 0; j < V; j++) sum += expf(logits_t[j] - mx);
                float log_prob = (logits_t[target] - mx) - logf(sum + 1e-10f);
                total_loss -= log_prob;
            }
            out->data[0] = total_loss / T;
            if (am_tape_is_active())
                am_tape_record3(out, AM_OP_SEQ_CROSSENT,
                    tape_ensure_entry(vl->array), tape_ensure_entry(vt->array), -1,
                    (float)T, (float)V);
            return out;
        }
        return NULL;
    }

    // NOTE: user-defined functions that return arrays are handled at
    // the assignment level (aml_exec_line), not here. aml_try_array_expr
    // only handles known array-producing builtins to avoid accidentally
    // eating scalar returns from user functions.
    return NULL;
}

// Execute a single line in Level 2 context
static int aml_exec_line(AML_ExecCtx* ctx, int idx) {
    char* text = ctx->lines[idx].text;

    // v4.0: propagate return — if has_return is set, stop executing
    if (ctx->has_return) return ctx->nlines;

    // --- def: skip (already registered) ---
    if (strncmp(text, "def ", 4) == 0) {
        // skip body
        return aml_find_block_end(ctx->lines, ctx->nlines, idx);
    }

    // --- v4.0: return statement ---
    if (strncmp(text, "return ", 7) == 0 || strcmp(text, "return") == 0) {
        const char* rhs = text + 6;
        while (*rhs == ' ') rhs++;
        if (*rhs) {
            // Try array builtin expression first (zeros, randn, add, mul, scale, literal)
            AM_Array* arr = aml_try_array_expr(ctx, rhs);
            if (arr) {
                ctx->has_return = 1;
                ctx->return_type = AML_TYPE_ARRAY;
                ctx->return_value = 0;
                ctx->return_array = arr;
            } else {
                // Check if RHS is just an array variable name
                char rhs_name[AML_MAX_NAME] = {0};
                const char* rp = rhs;
                int ri = 0;
                while ((isalnum((unsigned char)*rp) || *rp == '_') && ri < AML_MAX_NAME - 1)
                    rhs_name[ri++] = *rp++;
                rhs_name[ri] = 0;
                while (*rp == ' ') rp++;
                if (ri > 0 && *rp == '\0') {
                    AML_Var* src = resolve_var_full(ctx, rhs_name);
                    if (src && src->type == AML_TYPE_ARRAY && src->array) {
                        ctx->has_return = 1;
                        ctx->return_type = AML_TYPE_ARRAY;
                        ctx->return_value = 0;
                        ctx->return_array = src->array; // refcount bumped in aml_call_func
                        return ctx->nlines;
                    }
                }
                // Scalar return (may call user functions via aml_eval)
                float val = aml_eval(ctx, rhs);
                // Check if a user function returned an array through eval
                if (ctx->has_return && ctx->return_array) {
                    // Already set by the function call, keep it
                } else {
                    ctx->has_return = 1;
                    ctx->return_type = AML_TYPE_FLOAT;
                    ctx->return_value = val;
                    ctx->return_array = NULL;
                }
            }
        } else {
            ctx->has_return = 1;
            ctx->return_value = 0;
            ctx->return_array = NULL;
        }
        return ctx->nlines; // stop block execution
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

        while (aml_eval(ctx, cond) != 0.0f && iterations < 10000 && !ctx->has_return) {
            aml_exec_block(ctx, idx + 1, body_end);
            iterations++;
        }
        return body_end;
    }

    // --- v4.0: SPAWN name: (async block) ---
#ifndef AM_ASYNC_DISABLED
    if (strncasecmp(text, "SPAWN ", 6) == 0) {
        // Parse: SPAWN name:
        char spawn_name[AM_SPAWN_NAME_LEN] = {0};
        const char* sp = text + 6;
        while (*sp == ' ') sp++;
        int ni = 0;
        while (*sp && *sp != ':' && *sp != ' ' && ni < AM_SPAWN_NAME_LEN - 1)
            spawn_name[ni++] = *sp++;
        spawn_name[ni] = 0;

        int body_end = aml_find_block_end(ctx->lines, ctx->nlines, idx);

        // Build script string from indented block
        // Calculate total size needed
        int total = 0;
        for (int bi = idx + 1; bi < body_end; bi++)
            total += (int)strlen(ctx->lines[bi].text) + 1; // +1 for newline
        total += 1; // null terminator

        char* script = (char*)malloc(total);
        if (script) {
            script[0] = 0;
            for (int bi = idx + 1; bi < body_end; bi++) {
                strcat(script, ctx->lines[bi].text);
                strcat(script, "\n");
            }
            am_spawn_launch(spawn_name, script);
            free(script);
        }

        return body_end;
    }
#endif // AM_ASYNC_DISABLED

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

    // --- v4.0: array element write: name[index] = expr ---
    {
        // Look for pattern: identifier[expr] = expr
        const char* bracket = strchr(text, '[');
        if (bracket) {
            const char* close_bracket = strchr(bracket, ']');
            if (close_bracket) {
                const char* eq_after = close_bracket + 1;
                while (*eq_after == ' ') eq_after++;
                if (*eq_after == '=' && eq_after[1] != '=') {
                    // Extract variable name
                    char varname[AML_MAX_NAME] = {0};
                    int ni = 0;
                    const char* p = text;
                    while (p < bracket && ni < AML_MAX_NAME - 1) {
                        if (!isspace((unsigned char)*p))
                            varname[ni++] = *p;
                        p++;
                    }
                    varname[ni] = 0;

                    if (ni > 0) {
                        // Evaluate index
                        char idx_expr[AML_MAX_LINE_LEN] = {0};
                        int ie = 0;
                        const char* ip = bracket + 1;
                        while (ip < close_bracket && ie < AML_MAX_LINE_LEN - 1)
                            idx_expr[ie++] = *ip++;
                        idx_expr[ie] = 0;
                        int index = (int)aml_eval(ctx, idx_expr);

                        // Evaluate value
                        float val = aml_eval(ctx, eq_after + 1);

                        // Find the array variable and write to it
                        AML_Var* var = resolve_var_full(ctx, varname);
                        if (var && var->type == AML_TYPE_ARRAY && var->array) {
                            if (index >= 0 && index < var->array->len)
                                var->array->data[index] = val;
                        }
                        return idx + 1;
                    }
                }
            }
        }
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
                // v4.0: try array expression first (builtins only)
                const char* rhs = eq + 1;
                while (*rhs == ' ') rhs++;
                AM_Array* arr = aml_try_array_expr(ctx, rhs);
                if (arr) {
                    AML_Symtab* tab = (ctx->call_depth > 0)
                        ? &ctx->locals[ctx->call_depth - 1]
                        : &ctx->globals;
                    symtab_set_array(tab, varname, arr);
                    return idx + 1;
                }

                // v4.0: also check if RHS is just a variable name holding an array
                {
                    char rhs_name[AML_MAX_NAME] = {0};
                    const char* rp = rhs;
                    int ri = 0;
                    while ((isalnum((unsigned char)*rp) || *rp == '_') && ri < AML_MAX_NAME - 1)
                        rhs_name[ri++] = *rp++;
                    rhs_name[ri] = 0;
                    while (*rp == ' ') rp++;
                    if (ri > 0 && *rp == '\0') {
                        // RHS is a bare identifier — check if it's an array variable
                        AML_Var* src = resolve_var_full(ctx, rhs_name);
                        if (src && src->type == AML_TYPE_ARRAY && src->array) {
                            AM_Array* clone = am_array_clone(src->array);
                            if (clone) {
                                AML_Symtab* tab = (ctx->call_depth > 0)
                                    ? &ctx->locals[ctx->call_depth - 1]
                                    : &ctx->globals;
                                symtab_set_array(tab, varname, clone);
                                return idx + 1;
                            }
                        }
                    }
                }

                float val = aml_eval(ctx, eq + 1);

                // v4.0: check if a user function returned an array
                if (ctx->has_return && ctx->return_array) {
                    AML_Symtab* tab = (ctx->call_depth > 0)
                        ? &ctx->locals[ctx->call_depth - 1]
                        : &ctx->globals;
                    symtab_set_array(tab, varname, ctx->return_array);
                    ctx->has_return = 0;
                    ctx->return_array = NULL;
                    return idx + 1;
                }
                ctx->has_return = 0;

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
    while (i < end && i < ctx->nlines && !ctx->has_return) {
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

    // v4.0: restore persistent globals if enabled
    persistent_restore(&ctx.globals);

    // register built-in functions (native AML, not external bindings)
    aml_register_builtins(&ctx);

    // first pass: register user-defined function definitions
    aml_register_funcs(&ctx);

    // second pass: execute top-level block
    aml_exec_block(&ctx, 0, nlines);

    // v4.0: save globals to persistent storage, then clean up
    persistent_save(&ctx.globals);
    symtab_clear_arrays(&ctx.globals);

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
// BLAS path: cblas_sgemv × 2 (matrix-vector multiply)
void am_apply_delta(float* out, const float* A, const float* B,
                    const float* x, int out_dim, int in_dim, int rank,
                    float alpha) {
    if (!out || !A || !B || !x || alpha == 0.0f) return;
    if (rank > 128) rank = 128;

    float temp[128];

#ifdef USE_BLAS
    // temp = B @ x  (BLAS: sgemv, rank × in_dim @ in_dim × 1 → rank × 1)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rank, in_dim,
                1.0f, B, in_dim, x, 1, 0.0f, temp, 1);
    // out += alpha * A @ temp  (BLAS: sgemv, out_dim × rank @ rank × 1 → out_dim × 1)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, rank,
                alpha, A, rank, temp, 1, 1.0f, out, 1);
#else
    // Scalar fallback: portable, no dependencies
    for (int r = 0; r < rank; r++) {
        temp[r] = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            temp[r] += B[r * in_dim + j] * x[j];
        }
    }
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int r = 0; r < rank; r++) {
            sum += A[i * rank + r] * temp[r];
        }
        out[i] += alpha * sum;
    }
#endif
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
// A: [in_dim × rank], B: [rank × out_dim]
// x: input hidden state [in_dim], dy: output gradient proxy [out_dim]
// signal: teaching signal (positive = reinforce, negative = suppress)
// BLAS path: cblas_sger × 2 (rank-1 outer product updates)
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

#ifdef USE_BLAS
    // A += (lr * g) * x ⊗ u  (BLAS: rank-1 update, in_dim × rank)
    cblas_sger(CblasRowMajor, in_dim, rank, lr * g, x, 1, u, 1, A, rank);
    // B += (lr * g) * u ⊗ dy  (BLAS: rank-1 update, rank × out_dim)
    cblas_sger(CblasRowMajor, rank, out_dim, lr * g, u, 1, dy, 1, B, out_dim);
#else
    // Scalar fallback: portable, no dependencies
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
#endif

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
// LILITH — I/O subsystem (named pipes for data infrastructure)
//
// "Та, которая была до Евы."
// Infra that existed before the human intervened.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef AM_IO_DISABLED

// Find pipe by logical name. Returns index or -1.
static int pipe_find(const char* name) {
    for (int i = 0; i < g_pipe_count; i++) {
        if (g_pipes[i].active && strcmp(g_pipes[i].name, name) == 0)
            return i;
    }
    return -1;
}

// Find first free pipe slot. Returns index or -1.
static int pipe_find_free(void) {
    // Reuse inactive slots first
    for (int i = 0; i < g_pipe_count; i++) {
        if (!g_pipes[i].active) return i;
    }
    if (g_pipe_count < AM_MAX_PIPES) return g_pipe_count++;
    return -1;
}

int am_pipe_create(const char* path) {
    if (!path || !path[0]) return -1;

    // Remove existing file/pipe at path first (idempotent)
    unlink(path);

    if (mkfifo(path, 0666) != 0) {
        // EEXIST is OK — pipe already exists
        if (errno != EEXIST) {
            printf("[LILITH] mkfifo(%s) failed: %s\n", path, strerror(errno));
            return -1;
        }
    }
    return 0;
}

int am_pipe_open(const char* name, const char* path, int mode) {
    if (!name || !name[0] || !path || !path[0]) return -1;

    // Check if already open with this name
    int existing = pipe_find(name);
    if (existing >= 0) {
        printf("[LILITH] pipe '%s' already open\n", name);
        return existing;
    }

    int slot = pipe_find_free();
    if (slot < 0) {
        printf("[LILITH] max pipes reached (%d)\n", AM_MAX_PIPES);
        return -1;
    }

    int flags;
    if (mode == AM_PIPE_MODE_READ) {
        flags = O_RDONLY | O_NONBLOCK;
    } else {
        // O_WRONLY + O_NONBLOCK on FIFO returns ENXIO if no reader yet.
        // Use O_RDWR to avoid blocking — works on both macOS and Linux.
        flags = O_RDWR | O_NONBLOCK;
    }

    int fd = open(path, flags);
    if (fd < 0) {
        printf("[LILITH] open(%s) failed: %s\n", path, strerror(errno));
        return -1;
    }

    snprintf(g_pipes[slot].name, AM_PIPE_NAME_LEN, "%.31s", name);
    snprintf(g_pipes[slot].path, AM_PIPE_PATH_LEN, "%.255s", path);
    g_pipes[slot].fd = fd;
    g_pipes[slot].mode = mode;
    g_pipes[slot].active = 1;

    printf("[LILITH] pipe '%s' opened (%s, %s)\n", name, path,
           mode == AM_PIPE_MODE_READ ? "READ" : "WRITE");
    return slot;
}

int am_pipe_write(const char* name, const char* message) {
    if (!name || !message) return -1;

    int idx = pipe_find(name);
    if (idx < 0) {
        printf("[LILITH] pipe '%s' not found\n", name);
        return -1;
    }
    if (!g_pipes[idx].active || g_pipes[idx].fd < 0) return -1;

    // Append newline as message delimiter
    int mlen = (int)strlen(message);
    char buf[AM_PIPE_BUF_SIZE];
    if (mlen + 2 > AM_PIPE_BUF_SIZE) mlen = AM_PIPE_BUF_SIZE - 2;
    memcpy(buf, message, mlen);
    buf[mlen] = '\n';
    buf[mlen + 1] = 0;

    ssize_t n = write(g_pipes[idx].fd, buf, mlen + 1);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            printf("[LILITH] pipe '%s' write: no reader (EAGAIN)\n", name);
            return 0;
        }
        printf("[LILITH] pipe '%s' write error: %s\n", name, strerror(errno));
        return -1;
    }
    return (int)n;
}

int am_pipe_read(const char* name, char* buf, int bufsize) {
    if (!name || !buf || bufsize <= 0) return -1;

    int idx = pipe_find(name);
    if (idx < 0) {
        printf("[LILITH] pipe '%s' not found\n", name);
        return -1;
    }
    if (!g_pipes[idx].active || g_pipes[idx].fd < 0) return -1;

    ssize_t n = read(g_pipes[idx].fd, buf, bufsize - 1);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            buf[0] = 0;
            return 0;  // nothing available (non-blocking)
        }
        printf("[LILITH] pipe '%s' read error: %s\n", name, strerror(errno));
        buf[0] = 0;
        return -1;
    }
    if (n == 0) {
        buf[0] = 0;
        return 0;  // EOF / no data
    }

    buf[n] = 0;
    // Strip trailing newline
    if (n > 0 && buf[n - 1] == '\n') buf[--n] = 0;

    // Parse first number found anywhere in response into g_pipe_last_value
    // Scans forward until a digit or sign-before-digit is found
    {
        const char* p = buf;
        while (*p) {
            if (isdigit((unsigned char)*p) ||
                ((*p == '-' || *p == '+' || *p == '.') && isdigit((unsigned char)p[1]))) {
                char* endptr = NULL;
                float val = strtof(p, &endptr);
                if (endptr != p) {
                    g_pipe_last_value = val;
                    break;
                }
            }
            p++;
        }
    }

    return (int)n;
}

void am_pipe_close(const char* name) {
    int idx = pipe_find(name);
    if (idx < 0) return;

    if (g_pipes[idx].fd >= 0) {
        close(g_pipes[idx].fd);
    }
    printf("[LILITH] pipe '%s' closed\n", g_pipes[idx].name);
    g_pipes[idx].fd = -1;
    g_pipes[idx].active = 0;
    g_pipes[idx].name[0] = 0;
}

void am_pipe_close_all(void) {
    for (int i = 0; i < g_pipe_count; i++) {
        if (g_pipes[i].active) {
            if (g_pipes[i].fd >= 0) close(g_pipes[i].fd);
            g_pipes[i].fd = -1;
            g_pipes[i].active = 0;
        }
    }
    g_pipe_count = 0;
    printf("[LILITH] all pipes closed\n");
}

float am_pipe_last_value(void) { return g_pipe_last_value; }

int am_pipe_count(void) {
    int count = 0;
    for (int i = 0; i < g_pipe_count; i++) {
        if (g_pipes[i].active) count++;
    }
    return count;
}

const AM_Pipe* am_pipe_get(int idx) {
    if (idx < 0 || idx >= g_pipe_count) return NULL;
    if (!g_pipes[idx].active) return NULL;
    return &g_pipes[idx];
}

#endif // AM_IO_DISABLED

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

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIC NET — weightless neural network in C
//
// Layer 1: Fourier decomposition of entropy history
// Layer 2: Correlation matrix (pairwise gamma cosines = the "weights")
// Layer 3: Phase aggregation (resonance + harmonics → steering refinement)
//
// No trainable weights. No backprop. Just harmonic resonance.
// Evolved in molequla, ported to core.
// ═══════════════════════════════════════════════════════════════════════════════

static struct {
    /* Entropy history (circular buffer) */
    float entropy_history[AM_HARMONIC_MAX_HISTORY];
    int   history_len;
    int   history_pos;

    /* Organism gammas for this step */
    float gammas[AM_HARMONIC_MAX_ORGANISMS][AM_HARMONIC_GAMMA_DIM];
    float org_entropy[AM_HARMONIC_MAX_ORGANISMS];
    int   n_organisms;
} HN;

void am_harmonic_init(void) {
    memset(&HN, 0, sizeof(HN));
}

void am_harmonic_clear(void) {
    HN.n_organisms = 0;
}

void am_harmonic_push_entropy(float entropy) {
    HN.entropy_history[HN.history_pos] = entropy;
    HN.history_pos = (HN.history_pos + 1) % AM_HARMONIC_MAX_HISTORY;
    if (HN.history_len < AM_HARMONIC_MAX_HISTORY)
        HN.history_len++;
}

void am_harmonic_push_gamma(int id, const float *gamma, int dim, float entropy) {
    (void)id;
    if (HN.n_organisms >= AM_HARMONIC_MAX_ORGANISMS) return;
    int idx = HN.n_organisms++;
    int copy_dim = dim < AM_HARMONIC_GAMMA_DIM ? dim : AM_HARMONIC_GAMMA_DIM;
    memcpy(HN.gammas[idx], gamma, copy_dim * sizeof(float));
    /* Zero-pad if needed */
    for (int i = copy_dim; i < AM_HARMONIC_GAMMA_DIM; i++)
        HN.gammas[idx][i] = 0.0f;
    HN.org_entropy[idx] = entropy;
}

AM_HarmonicResult am_harmonic_forward(int step) {
    (void)step;
    AM_HarmonicResult r;
    memset(&r, 0, sizeof(r));
    r.n_organisms = HN.n_organisms;
    r.strength_mod = 0.3f;

    if (HN.n_organisms == 0) return r;

    int T = HN.history_len;

    /* ── Layer 1: Fourier decomposition of entropy history ── */
    if (T >= 4) {
        for (int k = 0; k < AM_HARMONIC_N_FREQ; k++) {
            float sum = 0.0f;
            for (int t = 0; t < T; t++) {
                int idx = (HN.history_pos - T + t + AM_HARMONIC_MAX_HISTORY) % AM_HARMONIC_MAX_HISTORY;
                float phase = 2.0f * 3.14159265f * (float)(k + 1) * (float)t / (float)T;
                sum += HN.entropy_history[idx] * sinf(phase);
            }
            r.harmonics[k] = sum / (float)T;
        }
    }

    /* ── Layer 2: Correlation matrix (pairwise gamma cosines) ── */
    int n = HN.n_organisms;

    /* Compute norms */
    float norms[AM_HARMONIC_MAX_ORGANISMS];
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int d = 0; d < AM_HARMONIC_GAMMA_DIM; d++)
            s += HN.gammas[i][d] * HN.gammas[i][d];
        norms[i] = sqrtf(s);
        if (norms[i] < 1e-8f) norms[i] = 1e-8f;
    }

    /* Pairwise cosines + phase resonance */
    float mean_ent = 0.0f;
    for (int i = 0; i < n; i++) mean_ent += HN.org_entropy[i];
    mean_ent /= (float)n;

    for (int i = 0; i < n; i++) {
        float res = 0.0f;
        float phase_i = HN.org_entropy[i] - mean_ent;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            /* Cosine similarity */
            float dot = 0.0f;
            for (int d = 0; d < AM_HARMONIC_GAMMA_DIM; d++)
                dot += HN.gammas[i][d] * HN.gammas[j][d];
            float cos_ij = dot / (norms[i] * norms[j]);

            /* Phase similarity */
            float phase_j = HN.org_entropy[j] - mean_ent;
            float phase_sim = expf(-fabsf(phase_i - phase_j));

            res += cos_ij * phase_sim;
        }
        if (n > 1) res /= (float)(n - 1);
        r.resonance[i] = res;
    }

    /* ── Layer 3: Output ── */
    /* Find dominant harmonic */
    float max_amp = 0.0f;
    r.dominant_freq = 0;
    if (T >= 4) {
        for (int k = 0; k < AM_HARMONIC_N_FREQ; k++) {
            float a = fabsf(r.harmonics[k]);
            if (a > max_amp) { max_amp = a; r.dominant_freq = k; }
        }
    }

    /* Confidence: more data = more confident */
    float conf_t = T < 16 ? (float)T / 16.0f : 1.0f;
    float conf_n = n < 4 ? (float)n / 4.0f : 1.0f;
    r.strength_mod = 0.3f + 0.7f * conf_t * conf_n;

    return r;
}

// ═══════════════════════════════════════════════════════════════════════════════
// METHOD — distributed cognition operator (C implementation)
//
// The field operator. Works on collective organism data, not individuals.
// Host pushes organism snapshots, METHOD computes awareness and steering.
// Evolved in molequla, ported to core.
// ═══════════════════════════════════════════════════════════════════════════════

static AM_MethodState M;

void am_method_init(void) {
    memset(&M, 0, sizeof(AM_MethodState));
}

void am_method_clear(void) {
    M.n_organisms = 0;
}

void am_method_push_organism(int id, float entropy, float syntropy,
                             float gamma_mag, float gamma_cos) {
    if (M.n_organisms >= AM_METHOD_MAX_ORGANISMS) return;
    AM_MethodOrganism* o = &M.organisms[M.n_organisms++];
    o->id = id;
    o->entropy = entropy;
    o->syntropy = syntropy;
    o->gamma_mag = gamma_mag;
    o->gamma_cos = gamma_cos;
}

float am_method_field_entropy(void) {
    if (M.n_organisms == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < M.n_organisms; i++)
        sum += M.organisms[i].entropy;
    return sum / (float)M.n_organisms;
}

float am_method_field_syntropy(void) {
    if (M.n_organisms == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < M.n_organisms; i++)
        sum += M.organisms[i].syntropy;
    return sum / (float)M.n_organisms;
}

float am_method_field_coherence(void) {
    if (M.n_organisms == 0) return 1.0f;
    if (M.n_organisms == 1) return 1.0f;

    // Mean gamma_cos across organisms (host-computed pairwise)
    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < M.n_organisms; i++) {
        if (M.organisms[i].gamma_mag > 1e-6f) {
            sum += M.organisms[i].gamma_cos;
            count++;
        }
    }
    return count > 0 ? sum / (float)count : 1.0f;
}

AM_MethodSteering am_method_step(float dt) {
    AM_MethodSteering s;
    memset(&s, 0, sizeof(s));

    s.n_organisms = M.n_organisms;
    M.step_count++;
    s.step = M.step_count;

    if (M.n_organisms == 0) {
        s.action = AM_METHOD_WAIT;
        return s;
    }

    float entropy = am_method_field_entropy();
    float syntropy = am_method_field_syntropy();
    float coherence = am_method_field_coherence();

    s.entropy = entropy;
    s.syntropy = syntropy;
    s.coherence = coherence;

    // Push to circular history buffer
    int pos = M.history_pos % AM_METHOD_HISTORY_LEN;
    M.entropy_history[pos] = entropy;
    M.coherence_history[pos] = coherence;
    M.history_pos++;
    if (M.history_len < AM_METHOD_HISTORY_LEN)
        M.history_len++;

    // Compute entropy trend (positive = organizing, negative = dissolving)
    float trend = 0.0f;
    if (M.history_len >= 4) {
        float recent = 0.0f, earlier = 0.0f;
        int rc = 0, ec = 0;
        for (int i = 0; i < M.history_len && i < 8; i++) {
            int idx = ((M.history_pos - 1 - i) % AM_METHOD_HISTORY_LEN + AM_METHOD_HISTORY_LEN) % AM_METHOD_HISTORY_LEN;
            if (i < 4) { recent += M.entropy_history[idx]; rc++; }
            else        { earlier += M.entropy_history[idx]; ec++; }
        }
        if (rc > 0 && ec > 0)
            trend = (earlier / (float)ec) - (recent / (float)rc);
    }
    s.trend = trend;

    // Find best organism (lowest entropy)
    int best_id = M.organisms[0].id;
    float best_entropy = M.organisms[0].entropy;
    for (int i = 1; i < M.n_organisms; i++) {
        if (M.organisms[i].entropy < best_entropy) {
            best_entropy = M.organisms[i].entropy;
            best_id = M.organisms[i].id;
        }
    }
    s.target_id = best_id;

    // Decide action
    if (coherence < 0.3f) {
        s.action = AM_METHOD_REALIGN;
        s.strength = 1.0f - coherence;
    } else if (trend > 0.05f) {
        s.action = AM_METHOD_AMPLIFY;
        s.strength = fminf(1.0f, trend * 5.0f);
    } else if (trend < -0.05f) {
        s.action = AM_METHOD_DAMPEN;
        s.strength = fminf(1.0f, fabsf(trend) * 5.0f);
    } else if (entropy > 2.0f) {
        s.action = AM_METHOD_GROUND;
        s.strength = fminf(1.0f, (entropy - 1.5f) * 0.5f);
    } else if (entropy < 0.5f) {
        s.action = AM_METHOD_EXPLORE;
        s.strength = fminf(1.0f, (1.0f - entropy) * 0.5f);
    } else {
        s.action = AM_METHOD_SUSTAIN;
        s.strength = 0.1f;
    }

    // Advance AML field physics
    am_step(dt);

    // Translate steering to AML state
    switch (s.action) {
        case AM_METHOD_DAMPEN:
            am_exec("PAIN 0.3");
            am_exec("VELOCITY WALK");
            break;
        case AM_METHOD_AMPLIFY:
            am_exec("VELOCITY RUN");
            am_exec("DESTINY 0.6");
            break;
        case AM_METHOD_GROUND:
            am_exec("ATTEND_FOCUS 0.9");
            am_exec("VELOCITY NOMOVE");
            break;
        case AM_METHOD_EXPLORE:
            am_exec("TUNNEL_CHANCE 0.3");
            am_exec("VELOCITY RUN");
            break;
        case AM_METHOD_REALIGN:
            am_exec("PAIN 0.5");
            am_exec("ATTEND_FOCUS 0.8");
            break;
        default:
            break;
    }

    return s;
}

AM_MethodState* am_method_get_state(void) {
    return &M;
}
