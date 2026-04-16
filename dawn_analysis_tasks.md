# DAWN-SRW 분석 확장 작업 지시서

## 맥락 (먼저 이해할 것)

- 대상 파일: `scripts/analysis/standalone/spatial_analysis.py` (약 4,680줄).
- 모델: v4.0.1 구조. 40M 기준 QK=1580 / V=2600 / Know=25200, d=384, d_route=128, L=12.
  - 모델 파일 경로는 `--model_file` CLI로 전달됨 (default `models.dawn_spatial_v3`).
  - **v4.0.1 실험용 모델 파일 경로 (`dawn_spatial_v401_exp` 등)는 MADST에게 확인**. 확인 전까지 default 경로 그대로 둠.
  - **40M 외에 400M 스케일에서도 동일 분석 가능해야 함**. 모든 pool 크기 (`n_qk`, `n_v`, `n_know`, `d_model`, `n_layers`, `d_route`)는 **반드시 `cfg`/`model_cfg`에서 읽어오기**. 숫자 하드코딩 금지.
- 목표: 후속 논문 §6.3, §7.1~§7.4, §8, Appendix B를 위한 분석 함수를 `spatial_analysis.py`에 추가.
- 패턴: 기존 `analyze_*(params, cfg, val_tokens, output_dir, ...)` 함수들과 동일한 시그니처, `_should_run('<key>')` dispatch, `_save_json` 저장.

## 재사용 가능한 기존 인프라 (반드시 먼저 파악)

파일 수정 시작 **전에** 아래를 먼저 읽어라. 중복 구현 절대 금지.

### 1. 모델 측 공용 헬퍼

- **`_mod.analysis_forward(params, model_cfg, input_ids, mode=...)`**
  - JIT-compilable. `mode='light'` (per-layer scalar/축소 텐서만) / `mode='full'` (gate 원본 [L,B,S,N] 포함).
  - 새 분석에서 forward 필요하면 이걸 JIT로 감싸서 써라. 별도 forward 새로 짜지 마.
  - 예: `jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids, mode='light'))`.
- **`_mod._srw_inference(h, scores_base, emb_n, tau, read, write)`** — SRW 블록 내부 gate/out 계산. Per-neuron 분해 필요할 때만 풀어서 쓰기.
- **`_mod._layer_norm`** — 재사용.

### 2. 분석 측 기존 함수 (읽어보고 중복 피하기)

| 함수 | 라인 | 역할 |
|---|---|---|
| `_run_layerwise_analysis` | 3278 | per-layer residual/attn_out/know_out/per-neuron stats 수집. §8 대부분이 이걸로 해결. |
| `analyze_rw_projection` | 2009 | cos(r, w) 일부 계산 있음. §7.2.1 기반. |
| `analyze_gate_distribution` | 1685 | gate mean/std/eff_N per layer. §7.1.1 기반. |
| `analyze_neuron_utilization` | 1859 | per-neuron usage frequency, active count. §7.1.2 기반. |
| `analyze_suppression` | 1545 | suppression infrastructure. Appendix B 기반. |
| `analyze_cross_domain_suppression` | 2337 | cross-domain prompts + suppression. Appendix B 기반. |
| `analyze_routing` | 589 | lax.scan 기반 배치 누적 통계 예시. |
| `analyze_validation` | 295 | JIT val loss 계산 예시. |

### 3. CLI 파라미터 (이미 존재, 새로 만들지 마)

`main()`의 argparse에 아래가 이미 있음. 새 분석은 이 값들을 함수에 전달하기만 하면 됨.

```
--checkpoint, --config, --val_data, --output, --model_file
--max_batches  (default 200)   # val 데이터 최대 배치 수
--batch_size   (default 32)    # 배치 크기
--n_batches    (default None)  # 개별 분석 n_batches 오버라이드 (None이면 함수 default)
--only, --skip                 # dispatch 필터
--r2_sentences (default 5000)  # R.2 POS 전용
--p6_samples   (default 100)   # P6 전용
--prompt, --max_new_tokens, --temperature  # generation 전용
```

**규칙**: 새 분석이 기존에 없는 고유 파라미터를 필요로 하면 `--<key>_<param>` 형태로 추가 (예: `--dom_supp_topk`, `--addit_n_tokens`). 기존 `--n_batches`, `--batch_size`로 커버 가능하면 추가하지 마.

## 작업 원칙 (반드시 지킬 것)

### API timeout 방지 — 이게 가장 중요

이전 시도에서 한 번에 여러 분석을 추가하려다 API timeout이 반복됐다. 원인: 4,680줄 파일을 반복해서 읽고/쓰면서 컨텍스트가 비대해짐.

1. **한 태스크 = 한 분석 = 한 커밋**. 한 태스크 끝나면 다음 태스크 시작 전 "현재까지 추가된 분석:" 하고 상태만 요약. 파일 전체를 다시 읽지 마.
2. **파일 전체 읽기 금지**. `view` 호출은 반드시 `view_range`로 100줄 이하. 예: `analyze_rw_projection` 내부 확인이면 라인 2009~2105만.
3. **한 태스크 안에서 `str_replace`는 3회 이하**. 더 필요하면 태스크를 쪼개라.
4. **새 함수는 파일 끝(`main` 함수 바로 앞)에 append**. 중간 삽입 금지 — diff 크기 폭증.
5. **`main`의 dispatch 블록 수정은 태스크 끝에 1회만**. 한 `str_replace`로 dispatch 블록과 `--only` help 문자열을 함께 수정.
6. **태스크 사이에 `/clear` 혹은 새 세션 권장**. 한 Tier 끝날 때마다 새 세션 시작.
7. 에러 나면 **작은 단위로 쪼개 재시도**. 절대 "파일을 다시 만들자" 같은 대안 선택 금지 — 기존 구조 유지.

### 벡터라이즈 / JIT / 메모리 최적화 원칙 (반드시 지킬 것)

기존 스크립트는 전부 JIT + `lax.scan` 기반이야. 파이썬 레벨 forward 루프 금지.

1. **Forward는 `_mod.analysis_forward` 재사용**. 새 JIT forward 짜지 말고, 이미 있는 걸 JIT로 감싸 써라.
   ```python
   analysis_forward = _mod.analysis_forward
   jit_analysis = jax.jit(lambda p, ids: analysis_forward(p, model_cfg, ids, mode='light'))
   ```
   `mode='light'`가 default. `mode='full'` (gate 원본 텐서)은 메모리 크니 꼭 필요할 때만.

2. **배치 루프는 `jax.lax.scan`으로**. Python for loop로 `jit_analysis` 반복 호출하면 매번 dispatch 오버헤드. 예시는 `analyze_routing` (589), `analyze_gate_distribution` (1685).
   - Scalar/축소 통계 누적: scan + 누적 변수.
   - Streaming mean/std: Welford 온라인 알고리즘 or simple running sum.

3. **Vmap 우선**. Token/layer 축에 대한 연산은 `jax.vmap`. Python for loop로 토큰 순회 금지.

4. **메모리 reduce는 JIT 안에서**. `[B, S, N, d]` 같은 거대 텐서는 JIT 내부에서 즉시 `.mean() / .sum() / .norm()`으로 축소. `device_get` 후 축소하면 TPU 메모리/전송 낭비.

5. **`jax.device_get`은 루프 밖에서 한 번만**. `np.array(jax.device_get(x))`를 루프 안에서 호출하면 host-device sync 오버헤드 누적.

6. **Gate 원본 텐서 (`[L, B, S, N]`)가 필요하면**: `mode='full'`로 받되, layer/pool 하나씩 꺼내 즉시 reduce. 전체 유지 금지.
   - Know pool: `8 × 512 × 25200 × 4B ≈ 400MB per batch`. 여러 batch 누적하면 OOM.

7. **Per-neuron contribution은 norm-reduce로**. `[B, S, N, d]` 유지 불가. `contrib_norm[B, S, N] = ‖gate_i × (x · r_i) × w_i‖`까지는 JIT 안에서 reduce.

8. **Pairwise 연산은 chunking**. `know_write @ know_write.T`는 25200² × 4B ≈ 2.5GB. Chunked matmul or subsample.

9. **Tree map으로 파라미터 순회**. `jax.tree.map`으로 batch 안에서 여러 pool 일괄 처리.

### 기타 원칙

- 각 분석 함수는 독립적으로 `--only <key>`로 실행 가능해야 함.
- 함수 시그니처: `analyze_<n>(params, cfg, val_tokens, output_dir, n_batches=None, batch_size=8)`. `n_batches=None`이면 함수 내부 default 사용.
- Dispatch에서 `n_batches=_nb or <default>` 패턴 유지 (예: `_nb or 20`).
- `_save_json(data, output_dir, subdir, filename)` 사용. subdir는 분석 키로.
- **용어 일관성** (변수명, JSON 키, 출력 레이블):
  - Pool: `qk`, `v`, `know` (NOT `attn`, `attention`, `knowledge`, `Q`, `K`, `V`).
  - Gate 관련: `gate` (최종), `z` (pre-gate, `(scores - tau) / s_std`), `phi` (confidence, Gaussian CDF of z), `den` (gate-sum normalization denominator).
  - Tau 관련: `tau_offset` (학습된 bias), `tau` (최종 threshold = `s_mean + tau_offset * s_std`).
  - 활성도: `active_frac` (pool 내 활성 비율), `active_N` or `active_count` (활성 개수), `eff_N` (effective neuron count, entropy 기반).
  - Layer 인덱스: `layer_idx` or `li` (0-indexed).
  - 확신 안 서면 기존 함수의 출력 JSON을 grep해 통일된 키 확인.
- Pool 크기/차원/레이어 수는 **항상 `cfg`/`model_cfg`에서**. 40M 숫자 하드코딩 금지 (400M 재실행 보장).
- 설명 주석은 한국어 간략, 코드는 기존 스타일(변수명, 공백) 준수.
- 학습된 모델만 사용.

### Dispatch 추가 템플릿 (모든 태스크 공통)

태스크 끝에 한 번의 `str_replace`로 아래 세 곳을 동시 수정:

1. **`--only` help 문자열** (라인 4450 부근)에 새 키 추가.
2. **`_val_analyses` 리스트** (라인 4500 부근)에 val_tokens 필요하면 키 추가.
3. **Dispatch 블록** (적절한 위치)에 `if _should_run('<key>'):` 추가.

예시:
```python
# main() 내부
parser.add_argument("--only", default=None,
    help="...,rw_align,resid_dyn,phase,drift,gate_ci,write_cov,sel_gini,sel_trans,combo,addit,dom_supp,rw_func")

_val_analyses = [..., 'resid_dyn', 'drift', 'gate_ci', 'sel_gini', 'sel_trans', 'combo', 'addit']
# (forward 필요한 것만. rw_align, phase, write_cov, dom_supp, rw_func은 제외)

if _should_run('resid_dyn'):
    if val_tokens is not None:
        analyze_residual_dynamics(params, cfg, val_tokens, args.output,
                                  n_batches=_nb or 20, batch_size=_abs)
    else:
        print("\n  Skipping resid_dyn (no --val_data)")
```

### 공통 헬퍼 (Task 1.2에서 먼저 만들고 Task 3.3이 재사용)

§7.4.3 (Additivity)와 §8.2 (Per-Unit Contribution) 둘 다 per-neuron contribution 분해가 필요. 공통 헬퍼로 먼저 구현한 후 양쪽에서 호출.

```python
@functools.partial(jax.jit, static_argnums=(1, 3, 4))
def _compute_per_neuron_contribution(params, model_cfg, input_ids, pool, layer_idx,
                                     return_vector=False):
    """주어진 pool/layer에서 per-neuron contribution 반환.

    Returns dict:
        'contrib_norm': [B, S, N]  — ‖gate_i × (x · r_i) × w_i‖, d 축소
        'gate':         [B, S, N]
        'den':          [B, S]     — gate-sum normalization denominator
        'contrib_vec':  [B, S, N, d]  (return_vector=True일 때만, §7.4.3 additivity용)
    raw contribution = gate_i × (x · r_i) × w_i  (pre-normalization)
    normalized = raw / max(den, 1.0)
    """
```

메모리 주의:
- `return_vector=False` default → `[B, S, N]`만 반환. Know pool이면 `8 × 512 × 25200 × 4B ≈ 400MB`.
- `return_vector=True`는 `[B, S, N, d]`, Know pool이면 `160GB`. **반드시 single token × single layer 수준의 sub-sample에서만**.

## 작업 순서 (Tier별)

### === Session 1: Tier 1 (구조 결정용, 우선순위 최상) ===

#### Task 1.1 — `§7.2.1 Read-Write Alignment Distribution`

**의도**: 뉴런 각각의 `cos(r, w)` 분포를 pool별로 뽑아 삼봉/연속/단일봉 판정. §7.2 전체 서술이 이 결과에 달려있음.

**구현**:
- 새 함수 `analyze_rw_alignment(params, cfg, output_dir)`.
- Forward pass 불필요. 파라미터만 사용.
- 각 pool (`qk`, `v`, `know`)에 대해:
  - `r = pool_params['<pool>_read']`, `w = pool_params['<pool>_write']` (둘 다 `[N, d_model]`).
  - Unit normalize 후 `cos_rw = (r_n * w_n).sum(-1)` → `[N]`. JIT로 감싸 한 번 계산.
  - 히스토그램 (bins=100, range=[-1, 1]).
  - 통계: mean, std, min, max, median, p10, p90.
  - Mode detection: `scipy.signal.find_peaks` (없으면 numpy Gaussian KDE → local max). scipy 없으면 skip.
- 저장: `rw_alignment/rw_alignment_summary.json` + `<pool>_cos_rw.npy` (raw values).
- Dispatch key: `rw_align`. Val 불필요.

**태스크 완료 조건**:
1. 함수 추가됨.
2. `main`의 dispatch에 `if _should_run('rw_align'):` 블록 추가됨.
3. `--only` help 문자열에 `rw_align` 추가됨.
4. 문법 에러 없음 (`python -c "import ast; ast.parse(open('...').read())"`).

---

#### Task 1.2 — `§8.1 Residual Trajectory` + `§8.3 Force Composition` + `§8.2 Per-Unit Contribution` + 공통 헬퍼

세 분석이 **같은 forward pass**를 씀. 한 함수로 통합. **공통 헬퍼 `_compute_per_neuron_contribution`도 이 태스크에서 함께 추가** (Task 3.3이 재사용).

**구현**:

**(a) 공통 헬퍼**
- `_compute_per_neuron_contribution(params, model_cfg, input_ids, pool, layer_idx, return_vector=False)` 추가.
- JIT로 감쌀 것 (`static_argnums=(1, 3, 4)`).
- Know 블록 기준 구현, `pool` 인자로 qk/v 분기.
- 메모리 절약: `contrib_norm[B, S, N]`까지만 기본 반환.

**(b) 분석 함수**
- `analyze_residual_dynamics(params, cfg, val_tokens, output_dir, n_batches=None, batch_size=8)`.
- 기본값: `n_batches=20` (CLI `--n_batches`로 오버라이드).
- `_run_layerwise_analysis` 호출로 기본 통계 수집 (이미 `cos_x0`, `x_norm`, `dx`, `a_norm`, `k_norm`, `cos_ak` 있음).
- **추가 측정**:
  - `cos(x_{L+1}, x_L)` per layer: `x_pre` 이미 있음. `_run_layerwise_analysis` 반환 dict에 키 `cos_x_prev` 추가.
  - `cos(attn_out, x_after_attn)`, `cos(know_out, x_final_layer)`.
  - Per-layer `acos` 누적.
  - **§8.2**: 공통 헬퍼 호출해 per-neuron contribution norm. Batch/seq 축 평균하여 `[N]` scalar per pool per layer.
  - Top-1%, top-10% Pareto share.
- 배치 루프는 `jax.lax.scan`으로 reduce.
- 저장:
  - `residual_dynamics/layer_stats.json`: per-layer 모든 scalar metric.
  - `residual_dynamics/per_neuron_contrib_<pool>.npy`: per-neuron scalar (각 pool, 레이어 평균).
  - `residual_dynamics/pareto.json`: top-1% share, top-10% share per pool per layer.
- Dispatch key: `resid_dyn`. Val 필요.

**주의**:
- `_run_layerwise_analysis`의 반환 dict에 **키 추가만** 허용. 기존 키 삭제/이름변경 금지 (`analyze_deep_analysis` 라인 3418이 의존).

**태스크 완료 조건**: 공통 헬퍼 추가, 분석 함수 추가, dispatch 추가, `--only` help 업데이트, `_val_analyses` 리스트 업데이트, `analyze_deep_analysis` 기존 동작 보존.

---

#### Task 1.3 — `§6.3 Phase Dynamics`

**중요**: forward pass 아님. **학습 로그 post-processing**.

**구현**:
- 새 함수 `analyze_phase_dynamics(output_dir, dawn_log_path, baseline_log_path)`.
- 입력: 두 run의 loss/tau/active_frac 시계열 (JSON or CSV). 로그 포맷은 MADST에게 확인.
- CLI argparse 추가: `--dawn-log`, `--baseline-log`.
- 처리:
  - 두 run의 step 축 정렬 (batch_size 보정).
  - Gap = L_dawn - L_baseline.
  - Moving average (window=100 steps). `np.convolve` 사용.
  - Phase 경계: gap 도함수 부호 변화 지점 (`np.diff`, `np.where`).
  - Per-pool tau trajectory, active_frac trajectory.
- 저장: `phase_dynamics/timeseries.json`, `phase_dynamics/phases.json`.
- Dispatch key: `phase`. Val/checkpoint 불필요.

**주의**: 로그 파일이 없으면 작업 진입 전에 MADST에게 로그 파일 경로와 포맷을 **반드시** 물어라. 추측 금지.

**태스크 완료 조건**: 함수 추가, CLI arg 2개 추가, dispatch 추가, `--only` help 업데이트.

---

### === Session 2: Tier 2 ===

세션 1 끝낸 후 **새 세션 시작**.

#### Task 2.1 — `§8.4 Drift-Prediction Alignment`

**구현**:
- 새 함수 `analyze_drift_alignment(params, cfg, val_tokens, output_dir, n_batches=None, batch_size=8)`.
- 기본값: `n_batches=20`.
- **Tied embedding 확인 먼저**. 모델 코드 (`dawn_spatial_v3`) 또는 params 구조 확인:
  - `params` 키에 `lm_head`가 있는지.
  - 없으면 `token_emb`를 재사용 (tied).
- Per-layer `x_L`을 final layer norm 후 LM head 통과 → per-token logits, target token rank & top-k accuracy.
- `cos(x_L_normalized, emb[target])` per layer.
- JIT 버전의 per-layer early exit forward 작성. `_run_layerwise_analysis` 확장 or 별도.
- 배치 루프는 `lax.scan`.
- 저장: `drift_alignment/{layer_alignment, layer_accuracy, layer_rank}.json`.
- Dispatch key: `drift`. Val 필요.

---

#### Task 2.2 — `§7.1.1 Layer-wise Gate Distribution (Φ/z 분리)`

**구현**:
- 새 함수 `analyze_gate_confidence_intensity(params, cfg, val_tokens, output_dir, n_batches=None, batch_size=8)`.
- 기본값: `n_batches=20`.
- `_mod.analysis_forward`에 z/phi가 이미 노출돼 있는지 확인 (라인 2579 부근에 `z_np`, `phi_np`, `den_np` 추출 예시 있음).
- Per-layer per-pool:
  - `z = (scores - tau) / s_std` 분포 (histogram, bins=50, range=[-5, 5]).
  - `phi = 0.5 * (1 + erf(z / sqrt(2)))` 분포.
  - `gate = where(z > 0, z * phi, 0)` 분포 (log bucket).
  - Φ mean, z mean, gate mean per layer.
- 히스토그램은 JIT 안에서 `jnp.histogram` 사용.
- 배치 누적은 scan + 히스토그램 카운트 더하기.
- 저장: `gate_ci/{layer_pool_stats, z_hist, phi_hist, gate_hist}.json`.
- Dispatch key: `gate_ci`. Val 필요.

---

#### Task 2.3 — `§7.2.3 Write Direction Coverage`

**구현**:
- 새 함수 `analyze_write_coverage(params, cfg, output_dir)`.
- Forward pass 불필요. 파라미터만.
- Per pool:
  - SVD → effective rank: `exp(entropy of normalized S²)`. JAX `jnp.linalg.svd` 사용.
  - Pairwise cos sim 히스토그램:
    - qk, v (≤2600): full pairwise 가능. `W_n @ W_n.T`, off-diagonal flatten.
    - know (25200): 5000 neuron subsample × 5000 subsample.
  - Covering radius: 10,000 random unit vector (JAX RNG), 각각에 대해 nearest write까지 max cos.
- Chunked matmul 사용 (5000×5000 chunk).
- 저장: `write_cov/{per_pool_stats, cover_hist, pairwise_hist}.json`.
- Dispatch key: `write_cov`. Val 불필요.

---

#### Task 2.4 — `§7.1.2 Selection Gini / Concentration per Pool`

**구현**:
- 새 함수 `analyze_selection_gini(params, cfg, val_tokens, output_dir, n_batches=None, batch_size=8)`.
- 기본값: `n_batches=20`.
- 기존 `analyze_neuron_utilization` (1859) 내용 먼저 확인. 중복되면 **layer 분리 측정만 추가**.
- Per layer per pool:
  - Per-neuron activation frequency `f_i` (gate > 0 인 토큰 비율).
  - Gini: `G = 1 - Σ f_i × (2 × cum_rank - 1) / N`. JIT 친화적으로 sort + cumsum.
  - Activation coverage: `(f_i > 0).sum() / N`.
  - Mean active count per token.
- Scan으로 배치 누적.
- 저장: `sel_gini/per_layer_pool.json` (pool × layer × metric).
- Dispatch key: `sel_gini`. Val 필요.

---

### === Session 3: Tier 3 + 4 ===

세션 2 끝낸 후 새 세션.

#### Task 3.1 — `§7.1.3 Selection Transition`

**구현**:
- 새 함수 `analyze_selection_transition(params, cfg, val_tokens, output_dir, n_batches=None, batch_size=4)`.
- 기본값: `n_batches=10`.
- Layer L과 L+1 active set (gate > 0) Jaccard similarity per token per pool.
- JIT 안에서 bitmask 계산: `active_L`, `active_L_next` → `(active_L & active_L_next).sum() / (active_L | active_L_next).sum()`.
- vmap으로 token 축 병렬.
- Pool별 측정 (pool이 layer 공유라 의미 있음).
- Token 평균, 분포 (히스토그램).
- 저장: `sel_trans/layer_pair_jaccard.json`.
- Dispatch key: `sel_trans`. Val 필요.

---

#### Task 3.2 — `§7.4.1 + §7.4.2 Combinatorial Coverage & Reuse`

**구현**:
- 새 함수 `analyze_combinatorial_coverage(params, cfg, val_tokens, output_dir, n_batches=None, batch_size=8)`.
- 기본값: `n_batches=100`.
- Per layer per pool:
  - 각 토큰의 active set을 integer hash로 축소 (device_get 후 Python set 사용).
  - **메모리 경계**: JIT 내에서 `active_mask [B, S, N]`을 hash로 축소. `hash = (active_mask * prime_vec).sum() mod LARGE_PRIME` (collision은 허용, 근사치).
  - Host측 `set`에 누적 (streaming).
  - Unique count, top-1000 most common (collections.Counter).
  - Reuse entropy `H = -Σ p_i log p_i`.
- 저장: `combo/{per_layer_pool_stats, top_combinations, entropy}.json`.
- Dispatch key: `combo`. Val 필요.

**메모리 주의**: `[B, S, N]` 마스크를 host로 꺼내지 말 것. Hash는 device에서 계산해 `[B, S]` int64로 축소 후 꺼내기.

---

#### Task 3.3 — `§7.4.3 Additivity Test`

**구현**:
- 새 함수 `analyze_additivity(params, cfg, val_tokens, output_dir, n_tokens=10, n_batches=1)`.
- **Task 1.2에서 만든 공통 헬퍼 `_compute_per_neuron_contribution` 재사용** (`return_vector=True`).
- 샘플 규모 엄격 제한: 1 batch × 2 tokens × 2 layers × 3 pools × active neurons only.
  - 이유: `[B, S, N, d]`가 너무 큼. 토큰 단위로만.
- Per (sampled token t, layer L, pool p):
  - Active set A (`gate > 0`).
  - Full output `out_A = Σ_{i∈A} contrib_i / den_A`.
  - Leave-one-out per i∈A: `out_{A\i}`. `den_{A\i} = max(den_A - gate_i, 1.0)`.
  - Metric: `cos(out_A - out_{A\i}, contrib_i / den_A)`.
  - Metric (raw, pre-norm): `cos(raw_A - raw_{A\i}, contrib_i)` → 이상적으로 1.0.
- Optional JIT (작은 샘플이라 오버헤드 덜 중요).
- 저장: `addit/{raw_stats, normalized_stats, per_layer_pool}.json`.
- Dispatch key: `addit`. Val 필요. 전용 CLI: `--addit_n_tokens` (기본 10), `--addit_n_layers` (기본 3).

---

#### Task 3.4 — `Appendix B.1 + B.2 Domain Suppression`

**구현**:
- 기존 `analyze_cross_domain_suppression` (2337)와 `analyze_suppression` (1545) 내부 먼저 확인. 재사용 가능한 build_suppressed_forward 헬퍼 있을 것.
- 새 함수 `analyze_domain_suppression_ext(params, cfg, output_dir)`.
- Prompts:
  - Target: physics/astronomy.
  - Control: biology, geography, history.
  - 전 논문 prompt 리스트 경로가 있으면 재사용.
- Per prompt: target token top-20 진입 여부 (JIT forward로 logits 계산 후 argsort).
- Contrastive score per neuron: `target_activation_freq - baseline_activation_freq`.
- Top N% (1%, 3%, 5%, 10%) 뉴런 suppress.
- Selectivity index: `(target_drop - control_drop) / target_drop`.
- **B.2 Random baseline**: 같은 수 랜덤 뉴런 suppress (3회 평균).
- 저장: `dom_supp/{targeted, random_baseline, selectivity}.json`.
- Dispatch key: `dom_supp`. Val 불필요 (prompt 기반). 전용 CLI: `--dom_supp_suppress_levels` (기본 "0.01,0.03,0.05,0.10").

---

#### Task 3.5 — `§7.2.2 R-W Angle vs Function (조건부)`

**실행 조건**: Task 1.1 (`rw_align`) 결과에서 삼봉 또는 연속 분포가 확인된 경우만.

**구현**:
- 새 함수 `analyze_rw_function_correlation(params, cfg, val_tokens, output_dir)`.
- 진입 조건 체크: `output_dir/rw_alignment/*_cos_rw.npy` 존재 확인. 없으면 메시지 출력 후 리턴.
- Task 1.1 산출물 로드 → pool별로 10개 bin.
- 각 bin의 뉴런들에 대해:
  - POS selectivity: 기존 R.2 결과 파일에서 로드 (`output_dir/r2/*.json` or 유사).
  - Q/K usage ratio: 기존 R.1 결과 파일에서 로드 (QK pool만).
  - Layer-wise activation pattern: `_run_layerwise_analysis`의 `neur_active`에서 layer별 pool별 추출 → early/mid/late peak 분류.
- 저장: `rw_func/bin_vs_metric.json`, `rw_func/scatter_data.npz`.
- Dispatch key: `rw_func`. Val 필요 (layer-wise activation 때문).

---

## 각 태스크 공통 체크리스트

태스크 끝낼 때마다 확인:

- [ ] 함수가 파일 끝 (`main` 바로 앞)에 추가됨
- [ ] `main`의 dispatch에 `if _should_run('<key>'):` 블록 추가됨
- [ ] `--only` help 문자열에 새 키 추가됨
- [ ] `_val_analyses` 리스트에 필요 시 추가됨 (val_tokens 쓰는 분석만)
- [ ] 기존 `analyze_*` 함수 수정 없음 (또는 있다면 이유 명시)
- [ ] `python -c "import ast; ast.parse(open('scripts/analysis/standalone/spatial_analysis.py').read())"` 통과
- [ ] 저장 경로가 `output_dir/<subdir>/`로 일관됨
- [ ] `_save_json` 사용
- [ ] Forward는 `_mod.analysis_forward` 재사용 (새로 짜지 않음)
- [ ] 배치 루프는 `jax.lax.scan` (python for 루프 아님)
- [ ] `device_get`은 루프 밖 1회
- [ ] Pool 크기/차원 숫자 하드코딩 없음 (전부 `cfg`/`model_cfg`에서)
- [ ] 용어 일관성 (`qk`/`v`/`know`, `gate`/`z`/`phi`/`den`, `tau_offset`/`tau`)
- [ ] 진행 요약 1~3줄로 보고하고 다음 태스크 대기

## 예외 처리

- 특정 태스크에서 기존 함수 수정이 필요해지면 **수정 전 MADST에게 확인**.
- 파라미터 구조 (`pool_params` 키 이름 등) 확신 안 서면 **추측 말고 먼저 확인** (라인 3295~3305 참고).
- Forward pass가 OOM 나면 batch_size 줄이고, n_batches 유지.
- `mode='full'` 필요한 분석은 메모리 모니터링. OOM이면 `mode='light'` + 분리 forward로 대체.

## 태스크 전체 요약

| Session | Task | Section | Key | Val | Default n_batches |
|---|---|---|---|---|---|
| 1 | 1.1 | §7.2.1 R-W Alignment | `rw_align` | N | — |
| 1 | 1.2 | §8.1+§8.2+§8.3 Residual Dynamics (+공통 헬퍼) | `resid_dyn` | Y | 20 |
| 1 | 1.3 | §6.3 Phase Dynamics | `phase` | N | — |
| 2 | 2.1 | §8.4 Drift-Prediction | `drift` | Y | 20 |
| 2 | 2.2 | §7.1.1 Gate Φ/z | `gate_ci` | Y | 20 |
| 2 | 2.3 | §7.2.3 Write Coverage | `write_cov` | N | — |
| 2 | 2.4 | §7.1.2 Selection Gini | `sel_gini` | Y | 20 |
| 3 | 3.1 | §7.1.3 Selection Transition | `sel_trans` | Y | 10 |
| 3 | 3.2 | §7.4.1+§7.4.2 Combinatorial Coverage | `combo` | Y | 100 |
| 3 | 3.3 | §7.4.3 Additivity | `addit` | Y | 1 |
| 3 | 3.4 | Appendix B.1+B.2 Domain Suppression | `dom_supp` | N | — |
| 3 | 3.5 | §7.2.2 R-W Angle vs Function (조건부) | `rw_func` | Y | 10 |

총 12개 태스크. 이미 확보된 §6.1, §6.2, §7.3.1, §7.3.2는 제외.

기본값은 CLI `--n_batches`, `--batch_size`로 실행 시 오버라이드 가능. 함수 내부 default는 fallback일 뿐.

## 시작

Task 1.1부터 시작. 이전 맥락 없이 이 문서만으로 작업 가능.
