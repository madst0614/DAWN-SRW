# train_analysis_pool

`train_analysis_pool`은 고정된 DAWN 체크포인트의 행동 평가와 RW 연산 공간 해석을 한 item registry에서 관리한다. 분석 대상 데이터는 item ID에 포함되며 별도의 `benchmark` 실행 축은 없다. 파라미터를 갱신하는 downstream fine-tuning은 이 시스템에 포함하지 않는다.

지원 체크포인트 버전:

- `spatial-r1-v4.1.7.1`
- `spatial-r1-v4.1.7.2`

## 실행 모델

한 번의 실행은 다음 네 요소로 정의된다.

```text
run = target × items/preset × runtime × protocol
```

- **item**: 무엇을 측정하는지와 입력 task를 함께 고정한다.
- **preset**: 자주 함께 실행하는 item ID만 묶는다.
- **target**: 모델 버전, 체크포인트, scale, model-axis mesh를 고정한다.
- **runtime**: 물리 accelerator와 전역 JAX device 수를 고정한다.

Target과 runtime은 독립적이지만 실행 직전에 결합된다. Target의 `mesh_model`은 체크포인트 model-axis와 정확히 같아야 하고, runtime의 device 수로 data-axis를 계산한다.

```text
mesh_data = runtime.global_device_count / target.mesh_model
```

기본 runtime은 `v4-64`이다. `v4-64`는 JAX device 32개이므로 `mesh_model=2`인 400M target은 `mesh_data=16`, 즉 `16×2` mesh로 실행된다. 나눗셈이 정확하지 않거나 실제 보이는 JAX device/process 수가 runtime과 다르면 자동 보정하지 않고 실패한다. Target의 canonical config에 선언된 model fields도 checkpoint `full_config.model`과 대조하므로 이름만 400M이고 실제 구조가 다른 checkpoint는 통과하지 못한다.

Target과 runtime registry는 [`configs/train_analysis_pool.yaml`](../configs/train_analysis_pool.yaml)에 있다. TPU 이름, zone, project는 target에 포함하지 않는다.

## 등록 target

| Target | Model version | Scale | Model mesh axis |
|---|---|---:|---:|
| `v4171_400m` | `spatial-r1-v4.1.7.1` | 400M | 2 |
| `v4172_400m_den_qk0p5_v1p0_rst1p2` | `spatial-r1-v4.1.7.2` | 400M | 2 |

Target 경로가 run/root를 가리키더라도 실행 시작 시 committed numeric Orbax step 하나로 고정한다. 결과에는 요청 target, 실제 checkpoint path와 step, checkpoint mesh, effective mesh를 모두 기록한다.

## 등록 runtime

| Runtime | Accelerator | JAX devices | Workers |
|---|---|---:|---:|
| `v4-32` | `v4-32` | 16 | 4 |
| `v4-64` | `v4-64` | 32 | 8 |
| `v4-128` | `v4-128` | 64 | 16 |

## Concrete items

### MIB circuit items

다음 여섯 item이 각 `mib_ioi`, `mib_mcqa`, `mib_arithmetic`, `mib_arc` prefix 아래에 있다.

```text
<mib>.input_contract
<mib>.behavioral_eligibility
<mib>.operator_localization
<mib>.conditional_circuit_sufficiency
<mib>.autonomous_circuit_sufficiency
<mib>.circuit_necessity
```

예시:

```text
mib_ioi.behavioral_eligibility
mib_ioi.operator_localization
mib_ioi.conditional_circuit_sufficiency
mib_ioi.autonomous_circuit_sufficiency
mib_ioi.circuit_necessity
```

`behavioral_eligibility`는 frozen inference만 수행한다. Base와 source의 positive-minus-negative log-probability margin을 계산하고 둘 다 맞은 예제만 후속 기전 분석에 전달한다. Optimizer, weight update, checkpoint write는 없다.

### RAVEL items

```text
ravel.input_contract
ravel.behavioral_eligibility
ravel.operator_localization
ravel.operator_space_structure
ravel.causal_mediation
ravel.multilayer_trajectory
```

`ravel.causal_mediation`은 route-native contribution interchange로 family, seed-only control, contribution-matched disjoint control을 비교한다. Cause와 isolation을 함께 통과해야 하며 다중 검정은 BH 보정한다.

### Auxiliary mechanistic behavior items

```text
blimp.input_contract
blimp.behavioral_eligibility
counterfact.input_contract
counterfact.behavioral_eligibility
```

이 결과는 외부 행동 확인용이며 primary scientific claim 선택에는 사용하지 않는다.

### Stock zero-shot items

```text
zero_shot.lambada_openai
zero_shot.hellaswag
zero_shot.piqa
zero_shot.arc_easy
zero_shot.arc_challenge
zero_shot.winogrande
```

Zero-shot backend는 `lm-eval==0.4.2`, `num_fewshot=0`의 stock task를 사용한다. 선택된 task들은 체크포인트 restore와 evaluator 실행을 공유하지만 결과와 protocol은 item별로 저장한다. 이 item들은 auxiliary이며 functional-family 발견, circuit 선택, RAVEL family 선택, scientific claim gate의 입력으로 사용하지 않는다.

### Scientific aggregation

```text
scientific_claims.primary
```

이 item은 primary MIB/RAVEL 기전 item의 fail-closed claim ladder만 집계한다. Zero-shot과 BLiMP/CounterFact 결과는 읽지 않는다.

## Presets

| Preset | 목적 |
|---|---|
| `contract` | primary input contract만 확인 |
| `zero_shot` | stock zero-shot 6개 |
| `mechanistic_screen` | primary 행동 적격성만 확인 |
| `mib_ioi_circuit` | IOI circuit 분석 |
| `mib_mcqa_circuit` | MCQA circuit 분석 |
| `mib_arithmetic_circuit` | arithmetic circuit 분석 |
| `mib_arc_circuit` | ARC circuit 분석 |
| `circuit` | 네 MIB circuit 전체 |
| `ravel_causal` / `causal` | RAVEL causal + trajectory |
| `scientific` | primary scientific claim dependency closure 전체 |
| `all` | scientific + auxiliary behavior + zero-shot |

기본 preset은 `scientific`이다. Preset에는 checkpoint, TPU, runtime, benchmark selector를 넣지 않는다.

현재 catalog 전체는 코드에서 직접 확인할 수 있다.

```bash
python3 scripts/analyze_train_analysis_pool.py --list-items --list-targets
```

## 실행

Target 사용:

```bash
python3 -u scripts/analyze_train_analysis_pool.py \
  --target v4171_400m \
  --runtime v4-64 \
  --preset mechanistic_screen \
  --init-distributed
```

개별 item 사용:

```bash
python3 -u scripts/analyze_train_analysis_pool.py \
  --target v4171_400m \
  --items mib_ioi.behavioral_eligibility,mib_ioi.operator_localization \
  --init-distributed
```

Ad-hoc checkpoint 사용:

```bash
python3 -u scripts/analyze_train_analysis_pool.py \
  --checkpoint gs://.../checkpoints/000000003200 \
  --runtime v4-64 \
  --preset zero_shot \
  --init-distributed
```

`--target`과 `--checkpoint`는 상호 배타적이다. 등록 target에서는 `--config`나 다른 `mesh_model`로 target 계약을 덮어쓸 수 없다.

TPU pod launcher:

```bash
bash scripts/launch_train_analysis_pool_tpu_pod.sh \
  --tpu spatial-400m \
  --zone us-central2-b \
  --project dawn-486218 \
  --branch main \
  --target v4171_400m \
  --runtime v4-64 \
  --preset scientific
```

Launcher는 실제 TPU `acceleratorType`이 선택 runtime과 같은지 먼저 검사한다.

## 산출물

기본 출력:

```text
<run>/side_analysis/train_analysis_pool/<checkpoint-step>/
```

주요 파일:

```text
items/mib_ioi/behavioral_eligibility.json
items/mib_ioi/operator_localization.json
items/ravel/causal_mediation.json
items/zero_shot/hellaswag.json
items/scientific_claims/primary.json
backends/operator_interpretability/summary.json
backends/stock_zero_shot/results_summary.json
backends/stock_zero_shot/run_manifest.json
summary.json
```

Mechanistic item protocol에는 checkpoint identity, model config, benchmark manifest, target, runtime, checkpoint/effective mesh가 포함된다. Zero-shot item protocol에는 concrete checkpoint, parameter identity, stock task config hash, dataset fingerprint, tokenizer 정책, target/runtime mesh가 포함된다.

`--resume`은 item protocol이 정확히 일치할 때만 허용한다. 기존 artifact가 있지만 target, concrete checkpoint, runtime, code revision 또는 protocol이 다르면 덮어쓰지 않고 실패하며, 의도적으로 다시 계산할 때만 `--no-resume`을 사용한다.

## 과학적 경계

- Test phase는 circuit/family 선택에 사용하지 않는다.
- Conditional sufficiency와 autonomous sufficiency를 구분한다.
- Necessity는 full production denominator 아래 selected numerator suppression으로 계산한다.
- Interchange는 `base - selected(base) + selected(source)`이며 V contribution은 V 위치에만 삽입한다.
- Empty/padded group은 production 계약에 맞는 contribution no-op으로 처리한다.
- Zero-shot과 secondary behavior item은 scientific claim gate에 사용하지 않는다.
- Downstream fine-tuning은 파라미터를 변경하므로 pool 밖에 둔다.
