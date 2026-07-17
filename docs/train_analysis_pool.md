# train_analysis_pool

`train_analysis_pool`은 DAWN의 공유 RW 연산 공간이 의미 있는 인과적 계산 단위인지 판별하는 단일 분석 시스템이다. 사용자 작성 프롬프트, 생성 IOI, 임의 synthetic binding, 버전별 분석 엔진, 과거 probe 산출물은 입력으로 받지 않는다.

지원 체크포인트 버전은 다음 둘뿐이다.

- `spatial-r1-v4.1.7.1`
- `spatial-r1-v4.1.7.2`

v4172는 v4171과 같은 production retention·contribution·interchange 훅을 사용하고, 체크포인트가 선언한 generalized bilinear operator key를 그대로 materialize한다. 다른 모델 버전, schema, tokenizer, benchmark manifest, checkpoint identity, model config, protocol이 섞이면 실행 또는 resume이 실패한다. 예전 이름이나 artifact를 새 형식으로 묵시적으로 해석하는 호환 경로는 없다.

## 공식 입력

기본 `primary` build는 다음 공개 데이터만 사용한다.

| ID | 공식 소스 | 공식 counterfactual | 분석 트랙 |
|---|---|---|---|
| `mib_ioi` | `mib-bench/ioi` | `s2_io_flip_counterfactual` | MIB-adapted operator-site circuit |
| `mib_mcqa` | `mib-bench/copycolors_mcqa`, `4_answer_choices` | `symbol_counterfactual` | MIB-adapted operator-site circuit |
| `mib_arithmetic` | `mib-bench/arithmetic_addition` | `random_counterfactual` | MIB-adapted operator-site circuit |
| `mib_arc` | `mib-bench/arc_easy` | `symbol_counterfactual` | MIB-adapted operator-site circuit |
| `ravel` | `mib-bench/ravel` | `prompt_template_counterfactual`, `attribute_counterfactual` | RAVEL-style operator-contribution mediation |

선택적 `all` build는 공식 BLiMP와 CounterFact를 행동 확인 자료로 추가하지만 primary circuit 또는 RAVEL claim에는 사용하지 않는다.

준비 단계는 원본 dataset revision, tokenizer revision·vocabulary hash, 실제 phase별 row 수, 제외 사유, 각 JSONL SHA-256을 immutable `manifest.json`에 기록한다. 원본 split을 재사용해야 할 때는 공식 row identity를 hash-bucket해 discovery·validation·test에 정확히 한 번만 배정한다.

RAVEL schema v2는 같은 공식 base row에서 만든 cause와 isolation을 동일한 `pair_group_id`로 묶고 두 행을 항상 같은 phase에 원자적으로 배정한다. trace 위치는 demonstration 속 동명이 아니라 prompt에서 대상 entity가 마지막으로 나타나는 token이다. 구 schema build는 거부된다.

```bash
python3 scripts/prepare_interpretability_benchmarks.py \
  --output-root gs://dawn-tpu-data-c4/dataset/operator_interpretability \
  --benchmarks primary \
  --publish-latest
```

## 분석 아이템

모든 아이템은 v4171과 v4172를 지원하며 버전별 별칭은 없다.

| 아이템 | 판별 대상 | 선행 아이템 | 지원 버전 |
|---|---|---|---|
| `benchmark_contract` | 소스·tokenizer·phase·checkpoint·protocol 고정 | 없음 | v4171, v4172 |
| `behavioral_eligibility` | 해석할 만큼 base와 source 모두 정답 행동을 보이는가 | `benchmark_contract` | v4171, v4172 |
| `operator_localization` | 어느 `(layer, route, operator)`가 production contribution을 운반하는가 | `behavioral_eligibility` | v4171, v4172 |
| `conditional_circuit_sufficiency` | 선택 회로가 전체 production admission denominator 아래 충분한가 | `operator_localization` | v4171, v4172 |
| `autonomous_circuit_sufficiency` | 선택 회로가 자체 admission denominator만으로 충분한가 | `operator_localization` | v4171, v4172 |
| `circuit_necessity` | validation에서 선택한 회로를 억제하면 held-out margin이 감소하는가 | `conditional_circuit_sufficiency` | v4171, v4172 |
| `operator_space_structure` | 주소와 무관하게 RW 함수의 국소 family가 재현되는가 | `operator_localization` | v4171, v4172 |
| `ravel_causal_mediation` | operator family가 목표 변수를 전달하고 비목표 효과를 격리하는가 | `operator_space_structure` | v4171, v4172 |
| `multilayer_trajectory` | held-out same-variable 경로가 disjoint cross-variable 대조보다 유사한가 | `operator_localization` | v4171, v4172 |
| `scientific_claims` | 모든 선행 조건을 통과한 가장 강한 claim은 무엇인가 | 전체 결과 아이템 | v4171, v4172 |

현재 preset은 `contract`, `circuit`, `causal`, `scientific`이며 기본값은 `scientific`이다. 코드와 같은 catalog는 다음 명령으로 확인한다.

```bash
python3 scripts/analyze_train_analysis_pool.py --list-items
```

## 사전 고정된 분석 규칙

- 분석 단위의 전체 모집단은 모든 `(layer, route, operator)` site다.
- 회로 fraction은 MIB circuit track의 `0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%`다.
- 행동 적격성은 base와 source의 clean-label positive-minus-negative margin이 모두 양수인 행만 인정한다. RAVEL localization·trajectory의 독립 표본 단위는 cause/isolation 행이 아니라 공식 base row다.
- sparse 순위는 production precision에서 각 operator의 post-denominator contribution vector norm을 cross-operator cancellation 전에 계산한다. gate mass나 scalar coefficient를 contribution의 대용치로 쓰지 않는다.
- 후보 순위는 discovery에서만 만든다. captured mass 95%를 못 채우면 route별 폭을 사전 고정된 최대치까지 늘리고, 그래도 부족한 행은 attrition으로 제외한다. validation/test에서는 후보를 다시 순위화하지 않는다.
- 가장 작은 회로는 validation faithfulness의 bootstrap 95% CI 하한이 기준을 통과할 때만 선택한다. 선택을 고정한 뒤 test를 한 번 평가한다.
- faithfulness는 `(circuit - corrupted) / (baseline - corrupted)`다. corrupted margin은 공식 MIB처럼 clean label 방향을 유지한다.
- conditional sufficiency는 선택 numerator와 전체 production admission denominator를 쓴다. autonomous sufficiency는 numerator와 denominator를 모두 선택 회로에서 계산한다.
- necessity는 선택 회로 전체의 numerator를 억제한 paired margin drop이며 bootstrap CI, paired permutation, benchmark 간 BH 보정을 모두 통과해야 한다.
- functional family는 정규화한 rank-one RW map의 함수 유사도만으로 찾는다. reciprocal seed-local neighborhood만 family로 인정하고 transitive closure는 하지 않는다. address는 discovery에서 제외하고 사후 compactness 확인에만 쓴다.
- operation-space 계산은 설정된 후보 부분공간에 한정된다. 후보가 전체 pool이 아니면 결과에 funnel limitation을 명시하고 전체 공간 또는 충분성 claim을 통과시키지 않는다.
- interchange는 production route의 post-denominator·learned-pool-scale contribution에 `base - selected(base) + selected(source)`를 적용한다.
- RAVEL은 `Continent`, `Country`, `Language`마다 discovery seed와 그 seed를 포함하는 가장 작은 RW family를 별도로 고른다. family는 seed-only 및 같은 크기의 disjoint nonfamily control보다 cause-margin improvement가 커야 한다. nonfamily control은 변수별 discovery mean absolute contribution으로 중복 없이 matching한다. 세 변수와 두 대조의 6개 검정을 한 번에 BH 보정한다.
- RAVEL cause/isolation은 변수별로 각각 최소 8쌍을 요구한다. cause는 positive causal transfer CI와 paired permutation을, isolation은 absolute non-target effect CI 상한을 판정한다.
- multilayer trajectory는 공식 base group을 anchor·same-variable·cross-variable control로 겹치지 않게 나누며, triplet 하나를 통계 표본 하나로 bootstrap/permutation한다.
- seed, 표본 수, alpha와 모든 임계값은 protocol hash에 포함된다. 중간 단계가 실패하거나 captured mass·rank stability가 부족하면 더 강한 claim으로 승격하지 않는다.

checkpoint identity는 resolved path, step, run metadata와 parameter path·shape·dtype schema hash를 묶는다. 400M parameter value 전체의 content hash는 수집하지 않으며 이 제한을 결과에 명시한다. 따라서 결과는 해당 checkpoint identity에만 유효하고, 복제 checkpoint나 다른 checkpoint로의 일반화는 별도 반복 실험 없이는 주장하지 않는다.

이 구현은 MIB의 TransformerLens edge graph 또는 RAVEL의 공식 featurizer와 동치라고 주장하지 않는다. 정확한 명칭은 `MIB-adapted operator-site circuit`과 `RAVEL-style operator-contribution mediation`이다.

## 실행

```bash
python3 scripts/analyze_train_analysis_pool.py \
  --checkpoint gs://.../checkpoints/latest \
  --benchmark-root gs://dawn-tpu-data-c4/dataset/operator_interpretability \
  --preset scientific \
  --init-distributed
```

TPU pod 전체 worker에서는 다음 launcher를 쓴다.

```bash
bash scripts/launch_train_analysis_pool_tpu_pod.sh \
  --tpu spatial-400m \
  --zone us-central2-b \
  --project dawn-486218 \
  --branch main \
  --checkpoint gs://.../checkpoints/latest \
  --preset scientific
```

기본 출력은 해당 run의 `side_analysis/train_analysis_pool/<checkpoint-step>/`이다. `items/<item>.json`과 `summary.json`은 protocol, checkpoint, model-config, benchmark manifest hash를 포함한다. protocol record가 정확히 일치하지 않는 산출물은 resume하지 않는다.

## Claim ladder

최종 claim은 다음 순서로 fail-closed 평가한다.

1. localization
2. necessity
3. conditional sufficiency
4. autonomous sufficiency
5. interchange causality
6. non-target isolation
7. held-out generalization
8. spatial trajectory confirmation

마지막 단계는 RAVEL family가 두 대조군을 모두 이긴 6-test BH 결과와 held-out trajectory CI·paired null까지 요구한다. 중간 단계가 실패하면 그 아래 단계까지만 보고하며, 결과가 좋지 않다는 이유로 임계값을 완화하거나 test에서 회로를 다시 선택하지 않는다.
