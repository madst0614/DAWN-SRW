# Operator dataset source probe

Generated: 2026-07-14T00:33:27+00:00

## ravel

- Status: `ready_for_adapter`
- Splits/categories: `['test', 'train', 'val']`
- Observed columns: `['Attribute', 'City', 'Continent', 'Country', 'Entity', 'ID', 'Language', 'Latitude', 'Longitude', 'Source', 'Template', 'Timezone', 'URL']`
- Row count: `{'city_entity/test-00000-of-00001.parquet': 1126, 'city_entity/train-00000-of-00001.parquet': 2041, 'city_entity/val-00000-of-00001.parquet': 970, 'city_prompt/test-00000-of-00001.parquet': 372, 'city_prompt/train-00000-of-00001.parquet': 442, 'city_prompt/val-00000-of-00001.parquet': 397}`
- Prepared sample count: `5`
- Pair types: `['same_attribute_different_entity', 'same_entity_different_attribute', 'same_attribute_different_prompt', 'cross_attribute_control']`

Source issues:

- "RAVEL is normalized: city_entity supplies City and attribute values; city_prompt supplies Attribute and one-%s Template. Adapter must join them, not expect input/label columns."

## blimp

- Status: `ready_for_adapter`
- Splits/categories: `['train']`
- Observed columns: `['UID', 'field', 'lexically_identical', 'linguistics_term', 'one_prefix_method', 'pair_id', 'sentence_bad', 'sentence_good', 'simple_LM_method', 'two_prefix_method']`
- Row count: `{'anaphor_gender_agreement/train-00000-of-00001.parquet': 1000, 'determiner_noun_agreement_1/train-00000-of-00001.parquet': 1000, 'adjunct_island/train-00000-of-00001.parquet': 1000}`
- Prepared sample count: `5`
- Pair types: `['good_bad_minimal_pair']`

## lama

- Status: `ready_for_adapter`
- Splits/categories: `['ConceptNet', 'Google_RE', 'Squad', 'TREx']`
- Observed columns: `['evidences', 'id', 'judgments', 'masked_sentences', 'obj', 'obj_aliases', 'obj_label', 'obj_uri', 'obj_w', 'pred', 'predicate_id', 'source_category', 'source_file', 'sub', 'sub_aliases', 'sub_label', 'sub_uri', 'sub_w', 'uuid']`
- Row count: `{'archive_entries': 52, 'sampled_file_rows_read': {'data/TREx/P740.jsonl': 936, 'data/Google_RE/place_of_birth_test.jsonl': 2937, 'data/ConceptNet/test.jsonl': 29774, 'data/Squad/test.jsonl': 305, 'data/Google_RE/date_of_birth_test.jsonl': 1825, 'data/Google_RE/place_of_death_test.jsonl': 766}, 'note': 'probe intentionally stops after enough rows; these are not full file counts'}`
- Prepared sample count: `5`
- Pair types: `['same_relation_different_subject']`

## counterfact

- Status: `ready_for_adapter`
- Splits/categories: `['counterfact']`
- Observed columns: `['attribute_prompts', 'case_id', 'generation_prompts', 'neighborhood_prompts', 'paraphrase_prompts', 'pararel_idx', 'requested_rewrite']`
- Row count: `21919`
- Prepared sample count: `5`
- Pair types: `['true_object_vs_rewrite_object']`

## ioi

- Status: `ready_for_adapter`
- Splits/categories: `['generated']`
- Observed columns: `['clean_prompt', 'correct', 'corrupt_prompt', 'distractor', 'id', 'target_position', 'task_variable', 'template_id']`
- Row count: `10`
- Prepared sample count: `5`
- Pair types: `['clean_corrupt_name_binding']`

Source issues:

- "Current generator stores one correct/distractor pair, but corrupt name order reverses the semantic labels."

## synthetic

- Status: `ready_for_adapter`
- Splits/categories: `['generated']`
- Observed columns: `['answer', 'attribute', 'entity', 'id', 'prompt', 'target_position', 'task_variable']`
- Row count: `{'current_generated_probe': 10, 'pair_family_probe': 7}`
- Prepared sample count: `5`
- Pair types: `['same_operation_different_entity', 'same_entity_different_attribute', 'attribute_swap', 'entity_swap', 'distractor_swap', 'order_permutation', 'irrelevant_fact_addition']`

Source issues:

- "Current generator has independent rows and no distractor, pair_id, family, or controlled_change."
