# Operator dataset schema recommendation

Generated: 2026-07-14T00:33:27+00:00

## Common contract

Schema candidate: `dawn_operator_pair_v2_candidate` version `2`.

A/B is retained for every row. Datasets without a second condition set B lengths to 0; BLiMP uses A/B as good/bad full sequences with all candidate lengths 0. Continuation datasets store complete multi-token candidates and are scored teacher-forced by summing all candidate token log probabilities.

`target_span_start/end` must index a span that is actually stored in context/full-sequence arrays. For BLiMP it is the first divergence token span. Continuation candidate spans are derived at scoring time as `[context_len, context_len + candidate_len)` and must not be put into context span fields; use `-1/-1` when no source-grounded context span is defined.

| Dataset | Status | Score mode | Trace definition | Negative rule |
|---|---|---|---|---|
| ravel | ready_for_adapter | continuation_margin | last_context_token_prediction_state | n/a |
| blimp | ready_for_adapter | paired_sequence_logprob | pre_divergence_prediction_state | unused (length 0) |
| lama | ready_for_adapter | continuation_margin | last_context_token_prediction_state | different object from the same relation |
| counterfact | ready_for_adapter | true_new_object_margin | last_context_token_prediction_state | requested_rewrite.target_new.str |
| ioi | ready_for_adapter | clean_corrupt_continuation_margin | last_context_token_prediction_state | clean_distractor |
| synthetic | ready_for_adapter | continuation_margin | last_context_token_prediction_state | n/a |

## Dataset mappings

### ravel

| Common field | Source rule |
|---|---|
| `context_ids_a/b` | "city_prompt.Template % city_entity.City (Template must contain exactly one %s)" |
| `positive_ids_a/b` | "city_entity[row][city_prompt.Attribute]" |
| `negative_ids_a/b` | "same Attribute column from a different entity with a different value" |
| `entity_extension` | "city_entity.City and ID" |
| `template_extension` | "city_prompt.Template, Source, Entity" |
| `attribute/group_id` | "city_prompt.Attribute / controlled pair family" |

Observed issues/risks:

- "RAVEL is normalized: city_entity supplies City and attribute values; city_prompt supplies Attribute and one-%s Template. Adapter must join them, not expect input/label columns."

### blimp

| Common field | Source rule |
|---|---|
| `context_ids_a` | "sentence_good" |
| `context_ids_b` | "sentence_bad" |
| `phenomenon/group_id` | "parquet parent directory" |
| `source_id` | "UID when present; otherwise phenomenon + row index" |
| `trace_position_a/b` | "longest common prefix length - 1" |
| `candidate_arrays` | "unused (length 0)" |

### lama

| Common field | Source rule |
|---|---|
| `relation` | "predicate_id, relation, or relation file stem" |
| `subject` | "sub_label or subject" |
| `positive_ids_a` | "obj_label or object" |
| `negative_ids_a` | "different object from the same relation" |
| `context_ids_a` | "prefix before the single terminal [MASK] in masked_sentences[0]" |
| `source_extension` | "source category/file, original masked sentence, object id" |

Observed issues/risks:

- {"source_file": "data/Squad/test.jsonl", "risk": "nontrivial suffix after [MASK]: ' color was used.'", "row": {"masked_sentences": ["To emphasize the 50th anniversary of the Super Bowl the [MASK] color was used."], "obj_label": "gold", "id": "56be4db0acb8001400a502f0_0", "sub_label": "Squad"}}
- {"source_file": "data/Squad/test.jsonl", "risk": "nontrivial suffix after [MASK]: ' position for the Denver Broncos.'", "row": {"masked_sentences": ["Von Miller plays in [MASK] position for the Denver Broncos."], "obj_label": "linebacker", "id": "56bf17653aeaaa14008c9513_0", "sub_label": "Squad"}}
- {"source_file": "data/Google_RE/date_of_birth_test.jsonl", "risk": "nontrivial suffix after [MASK]: ') is an American politician who is a Democratic member of the Rhode Island House of Representatives .'", "row": {"pred": "/people/person/date_of_birth", "sub": "/m/09gb0bw", "obj": "1941", "evidences": [{"url": "http://en.wikipedia.org/wiki/Peter_F._Martin", "snippet": "Peter F. Martin (born 1941) is an American politician who is a Democratic member of the Rhode Island House of Representatives. He has represented the 75th District Newport since 6 January 2009. He is currently serves on the House Committees on Judiciary, Municipal Government, and Veteran's Affairs. During his first term of office he served on the House Committees on Small Business and Separation of Powers & Government Oversight. In August 2010, Representative Martin was appointed as a Commissioner on the Atlantic States Marine Fisheries Commission", "considered_sentences": ["Peter F Martin (born 1941) is an American politician who is a Democratic member of the Rhode Island House of Representatives ."]}], "judgments": [{"rater": "18349444711114572460", "judgment": "yes"}, {"rater": "17595829233063766365", "judgment": "yes"}, {"rater": "4593294093459651288", "judgment": "yes"}, {"rater": "7387074196865291426", "judgment": "yes"}, {"rater": "17154471385681223613", "judgment": "yes"}], "sub_w": null, "sub_label": "Peter F. Martin", "sub_aliases": [], "obj_w": null, "obj_label": "1941", "obj_aliases": [], "uuid": "18af2dac-21d3-4c42-aff5-c247f245e203", "masked_sentences": ["Peter F Martin (born [MASK]) is an American politician who is a Democratic member of the Rhode Island House of Representatives ."]}}
- {"source_file": "data/Google_RE/date_of_birth_test.jsonl", "risk": "nontrivial suffix after [MASK]: ') is a team handball player from Argentina .'", "row": {"pred": "/people/person/date_of_birth", "sub": "/m/0hrd6l2", "obj": "1982-05-22", "evidences": [{"url": "http://en.wikipedia.org/wiki/Maria_Pilar_Romero", "snippet": "Maria Pilar Romero (born 22 May 1982) is a team handball player from Argentina. She plays on the Argentine national team, and participated at the 2011 World Women's Handball Championship in Brazil.", "considered_sentences": ["Maria Pilar Romero (born 22 May 1982) is a team handball player from Argentina ."]}], "judgments": [{"rater": "14728826318420078902", "judgment": "yes"}, {"rater": "7520160179218527648", "judgment": "yes"}, {"rater": "3571030073219370006", "judgment": "yes"}, {"rater": "17154471385681223613", "judgment": "yes"}, {"rater": "14953915240951219400", "judgment": "yes"}], "sub_w": "Q7723827", "sub_label": "Pilar Romero", "sub_aliases": ["Maria Pilar Romero"], "obj_w": null, "obj_label": "1982", "obj_aliases": [], "uuid": "a6847930-759b-4100-adbc-3259f1169521", "masked_sentences": ["Maria Pilar Romero (born 22 May [MASK]) is a team handball player from Argentina ."]}}
- {"source_file": "data/Google_RE/place_of_death_test.jsonl", "risk": "masked sentence does not contain exactly one [MASK]", "row": {"pred": "/people/deceased_person/place_of_death", "sub": "/m/0205jm", "obj": "/m/06mzp", "evidences": [{"url": "http://en.wikipedia.org/wiki/John_Renshaw_Starr", "snippet": "After the war John Starr opened a night-club in Hanley, Staffordshire, in partnership with the brothers Alfred and Henry Newton, SOE agents whom he had met during his training and also at the Avenue Foch. The Newton brothers had been in the Buchenwald concentration camp. He later returned to live in Paris, before moving to Switzerland, where he died in 1996.", "considered_sentences": ["After the war John Starr opened a night-club in Hanley, Staffordshire, in partnership with the brothers Alfred and Henry Newton, SOE agents whom he had met during his training and also at the Avenue Foch .", "He later returned to live in Paris, before moving to Switzerland, where he died in 1996 ."]}], "judgments": [{"rater": "17966044108931836156", "judgment": "yes"}, {"rater": "1406006087371975930", "judgment": "yes"}, {"rater": "16676975657004889938", "judgment": "yes"}, {"rater": "13855588585185268925", "judgment": "yes"}, {"rater": "4185483665120273114", "judgment": "yes"}], "sub_w": "Q3182510", "sub_label": "John Renshaw Starr", "sub_aliases": [], "obj_w": "Q39", "obj_label": "Switzerland", "obj_aliases": ["Swiss Confederation", "CH", "SUI", "Suisse", "Schweiz", "Svizzera", "🇨🇭"], "uuid": "61aad52c-4256-468a-a9ae-55e3fa4dc44e", "masked_sentences": ["After the war John Starr opened a night-club in Hanley, Staffordshire, in partnership with the brothers Alfred and Henry Newton, SOE agents whom he had met during his training and also at the Avenue Foch .", "He later returned to live in Paris, before moving to [MASK], where he died in 1996 ."]}}
- {"source_file": "data/Google_RE/place_of_death_test.jsonl", "risk": "nontrivial suffix after [MASK]: ' in 1551 .'", "row": {"pred": "/people/deceased_person/place_of_death", "sub": "/m/0c4031r", "obj": "/m/056_y", "evidences": [{"url": "http://en.wikipedia.org/wiki/Diego_de_Arroyo", "snippet": "Diego de Arroyo, a miniature painter, who was born at Toledo in 1498, is supposed to have studied either in Italy or under an Italian master. His delicate miniature portraits gained him much renown, and the appointment of painter to Charles V. He also illuminated choir-books for the cathedral of Toledo. Arroyo died at Madrid in 1551.", "considered_sentences": ["Arroyo died at Madrid in 1551 ."]}], "judgments": [{"rater": "3633697795227880988", "judgment": "yes"}, {"rater": "3422319225482654240", "judgment": "yes"}, {"rater": "8841266254638695693", "judgment": "yes"}, {"rater": "13855588585185268925", "judgment": "skip"}, {"rater": "15683306407052553031", "judgment": "yes"}], "sub_w": "Q5274857", "sub_label": "Diego de Arroyo", "sub_aliases": [], "obj_w": "Q2807", "obj_label": "Madrid", "obj_aliases": ["City of Madrid"], "uuid": "97483332-dd08-45d5-8181-a4438756c351", "masked_sentences": ["Arroyo died at [MASK] in 1551 ."]}}
- {"source_file": "data/Google_RE/place_of_death_test.jsonl", "risk": "nontrivial suffix after [MASK]: ', Indiana .'", "row": {"pred": "/people/deceased_person/place_of_death", "sub": "/m/0bmf_6s", "obj": "/m/0sn4f", "evidences": [{"url": "http://en.wikipedia.org/wiki/Art_Murakowski", "snippet": "Murakowski died in 1985 at age 60 at his home in Hammond, Indiana. He was survived by his wife, Lucille Murakowski, three sons, and three daughters.", "considered_sentences": ["Murakowski died in 1985 at age 60 at his home in Hammond, Indiana ."]}], "judgments": [{"rater": "14404876356854644346", "judgment": "yes"}, {"rater": "16676975657004889938", "judgment": "yes"}, {"rater": "14894151489932721675", "judgment": "yes"}, {"rater": "17966044108931836156", "judgment": "yes"}, {"rater": "4185483665120273114", "judgment": "yes"}], "sub_w": null, "sub_label": "Art Murakowski", "sub_aliases": [], "obj_w": "Q856860", "obj_label": "Hammond", "obj_aliases": ["Hammond, Indiana"], "uuid": "9ed48bca-4f61-412f-80c2-42bd44a63d58", "masked_sentences": ["Murakowski died in 1985 at age 60 at his home in [MASK], Indiana ."]}}
- {"source_file": "data/Google_RE/place_of_death_test.jsonl", "risk": "nontrivial suffix after [MASK]: 's until his death .'", "row": {"pred": "/people/deceased_person/place_of_death", "sub": "/m/05tmcf", "obj": "/m/0jdtt", "evidences": [{"url": "http://en.wikipedia.org/wiki/Laurent_Belissen", "snippet": "By 1722 Belissen settled in Marseille, where he succeeded Antoine Blanchard as maître de musique of the Abbey of St. Victor, which was then rapidly declining in importance--but he also secured a position directing the city's Académie de Concerts. Belissen remained in Marseilles until his death.", "considered_sentences": ["Belissen remained in Marseilles until his death ."]}], "judgments": [{"rater": "3633697795227880988", "judgment": "yes"}, {"rater": "18393280535210278125", "judgment": "yes"}, {"rater": "8841266254638695693", "judgment": "yes"}, {"rater": "13855588585185268925", "judgment": "yes"}, {"rater": "15683306407052553031", "judgment": "yes"}], "sub_w": "Q1808061", "sub_label": "Laurent Belissen", "sub_aliases": [], "obj_w": "Q23482", "obj_label": "Marseille", "obj_aliases": ["Bay of Marseille", "Massaliotes", "Massalia", "Marsailles", "Marseilles", "Marsielles", "Marsielle", "City of Marseille", "Marsaille", "Marsiglia"], "uuid": "10fb9157-33b5-4254-a27f-6abec0e5ce50", "masked_sentences": ["Belissen remained in [MASK]s until his death ."]}}

### counterfact

| Common field | Source rule |
|---|---|
| `context_ids_a` | "requested_rewrite.prompt.format(requested_rewrite.subject) when exactly one {} exists" |
| `positive_ids_a` | "requested_rewrite.target_true.str" |
| `negative_ids_a` | "requested_rewrite.target_new.str" |
| `relation` | "requested_rewrite.relation_id" |
| `source_id` | "case_id" |
| `extension` | "paraphrase_prompts, neighborhood_prompts, generation_prompts and target ids" |

### ioi

| Common field | Source rule |
|---|---|
| `context_ids_a` | "clean_prompt" |
| `positive_ids_a` | "clean_correct" |
| `negative_ids_a` | "clean_distractor" |
| `context_ids_b` | "corrupt_prompt" |
| `positive_ids_b` | "corrupt_correct" |
| `negative_ids_b` | "corrupt_distractor" |
| `group_id` | "template_id + pattern_type" |

Observed issues/risks:

- "Current generator stores one correct/distractor pair, but corrupt name order reverses the semantic labels."

### synthetic

| Common field | Source rule |
|---|---|
| `context_ids_a/b` | "controlled pair prompts" |
| `positive_ids_a/b` | "condition-specific correct value" |
| `negative_ids_a/b` | "same-attribute-domain matched distractor" |
| `group_id/phenomenon` | "pair family" |
| `extension` | "entity, paired_entity, attribute, controlled_change" |

Observed issues/risks:

- "Current generator has independent rows and no distractor, pair_id, family, or controlled_change."

## Drop conditions

Rows are dropped only by the full preparation adapter, never by this probe. Required reasons: context longer than max_seq_len; candidate longer than max_candidate_tokens; empty required candidate; identical positive/negative token sequence; target span lost; unstable/out-of-range trace position; ambiguous source schema; or a LAMA mask with a nontrivial suffix that cannot be converted to a decoder prefix without changing the task.

## Source-specific extensions

Keep RAVEL entities/attributes/templates, BLiMP UID and linguistic fields, LAMA original masks and object ids, CounterFact auxiliary prompts/target ids, IOI template/pattern/name fields, and Synthetic controlled-change fields under metadata extension objects. Do not widen fixed arrays.

## Undecided / fail-loud

Any dataset marked partial, source_unavailable, schema_inconsistent, or failed is not an adapter contract. In particular, no RAVEL join is inferred unless the observed parquet/raw keys support it, and no LAMA masked sentence with meaningful text after [MASK] is silently rewritten.
