# Terminology and Notation Registry

This file is the source of truth for terminology and reader-facing notation in the v15 manuscript. The governing rule is: **one concept = one canonical term = one canonical notation**.

| Canonical concept | Canonical term | Canonical notation | Exact meaning | Allowed variants | Do NOT use as synonym | First-definition location |
|---|---|---|---|---|---|---|
| Research problem | explicit learned state-transition organization | prose | Whether learned transition-producing computation can be architecturally organized so that the conditions, operations, and transition effects constituting actual execution are explicit objects of measurement and intervention. | explicit learned transition organization when repetition would be unwieldy | State-Conditioned Program; language modeling; DAWN-SRW | Introduction |
| Organizational coordinates | organizational coordinates | `\mathcal P=(\mathcal O,C,\Gamma)` | Minimal descriptive coordinates for one family of explicit learned transition organizations: an operation family, a state-conditioned applicability relation, and a composition rule. They are not a universal ontology or necessary decomposition and do not by themselves establish actual computational organization. | minimal organizational coordinates | factorization; program; architecture; hypothesis; proof of organization | Introduction |
| Learned operation family | operation family | `\mathcal O=\{O_i\}_i` | The indexed family of learned transformations whose controlled responses may contribute to a transition. | learned-operation family | operator pool when speaking generically beyond the DAWN realization | Introduction |
| Post-hoc algebraic decomposition | functional decomposability | prose | The fact that a function can be written as component terms after the fact; by itself this does not show that those terms are represented or executed as such. | post-hoc functional decomposition | computational organization; architectural exposure | Introduction; Limitations |
| Roles used in actual execution | computational organization | prose | A separation whose operation, applicability, and composition roles are explicitly realized in the actual forward computation. | actual computational organization | functional decomposability; architecture | Introduction |
| Concrete visibility of organizational roles | architectural exposure | prose | A realization that makes the relevant operation identities and controlled contributions referable for measurement or intervention. | exposed organization | functional decomposability; computational organization | Introduction |
| Representation at one position | state | `x` | A token/representation state on which transition-producing computation acts. | current state; representation state when needed for clarity | hidden object; memory state | Introduction; Method §3.1 |
| Sequence representation at depth | sequence representation state | `X_l` | The sequence-wide representation entering depth `l`. | sequence state when the depth-indexed object is already clear | hidden-state collection | Method §3.1 |
| Change produced by model computation | state transition | `X_{l+1}=X_l+\Delta_l(X_l)` | The transition from a current representation state to its next state. | transition-producing computation | update rule; state evolution when used as a synonym rather than a description | Introduction; Method §3.1 |
| Coarse attention transformation | attention transformation | `\operatorname{Attn}_l(X_l)` | The coarse attention-mediated contribution in the opening state-transition equation; distinct from the attention-weight matrix `A_l`. | none | `A_l(X_l)`; obscure replacement symbols | Method §3.1 |
| State-derived control of operation contributions | state-conditioned control | `C(x,O_i)=c_i(x)`; DAWN: `C_a(x)` | The applicability relation that determines when, whether, and to what degree each operation contributes for state `x`; `C_a(x)` is the DAWN type-specific applicability profile. | state-conditioned applicability relation; state-conditioned applicability profile | routing; gating; selection; dispatch; access | Abstract; Introduction; Method §3.2–3.4 |
| Abstract transformation or computational role | operation | `O_{a,i}(x)` | A learned transformation/computational role; `O_{a,i}(x)` is its output for `x`. | learned operation; identifiable learned operation; input-dependent operation output | primitive; memory unit; expert; module; atomic unit; rule | Abstract; Introduction |
| Indexed learned object implementing an operation | operator | index `i` | A stable learned architectural object whose identity and contribution can be measured or intervened on. | learned operator; operator identity; operator contribution; operator address; operator site; operator circuit | primitive; memory unit; expert; module; bank element; atomic unit; rule | Introduction; Method §3.3 |
| DAWN-SRW rank-1 operator | RW operator | `O_{a,i}(x)=\langle x,\bar r_{a,i}\rangle\bar w_{a,i}` | The rank-1 read--write realization of an operator in the evaluated DAWN-SRW realization; `\bar r_{a,i}` and `\bar w_{a,i}` are its normalized read and write directions. | rank-1 RW operator; RW transition component | RW primitive; memory unit; expert; module | Method §3.2 |
| Scalar content read by an RW operator | scalar state read | `\bar r_{a,i}^{\top}x` | The full-`D` inner product that reads one scalar response from the current representation. It is distinct from the low-dimensional applicability query and operator address. | scalar read response; RW read response | applicability; applicability score; operator address | Method §3.2–3.3 |
| Direction written by an RW operator | write direction | `\bar w_{a,i}` | The full-`D` learned direction scaled by the scalar state read and applicability to form an RW contribution. | RW write direction | applicability; operator address; operation response | Method §3.2 |
| Applicability of one operator | applicability | `c_{a,i}(x)` | A continuous-valued coefficient in `[0,1]`; zero means exact non-participation and a positive value means applicable. | continuous-valued applicability; applicability coefficient | gate; activation; admission weight; contribution weight | Abstract; Introduction; Method §3.3 |
| Applicability-modulated operation response | controlled operation contribution | `c_{a,i}(x)O_{a,i}(x)` | An operation response modulated by its state-conditioned applicability. | operator-wise contribution when the controlled referent is clear | condition; applicability; operation response alone | Abstract; Introduction; Method §3.4 |
| State-derived address | query | `q_a(x)` | Unit-normalized address derived from state `x` for computation type `a`. | state-derived query | routing query; control query after first definition | Method §3.3; §3.5 |
| Persistent address of an operator | operator address | `k_{a,i}` | Unit-normalized low-`d_c` coordinate used to determine applicability of operator `i` in the operation-address space; it is not the scalar RW read operand `\bar r_{a,i}^{\top}x`. | address when the referent is unambiguous | scalar read; key as a DAWN synonym outside Q/K prior-work or implementation contexts | Method §3.3; §3.5 |
| Query--address relation used for applicability | applicability score | `s_{a,i}(x)=q_a(x)^{\top}k_{a,i}` | The inner product between unit-normalized query and operator address. | cosine relation once at definition | similarity; directional similarity; matching value; geometric score; `\rho_{a,i}` | Method §3.3 |
| State-dependent cutoff for applicability | applicability threshold | `\tau_a(x)` | The state-dependent threshold against which an applicability score is compared. | threshold after the full term is established | boundary; support boundary; threshold boundary; DirectTau boundary | Method §3.3 |
| Operators with positive applicability | exact applicability support | `\mathcal I_a(x)=\{i:c_{a,i}(x)>0\}` | The exact set of applicable operators for state `x`; it is a subset of `{1,\ldots,N_a}`. | applicability support; state-conditioned exact applicability support when emphasizing the empirical conditioning; sparse applicability support when sparsity is specifically discussed | input-conditioned exact applicability support; hard support; active set; exact gate; sparse gate; participation mask; admission set; visibility set | Method §3.3 |
| Operators considered before applicability is determined | candidate scope | prose | The set/range of operators whose applicability is evaluated; in the evaluated DAWN-SRW realization it is the full type-specific operator pool. | full candidate scope | visibility; visible set; candidate set as a synonym for the pool | Method §3.6; Discussion |
| Operators of one computation type | operator pool | `{1,\ldots,N_a}` | The type-specific collection of all `N_a` learned operators. | persistent operator pool; type-specific operator pool; full operator pool; QK/V/RST operator pool | operation bank; operator bank; operation vocabulary; operator vocabulary; collection; library | Method §3.3 |
| Number of operators | operator-pool size | `N_a` | Number of operators in the type-specific operator pool for computation type `a`. | number of operators | vocabulary size; bank size | Method §3.3 |
| DAWN-SRW geometric applicability realization | operation-address space | `\mathbb R^{d_c}` | The geometric space in which DAWN-SRW query and operator-address variables realize state--operation applicability; `c_{a,i}(x)` is the resulting continuous-valued applicability. | none | control space as an alias; routing space; address geometry; operation geometry; state-operation space; addressing manifold | Method §3.5 |
| Operation-address dimensionality | control dimension | `d_c` | Dimension of the query and operator addresses; `d_c=256` in the evaluated models. | operation-address dimension once if explanatory | `R`; `d_{\mathrm{addr}}` | Method §3.5 |
| General combination of controlled operation contributions | composition rule | `\Gamma` | The rule by which controlled operation contributions form one resulting transition. It is a separate organizational role, not a third factor of the control--operation factorization and not the resulting transition itself. | composition when the rule referent is clear | `F(x)` as the rule; third factor; controlled composition as a named mechanism; aggregate update; routed output | Introduction |
| Result of applying an organizational specification | transition-producing result | `F_{\mathcal P}(x)`; DAWN: `F_a(x)` | The output obtained after `\Gamma` combines the controlled operation contributions for `x`; `F_a(x)` is the DAWN computation-type-specific result. | composed result | composition rule; `\Gamma` | Introduction; Method §3.4 |
| DAWN-SRW composition mechanism | normalized additive composition | `\Gamma` realized by the equation for `F_a(x)` | The normalization-factor-scaled additive realization by which DAWN combines multiple continuously controlled RW-operator contributions into one transition-producing result while preserving explicit operator-wise contributions at a fixed state. This does not make the overall state-dependent mapping linear. | none | normalized composition; additive composition; controlled composition; `F_a(x)` as the rule | Abstract; Method §3.4 |
| Composition normalization | normalization factor | `Z_a(x)` | The sum-derived factor used by normalized additive composition. | normalizer | denominator; admission normalization; composition normalization; normalization mass | Method §3.4 |
| Hardware execution axis | physical execution | prose | How the computation is executed on hardware, distinct from applicability support and candidate scope. | hardware execution | applicability support; candidate scope | Method §3.6; Discussion |
| DAWN-SRW physical execution behavior | dense RW execution | prose | The evaluated implementation executes RW computation densely even when applicability support is sparse. | physical RW execution is dense | execution or dense execution without an immediate RW referent; sparse execution; conditional execution when not implemented | Abstract; Method §3.6; Discussion; Limitations |
| Program-oriented organizational formulation | State-Conditioned Program | prose | One program-oriented formulation of explicit learned state-transition organization in which actual forward execution realizes current state → state-conditioned applicability over learned operations → controlled operation contributions → composition → resulting state transition. Merely writing organizational coordinates or a post-hoc decomposition is insufficient. | computational formulation; program formulation | research problem; three-way factorization; formal calculus; architecture; mechanism; realization; arbitrary functional decomposition | Introduction |
| Testable architectural proposition | architectural hypothesis | prose | Learned transition-producing computation can organize its actual forward execution through stable learned operations under explicit state-conditioned control over their contributions. | actual-execution architectural hypothesis | organizational form; formulation; architecture; realization; mechanism | Introduction; Method §3.2 |
| Concrete architecture studied | DAWN-SRW realization | prose | The evaluated concrete realization of the architectural hypothesis using rank-1 RW operators, an operation-address space, persistent identities, full candidate scope, normalized additive composition, and dense RW execution. | DAWN-SRW; evaluated DAWN-SRW realization; concrete realization | State-Conditioned Program; architectural hypothesis; v4172 as a scientific name | Abstract; Introduction; Method |
| Evidence that the realization trains competitively | viability | prose | The first bounded empirical claim. | none | performance preservation as a claim label | Abstract; Introduction; Results; Discussion; Conclusion |
| Evidence of state-conditioned applicability patterns | conditional organization | prose | The second bounded empirical claim; state-conditioned exact applicability support provides evidence of conditional organization but does not by itself establish state-to-state operator-identity variation. | none | established conditional organization; conditional structure; conditional routing structure; input-dependent organization | Abstract; Introduction; Results; Discussion; Conclusion |
| Evidence that identified operators can be intervened on | causal accessibility | prose | The third bounded empirical claim. | none | mechanistic interpretability as a claim label | Abstract; Introduction; Results; Discussion; Conclusion |
| Preregistered-stop outcome | fail-closed results | prose | Results for which later phases are not opened after a preregistered criterion fails. | fail-closed result for one study | fail-closed analyses; negative boundaries; fail-closed boundaries; failed boundary | Abstract; Results; Discussion; Appendix |
| Indirect-object-identification task | indirect object identification (IOI) | prose | The indirect object identification behavior used for the primary causal analysis. | IOI after first definition | undefined `IOI`; indirect-object-identification repeatedly re-expanded | Abstract |
| Direct residual update computation type | residual-state transition (RST) | `a=\mathrm{RST}`; `F_{\mathrm{RST}}` | The DAWN-SRW computation type whose normalized additive composition directly updates the post-attention residual state. | RST after first definition; RST operator pool | residual update as an acronym expansion; undefined `RST` | Introduction, Contributions |

## Formal Naming and Repository Mapping

| Identifier | Precise role | Usage policy |
|---|---|---|
| DAWN-SRW | Scientific architecture/model name used in the paper. | Use in ordinary scientific prose. |
| DAWN-SRW v4.1.7.2 | Exact evaluated implementation revision used by both the 400M and 1.3B models. | State once in the formal manuscript mapping and use when the implementation revision is material. |
| `spatial-r1-v4.1.7.2` | Repository model identifier recorded by both evaluated training configs, checkpoint runs, and artifact metadata. | Preserve literally in reproducibility identities. |
| `v4172` | Historical/internal shorthand retained in config names, source filenames, target names, and artifact paths. | Do not use as a standalone scientific name; preserve literal repository identities. It is not a separate architectural concept. |

## Conceptual Hierarchy

Research problem
\(\rightarrow\) explicit learned state-transition organization
\(\rightarrow\) minimal organizational coordinates \(\mathcal P=(\mathcal O,C,\Gamma)\)
\(\rightarrow\) State-Conditioned Program
\(\rightarrow\) actual-forward-execution architectural hypothesis
\(\rightarrow\) DAWN-SRW concrete realization
\(\rightarrow\) autoregressive language modeling as the current constructive test domain
\(\rightarrow\) empirical claims.

- **Computational-role distinction:** the separation between what transformation
  is performed and when, whether, and to what degree it contributes to the
  current state transition.
- **Control--operation factorization:** one explicit architectural representation
  that separates when, whether, and to what degree an operation contributes from
  what transformation it performs. It is a constructive operationalization, not
  a necessary, intrinsic, or unique decomposition.
- **Controlled operation contribution:** an operation response modulated by its
  state-conditioned applicability.
- **Composition:** the rule by which controlled operation contributions form one
  resulting transition. Composition combines contributions, not conditions, and
  is not a third factor of the control--operation factorization. The rule is
  \(\Gamma\); \(F_{\mathcal P}(x)\), or \(F_a(x)\) in DAWN, is its resulting output.
- **State-Conditioned Program:** one program-oriented organizational formulation
  of the broader research problem, organized around current state,
  state-conditioned control and learned operations, composition, and the next
  state, with those roles explicitly present in actual forward execution rather
  than supplied only as a post-hoc decomposition.
- **Architectural hypothesis:** the narrower, constructively testable proposition
  recorded in the registry above.
- **DAWN-SRW realization:** one concrete realization of that hypothesis; the bounded
  empirical claims test this realization under the evaluated conditions.

## Editing Rules

1. Before introducing a new term or symbol, check this file.

2. If an existing concept already has a canonical term or notation, reuse it exactly.

3. Do not replace an established term with a stylistic synonym merely to avoid repetition.

4. Repetition of a canonical technical term is good when it reduces ambiguity.

5. Use terminology from prior work when describing that prior work, but return to DAWN's canonical terminology when discussing DAWN.

6. If two terms genuinely encode different concepts, preserve the distinction.

7. If a terminology or notation decision is genuinely ambiguous, do not resolve it by inventing a new term. Add it to `AUTHOR_DECISIONS.md` with the current alternatives, exact manuscript locations, semantic difference (if any), and a recommended choice with rationale.
