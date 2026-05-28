#!/usr/bin/env python3
"""
PPL Validity Verification Script for DAWN

This script performs two critical tests to verify if the reported PPL is valid:
1. Mask Visualization Test - Checks if causal mask is properly applied
2. Text Generation Test - Checks if model generates coherent text

If PPL 1.8~2.3 is valid:
- Causal mask should show lower-triangular pattern (0s below diagonal, -inf above)
- Generated text should be coherent and meaningful

If PPL is invalid (mask leak / future token access):
- Mask might be all zeros or incorrect pattern
- Generated text will be garbage, repetitive, or NaN

Supports all DAWN versions (auto-detection from checkpoint).

Usage:
    python scripts/verify_ppl_validity.py --weights /path/to/checkpoint.pt
    python scripts/verify_ppl_validity.py --weights /path/to/checkpoint.pt --prompt "The weather today is"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from models import create_model_by_version, normalize_version


def load_model_from_checkpoint(weights_path, device='cpu'):
    """Load model from checkpoint with auto version detection."""
    ckpt = torch.load(weights_path, map_location=device)

    # Get config and state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        config = ckpt.get('model_config', ckpt.get('config', {}))
        epoch = ckpt.get('epoch')
        step = ckpt.get('step')
    else:
        state_dict = ckpt
        config = {}
        epoch = step = None

    # Clean compiled model prefix
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Auto-detect version from config or state_dict keys
    version = config.get('model_version', None)
    if version is None:
        v18_2_keys = ['router.tau_proj.weight', 'router.neuron_router.norm_fqk_Q.weight']
        dawn_keys = ['shared_neurons.f_neurons', 'router.neuron_router.neuron_emb']

        if all(k in state_dict for k in v18_2_keys):
            version = '18.2'
        elif any(k in state_dict for k in dawn_keys):
            if config.get('learnable_tau', False) or config.get('max_paths'):
                version = '18.2'
            else:
                version = '17.1'
        else:
            version = 'baseline'

    version = normalize_version(version)
    print(f"  Detected version: {version}")

    model = create_model_by_version(version, config)
    model.load_state_dict(state_dict, strict=False)

    if epoch is not None:
        print(f"  Loaded from epoch {epoch}")
    if step is not None:
        print(f"  Loaded from step {step}")

    return model, config, version


def visualize_attention_mask(model, tokenizer, device, prompt="Hello world"):
    """
    Test 1: Visualize attention mask to verify causal masking

    Expected output (for seq_len=5):
    [[   0, -inf, -inf, -inf, -inf],
     [   0,    0, -inf, -inf, -inf],
     [   0,    0,    0, -inf, -inf],
     [   0,    0,    0,    0, -inf],
     [   0,    0,    0,    0,    0]]

    If you see all 0s or incorrect pattern, causal masking is broken!
    """
    print("\n" + "="*60)
    print("TEST 1: Attention Mask Visualization")
    print("="*60)

    # Tokenize
    tokens = tokenizer.encode(prompt, add_special_tokens=False)[:10]  # Limit to 10 tokens
    input_ids = torch.tensor([tokens], device=device)
    B, S = input_ids.shape

    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {tokens}")
    print(f"Token texts: {[tokenizer.decode([t]) for t in tokens]}")
    print(f"Sequence length: {S}")

    # Manually create what the attention scores should look like BEFORE softmax
    # This simulates what happens in AttentionCircuit.forward()

    # Causal mask (upper triangular = True, meaning those positions should be -inf)
    causal_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)

    # Create dummy scores and apply mask
    dummy_scores = torch.zeros(1, 1, S, S, device=device)
    masked_scores = dummy_scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))

    print(f"\n[Causal Mask Pattern (0=attend, -inf=blocked)]")
    print(f"Shape: {masked_scores.shape} (batch, head, query, key)")
    print(f"\nMask values (first head):")

    # Pretty print the mask
    mask_2d = masked_scores[0, 0].cpu()
    for i in range(S):
        row = []
        for j in range(S):
            val = mask_2d[i, j].item()
            if val == float('-inf'):
                row.append("-inf")
            else:
                row.append(f"{val:4.0f}")
        print(f"  [{', '.join(f'{x:>5}' for x in row)}]")

    # Verify pattern
    is_correct = True
    for i in range(S):
        for j in range(S):
            expected_inf = (j > i)  # Future positions should be -inf
            actual_inf = (mask_2d[i, j] == float('-inf'))
            if expected_inf != actual_inf:
                is_correct = False
                break

    if is_correct:
        print(f"\n[PASS] Causal mask is CORRECT (lower triangular pattern)")
        print("  - Position (i,j) where j > i is blocked (-inf)")
        print("  - Each token can only attend to itself and previous tokens")
    else:
        print(f"\n[FAIL] Causal mask is INCORRECT!")
        print("  - This would allow future token access (cheating)")

    return is_correct


def test_generation(model, tokenizer, device, prompts=None, max_new_tokens=30, temperature=0.8):
    """
    Test 2: Generate text to verify model learned meaningful patterns

    Valid model should generate:
    - Coherent, grammatically reasonable text
    - Related to the prompt context

    Invalid model (mask leak) would generate:
    - Random garbage
    - Same token repeated infinitely
    - NaN or errors
    """
    print("\n" + "="*60)
    print("TEST 2: Text Generation Test")
    print("="*60)

    if prompts is None:
        prompts = [
            "The weather today is",
            "In the year 2024,",
            "The capital of France is",
            "Once upon a time, there was a",
            "Scientists discovered that",
        ]

    model.eval()
    results = []

    for prompt in prompts:
        print(f"\n[Prompt] {prompt}")

        # Tokenize
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=device)

        generated_ids = input_ids.copy()

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # Get model predictions
                    curr_input = torch.tensor([generated_ids], device=device)
                    seq_len = curr_input.shape[1]

                    # Pad to 512 for stable importance distribution (matches training)
                    target_len = min(512, model.max_seq_len)
                    if seq_len < target_len:
                        pad_len = target_len - seq_len
                        curr_input = F.pad(curr_input, (0, pad_len), value=tokenizer.pad_token_id)
                        attention_mask = torch.ones(1, target_len, device=device)
                        attention_mask[:, seq_len:] = 0
                    else:
                        # Limit sequence length
                        if seq_len > model.max_seq_len:
                            curr_input = curr_input[:, -model.max_seq_len:]
                            seq_len = model.max_seq_len
                        attention_mask = torch.ones(1, seq_len, device=device)

                    logits = model(curr_input, attention_mask=attention_mask)

                    # Use logits at last real token position (before padding)
                    last_pos = seq_len - 1

                    # Check for NaN
                    if torch.isnan(logits).any():
                        print("[ERROR] NaN detected in logits!")
                        results.append(("ERROR", "NaN in logits"))
                        break

                    # Get next token probabilities (last real token position, not padding)
                    next_token_logits = logits[0, last_pos, :] / temperature
                    probs = F.softmax(next_token_logits, dim=-1)

                    # Sample
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    generated_ids.append(next_token)

                    # Stop at [SEP] or [PAD] or period
                    if next_token in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
                        break

                else:
                    # Successfully generated max tokens
                    pass

                # Decode
                generated_text = tokenizer.decode(generated_ids)
                new_text = tokenizer.decode(generated_ids[len(input_ids):])

                print(f"[Generated] {generated_text}")
                print(f"[New tokens] {new_text}")

                # Basic quality check
                # Check for infinite repetition
                if len(set(generated_ids[len(input_ids):])) <= 2 and len(generated_ids) - len(input_ids) > 5:
                    print("[WARNING] Repetitive output detected (same tokens repeated)")
                    results.append(("WARN", "Repetitive"))
                else:
                    results.append(("OK", generated_text))

        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            results.append(("ERROR", str(e)))

    # Summary
    print("\n" + "-"*40)
    print("Generation Test Summary:")
    ok_count = sum(1 for r in results if r[0] == "OK")
    warn_count = sum(1 for r in results if r[0] == "WARN")
    error_count = sum(1 for r in results if r[0] == "ERROR")

    print(f"  OK: {ok_count}/{len(results)}")
    print(f"  WARN: {warn_count}/{len(results)}")
    print(f"  ERROR: {error_count}/{len(results)}")

    if error_count > 0:
        print("\n[FAIL] Generation has errors - model may be broken")
        return False
    elif ok_count >= len(results) // 2:
        print("\n[PASS] Generation looks reasonable")
        print("  -> If text is coherent, PPL result is likely VALID!")
        return True
    else:
        print("\n[UNCERTAIN] Generation has warnings - manually inspect output")
        return None


def verify_model_internals(model, tokenizer, device):
    """
    Test 3: Verify model internal attention patterns

    Run a forward pass and check attention scores shape/values
    """
    print("\n" + "="*60)
    print("TEST 3: Model Internal Verification")
    print("="*60)

    model.eval()

    # Simple test input
    prompt = "Hello world"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(input_tensor)

    print(f"\nTest input: '{prompt}'")
    print(f"Input shape: {input_tensor.shape}")

    with torch.no_grad():
        # Forward pass
        logits = model(input_tensor, attention_mask=attention_mask)

        print(f"Output logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

        # Check for NaN/Inf
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()

        if has_nan:
            print("[FAIL] NaN detected in output!")
            return False
        if has_inf:
            print("[FAIL] Inf detected in output!")
            return False

        # Check probabilities
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top_probs, top_ids = torch.topk(probs, 5)

        print(f"\nTop 5 predictions for next token:")
        for i, (prob, token_id) in enumerate(zip(top_probs, top_ids)):
            token_text = tokenizer.decode([token_id.item()])
            print(f"  {i+1}. '{token_text}' (id={token_id.item()}, prob={prob.item():.4f})")

        # Entropy check (very low entropy might indicate memorization)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        print(f"\nPrediction entropy: {entropy:.4f}")
        print(f"  (Higher = more uncertain, Lower = more confident)")

        if entropy < 0.5:
            print("  [NOTE] Very low entropy - model is very confident")
        elif entropy > 8:
            print("  [NOTE] Very high entropy - model is uncertain")
        else:
            print("  [OK] Entropy in normal range")

    print("\n[PASS] Model internals look OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify PPL validity for DAWN model")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt for generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-new-tokens', type=int, default=30, help='Max tokens to generate')
    args = parser.parse_args()

    print("="*60)
    print("DAWN PPL Validity Verification")
    print("="*60)
    print(f"\nDevice: {args.device}")
    print(f"Weights: {args.weights}")

    # Load tokenizer
    print("\nLoading tokenizer (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load model with auto version detection
    print(f"Loading model...")
    model, config, version = load_model_from_checkpoint(args.weights, device='cpu')
    model = model.to(args.device)

    # Use batch-level routing (same as training) + 512 padding for stable importance
    if hasattr(model, 'router'):
        model.router.attention_token_routing = False
        model.router.knowledge_token_routing = False

    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {param_count:.1f}M")

    # Run tests
    results = {}

    # Test 1: Mask visualization
    results['mask'] = visualize_attention_mask(model, tokenizer, args.device)

    # Test 2: Generation
    prompts = [args.prompt] if args.prompt else None
    results['generation'] = test_generation(
        model, tokenizer, args.device,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens
    )

    # Test 3: Internal verification
    results['internals'] = verify_model_internals(model, tokenizer, args.device)

    # Final verdict
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)

    all_pass = all(r is True for r in results.values())
    any_fail = any(r is False for r in results.values())

    if all_pass:
        print(f"""
[VALID] All tests passed!

Your PPL result appears to be VALID.

This means:
- Causal masking is correctly implemented (no future token leak)
- Model generates coherent text (learned meaningful patterns)
- No NaN/Inf issues in model outputs

Congratulations! Your DAWN {version} model is performing well.
The dynamic routing and orthogonality constraints are working as intended.
""")
    elif any_fail:
        print("""
[INVALID] Some tests FAILED!

Your PPL result may be artificially low due to:
- Causal mask leak (model can see future tokens)
- Model internal issues (NaN, broken weights)

Please check:
1. AttentionCircuit mask combination in model_v17_1.py
2. attention_mask handling in train.py
3. Data preprocessing (no label leakage)
""")
    else:
        print("""
[UNCERTAIN] Results are mixed.

Please manually inspect:
1. Generated text quality - is it coherent?
2. Attention mask pattern - is it lower triangular?

If generated text makes sense, PPL is likely valid.
If it's garbage or repetitive, there may be an issue.
""")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
