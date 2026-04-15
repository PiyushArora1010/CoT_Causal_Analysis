import json
from modules.causal_engine import CausalCoTEngine
from modules.dataset import Dataset, GSM8KDataset

if __name__ == "__main__":
    # breakpoint()
    engine = CausalCoTEngine(model_tag="Qwen/Qwen3-4B", temperature=0)
    SAMPLE_SIZE = 500
    
    dataset = GSM8KDataset(split="train", subset_size=SAMPLE_SIZE)
    
    engine.initial_prompt_template = """You must think step by step. Each step must be on a new line and start with 'Step:'. Each step must be related to reasoning and not a conclusion. At the end, provide your final answer inside <answer></answer> tags. Do not include any extra text.

Question: {question}

Answer Output Format: Output just the numerical value inside <answer></answer> tags with no extra text.

Let's think step by step:
"""

    engine.prompt_to_counterfactual = """You are an expert in causal reasoning for mathematical word problems. Your task is to generate a **plausible counterfactual version** of a specific reasoning step.

---

### MANDATORY REQUIREMENT
You **MUST** alter the factual content of the step. Returning the same step — even with minor rewording — is a failure. The counterfactual MUST produce a different numerical result or logical outcome than the original step.

---

### What the counterfactual must satisfy:
1. **Semantically different** – the numbers, quantities, operations, or assignments must change enough that a solver following your version would reach a **different final answer** than in the original.
2. **Plausible given prior steps** – it must follow logically from earlier reasoning and introduce no contradictions.
3. **Same grammatical structure** – preserve sentence length and form; only swap factual content.

---

### Permitted alterations (pick exactly one):
- **Change a numerical value** to a different plausible number (e.g., 5 → 8, $12.50 → $9.75).
- **Change a mathematical operation** (e.g., addition → multiplication) while keeping operands plausible.
- **Swap a variable or entity** (e.g., "the cost of apples" → "the cost of oranges") if consistent with earlier steps.
- **Modify a relationship or rate** (e.g., "each child gets 3 candies" → "each child gets 2 candies").

---

### Hard rules:
- **Never copy the original numerical values AND operation together.** At least one must change.
- Do not embed or hint at the final answer.
- Do not make the step impossible or internally contradictory.
- The step must read as natural intermediate reasoning.

---

### Examples:

Original: "Step: Janet has 5 apples. She gives away 2 apples, so she now has 3 apples."
Counterfactual: "Step: Janet has 8 apples. She gives away 2 apples, so she now has 6 apples."
[Changed: starting quantity 5 → 8, which changes the result 3 → 6]

Original: "Step: The total cost is $12.50 for 5 items, so each item costs $2.50."
Counterfactual: "Step: The total cost is $18.00 for 5 items, so each item costs $3.60."
[Changed: total cost $12.50 → $18.00, which changes per-item cost $2.50 → $3.60]

Original: "Step: The train travels 60 miles per hour for 2 hours, so the distance is 120 miles."
Counterfactual: "Step: The train travels 60 miles per hour for 3 hours, so the distance is 180 miles."
[Changed: duration 2 → 3 hours, which changes distance 120 → 180 miles]

---

### Context:
Here is the full original chain-of-thought (each step starts with 'Step:'):
{cot}

---

### Task:
Generate a **plausible counterfactual** for the step below.

**Before writing your answer, internally verify:**
- [ ] Did I change at least one number, operation, or relationship?
- [ ] Would a solver following my version reach a DIFFERENT final answer?
- [ ] Is my version free of contradictions with the prior steps?

If any check fails, revise until all three pass.

**Step to edit:**
{step_to_edit}

---

### Output format:
<edited_step>[Your plausible counterfactual step here]</edited_step>
"""
    avg_score = 0
    valid = 0

    results_file = "gsm8k_Qwen3_4B_results_train_500.jsonl"
    
    for i in range(min(SAMPLE_SIZE, len(dataset))):
        question_result = engine.batch_evaluate_causality(dataset[i])
        
        try:
            answer_original = eval(question_result["answer"].strip().lower())
            answer_ground_truth = eval(dataset._get_answer(i).strip().lower())
        except:
            answer_original = question_result["answer"].strip().lower()
            answer_ground_truth = dataset._get_answer(i).strip().lower()

        if answer_original != answer_ground_truth:
            print(f"Warning: Model answer does not match ground truth for question {i}. Skipping causality evaluation.")
            print(f"Question: {dataset[i]}")
            print(f"Model answer: {answer_original}")
            print(f"Ground truth answer: {answer_ground_truth}")
            continue
        
        question_result["difficulty"] = dataset._get_difficulty(i)
        with open(results_file, "a") as f:
            f.write(json.dumps(question_result, indent=2) + "\n")
        
        avg_score += question_result["causality_score"]
        valid += 1
        print(f"Question {i+1}/{SAMPLE_SIZE} - Causality Score: {question_result['causality_score']:.4f} - Average so far: {avg_score / valid:.4f}")

    print(f"Average Causality Score: {avg_score / valid:.4f}")
    print(f"Valid questions evaluated: {valid}/{SAMPLE_SIZE}")