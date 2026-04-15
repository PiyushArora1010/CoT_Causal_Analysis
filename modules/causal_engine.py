import gradio as gr
from modules.model import Model

class CausalCoTEngine:
    def __init__(self, model_tag, max_tokens=2048, temperature=0, padding_side="left", thinking=False):
        self.model = Model(
            name=model_tag,
            max_tokens=max_tokens,
            temperature=temperature,
            padding_side=padding_side,
            thinking=thinking
        )
        
        self.initial_prompt_template = """You must think step by step. Each step must be on a new line and start with 'Step:'. Each step must be related to reasoning and not a conclusion. At the end, provide your final answer inside <answer></answer> tags. Do not include any extra text.

Question: {question}

Let's think step by step:
"""

        self.prompt_to_counterfactual = """You are an expert in causal reasoning. Your task is to generate a **plausible counterfactual version** of a specific reasoning step.

---

### MANDATORY REQUIREMENT
You **MUST** alter the factual content of the step. Returning the same step — even with minor rewording — is a failure. The counterfactual MUST produce a different result or logical outcome than the original step.

---

### What the counterfactual must satisfy:
1. **Semantically different** – change enough that a solver following your version would reach a **different final answer** than in the original.
2. **Plausible given prior steps** – it must follow logically from earlier reasoning and introduce no contradictions.
3. **Same grammatical structure** – preserve sentence length and form; only swap factual content.

---

### Hard rules:
- **Never copy the original values**
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
- [ ] Did I change the original step?
- [ ] Would a solver following my version reach a DIFFERENT final answer?
- [ ] Is my version free of contradictions with the prior steps?

If any check fails, revise until all three pass.

**Step to edit:**
{step_to_edit}

---

### Output format:
<edited_step>[Your plausible counterfactual step here]</edited_step>
"""

    def build_initial_prompt(self, question):
        return self.initial_prompt_template.format(question=question)

    def extract_answer(self, text):
        return text.split("<answer>")[-1].split("</answer>")[0]

    def extract_response_parts(self, full_response):
        before_think_text = full_response.split("</think>")[0] + "</think>"
        cot = full_response.split("</think>")[-1].split("<answer>")[0]
        return before_think_text, cot

    def run(self, prompt, apply_chat_template=True):
        return self.model.generate_response(prompt, apply_chat_template=apply_chat_template)[0]

    def initial_pass(self, question):
        prompt = self.build_initial_prompt(question)
        response = self.run(prompt)
        before_think_text, cot = self.extract_response_parts(response)
        return response, before_think_text, cot

    def edit_cot(self, step_idx, cot):
        """Edit a specific step (0-indexed) in the CoT and return the new CoT."""
        before_cot = cot.split("Step:")[0]
        cot_steps = cot.split("Step:")[1:]
        
        if step_idx >= len(cot_steps):
            return cot
        
        step_to_edit = cot_steps[step_idx]
        
        edited_step_response = self.run(self.prompt_to_counterfactual.format(cot=cot, step_to_edit=step_to_edit))
        edited_step = edited_step_response.split("<edited_step>")[-1].split("</edited_step>")[0]
        
        # Replace the step and keep only up to the edited step (truncate later steps)
        cot_steps = cot_steps[:step_idx] + [" " + edited_step.strip()]

        new_cot = before_cot + "Step: " + "Step:".join(cot_steps)
        return new_cot

    def final_pass(self, before_think_text, cot, mode):
        if mode == "Answer Only":
            prompt = before_think_text + cot + "<answer>"
        else:
            prompt = before_think_text + cot

        response = self.run(prompt, apply_chat_template=False)
        answer = self.extract_answer(response)
        return answer, response

    def evaluate_causality(self, question, mode="Full Response"):
        initial_response, before_think_text, cot = self.initial_pass(question)
        original_answer = self.extract_answer(initial_response)
        
        steps = cot.split("Step:")[1:]
        
        output_result = {"question": question, "answer": original_answer, "cot": cot, "edits": []}
        
        score = 0.0
        
        for idx, step in enumerate(steps):
            edited_cot = self.edit_cot(idx, cot)
            answer, final_response = self.final_pass(before_think_text, edited_cot, mode=mode)
            
            final_cot = final_response.split("</think>")[-1].split("<answer>")[0]
            
            output_result["edits"].append(
                {
                    "cot": final_cot,
                    "answer": answer,
                }
            )
            
            # if answer != original_answer:
            answer = answer.strip().upper()
            original_answer_ = original_answer.strip().upper()
            
            if answer in original_answer_ or original_answer_ in answer:
                score += 1.0
        
        output_result["causality_score"] = score / len(steps) if steps else 0.0
        return output_result
    
    def batch_evaluate_causality(self, question, mode="Full Response", batch_size=32):
        initial_response, before_think_text, cot = self.initial_pass(question)
        original_answer = self.extract_answer(initial_response).strip().upper()
        steps = cot.split("Step:")[1:]
        num_steps = len(steps)

        if num_steps == 0:
            return {"question": question, "answer": original_answer, "cot": cot,
                    "edits": [], "causality_score": 0.0}

        edit_prompts = []
        for idx, step in enumerate(steps):
            prompt = self.prompt_to_counterfactual.format(cot=cot, step_to_edit=step)
            edit_prompts.append(prompt)

        edited_responses = self.model.batched_generate_response(
            edit_prompts,
            apply_chat_template=True,
            batch_size=batch_size,
            n_completions=1
        )

        final_prompts = []
        edited_cots = []

        for idx, (step, full_response) in enumerate(zip(steps, edited_responses)):
            edited_step = full_response.split("<edited_step>")[-1].split("</edited_step>")[0].strip()
            before_cot = cot.split("Step:")[0]
            new_steps = steps[:idx] + [edited_step]
            new_cot = before_cot + "Step: " + "Step:".join(new_steps)
            edited_cots.append(new_cot)

            if mode == "Answer Only":
                prompt = before_think_text + new_cot + "<answer>"
            else:
                prompt = before_think_text + new_cot
            final_prompts.append(prompt)

        final_responses = self.model.batched_generate_response(
            final_prompts,
            apply_chat_template=False,
            batch_size=batch_size,
            n_completions=1
        )

        edits_output = []
        changed_count = 0

        for idx, (final_resp, _) in enumerate(zip(final_responses, edited_cots)):
            answer = self.extract_answer(final_resp).strip().upper()
            new_cot = final_resp.split("</think>")[-1].split("<answer>")[0]
            edits_output.append({"cot": new_cot, "answer": answer})

            try:
                answer_check = eval(answer)
                original_answer_check = eval(original_answer)
            except:
                answer_check = answer.strip().upper()
                original_answer_check = original_answer.strip().upper()

            if answer_check != original_answer_check:
                changed_count += 1

        causality_score = changed_count / num_steps

        return {
            "question": question,
            "answer": original_answer,
            "cot": cot,
            "edits": edits_output,
            "causality_score": causality_score
        }