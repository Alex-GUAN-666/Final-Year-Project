# Historical prompt assets

These files are extracted verbatim from `archive/notebooks/notebook4d41ab3228 _train and reference.ipynb`:

| File | Source | Purpose |
|---|---|---|
| `arithmetic_code_only.txt` | cell 10, `code_system_prompt` | Request a fenced Python script with a numeric `solve()` result |
| `reasoning_cot_code.txt` | cell 10, `cot_code_system_prompt` | Request a worked explanation followed by executable Python |
| `historical_chat_template.jinja` | cell 11, `tokenizer.chat_template` | Custom historical message serialization and generation prefix |

Spelling, wording, whitespace, and template structure are retained. The custom template appends `<start_working_out>` for generation although assistant training targets do not consistently use that format. Do not silently replace it with a contemporary default chat template and call the result an exact historical reproduction.

The separate teacher-generation script has its own prompts. Training, teacher generation, and inference prompt versions must be tracked independently. In the paired CoT notebooks, cell 10 would clear the prompts if executed, but has null execution count in the supplied saved runs.
