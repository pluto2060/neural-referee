"""Meta Llama as the natural-language front end and the verdict explainer.

Backend: open-weights Llama 3.2 3B Instruct (GGUF, Q4_K_M) through llama.cpp on CPU, no API key.
Set LLAMA_GGUF to another file to swap the model.
"""
import json
import os
import re
import tempfile

from referee import tvm

LLAMA_GGUF = os.environ.get("LLAMA_GGUF", "/home/desk-1/Project/_ext/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
_llm = None

SYSTEM_C = """You write tiny C programs for a computer that runs inside a transformer.
Rules:
- Define exactly: void compute(const char *input) { ... }
- No #include, no libc, no arrays larger than needed, no recursion.
- Do NOT use *, / or % (no multiplication, division or modulo). Use loops of + and -.
- Available helpers: putchar(int), print_str(const char*), print_int(int), parse_int(const char*), str_len(const char*).
- Print the answer followed by '\\n'.
Reply with one ```c code block only.

- Characters of input are ASCII; a decimal digit c has value c - '0'.
- Do exactly what the request says, nothing more.

Example request: greet the input name
```c
void compute(const char *input) {
    print_str("Hello ");
    print_str(input);
    print_str("!\\n");
}
```

Example request: count the vowels in the input
```c
void compute(const char *input) {
    int n = 0;
    int i = 0;
    while (input[i]) {
        char c = input[i];
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') n = n + 1;
        i = i + 1;
    }
    print_int(n);
    putchar('\\n');
}
```"""

SYSTEM_EXPLAIN = """You explain a verification report to a non-expert in 3 short sentences.
Say whether the computation was correct, at which step it first went wrong, which part of the
model caused it, and what that means. State whether the final printed output matched the reference
(field outputs_match); if it matched, say the error stayed hidden in the execution and only the
step-by-step check found it. Use only facts from the report. No speculation."""


def _model():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(model_path=LLAMA_GGUF, n_ctx=4096, n_threads=os.cpu_count(), verbose=False, seed=0)
    return _llm


def chat(system, user, max_tokens=512):
    out = _model().create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0, max_tokens=max_tokens)
    return out["choices"][0]["message"]["content"]


def write_program(request, attempts=3):
    """Ask Llama for a C program; compile it; feed compiler errors back. Returns (code, log)."""
    from transformer_vm.compilation.compile_wasm import compile_c_to_wasm
    log, user = [], f"Request: {request}"
    for i in range(attempts):
        reply = chat(SYSTEM_C, user)
        m = re.search(r"```(?:c)?\s*\n(.*?)```", reply, re.S)
        code = m.group(1) if m else reply
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "prog.c")
            open(path, "w").write(code)
            try:
                compile_c_to_wasm(path)
                log.append({"attempt": i + 1, "compiled": True})
                return code, log
            except Exception as e:  # compiler error goes back to Llama
                err = str(e)[-800:]
                log.append({"attempt": i + 1, "compiled": False, "error": err})
                user = f"Request: {request}\nYour previous code:\n```c\n{code}\n```\nfailed to compile:\n{err}\nFix it."
    return None, log


def build(code, args, name="llama_prog"):
    """Compile code with input args into transformer-vm tokens and its reference trace."""
    from transformer_vm.compilation.compile_wasm import compile_program
    from transformer_vm.wasm.reference import generate_ref
    work = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "programs")
    os.makedirs(work, exist_ok=True)
    c_path = os.path.join(work, f"{name}.c")
    open(c_path, "w").write(code)
    base = os.path.join(work, name)
    compile_program(c_path, args, base)
    generate_ref(base + ".txt")
    return base + ".txt", base + "_ref.txt"


def decode_output(tokens):
    out = []
    for t in tokens:
        m = re.fullmatch(r"out\((.+)\)", t)
        if m:
            v = m.group(1)
            out.append(v if len(v) == 1 else chr(int(v, 16)))
    return "".join(out)


def explain(report):
    slim = {k: report.get(k) for k in ("verdict", "first_wrong_step", "trace_len", "faulty_sublayer",
                                       "fault_origin_position", "stage", "hash_comparisons",
                                       "prover_output", "reference_output")}
    slim["outputs_match"] = report.get("prover_output") == report.get("reference_output")
    return chat(SYSTEM_EXPLAIN, json.dumps(slim), max_tokens=200)


def write_checked_program(request, args, expect, name, attempts=3):
    """Write -> compile -> run on the plain WASM reference -> compare with the expected output.
    A wrong output goes back to Llama as feedback. Returns (code, txt, ref_txt, log)."""
    from referee import tvm
    log, req = [], request
    for i in range(attempts):
        code, clog = write_program(req)
        log.append({"round": i + 1, "compile_log": clog})
        if code is None:
            continue
        txt, ref_txt = build(code, args, f"{name}_r{i + 1}")
        got = decode_output(open(ref_txt).read().split()).strip()
        log[-1].update(output=got, expected=expect, correct=got == expect.strip())
        if got == expect.strip():
            return code, txt, ref_txt, log
        req = (f"{request}\nYour previous program:\n```c\n{code}\n```\nprinted {got!r} for input "
               f"{args!r}, but the correct output is {expect!r}. Write a corrected program.")
    return None, None, None, log
