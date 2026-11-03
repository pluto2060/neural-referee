"""How often does local Llama 3.2 3B write a correct program for the neural computer?
10 fixed requests, expected outputs computed in Python. Correct = the program's output on the plain
WASM reference equals the expected output. Reports first-try accuracy and accuracy within 3 rounds
(wrong output fed back to Llama).
Usage: python experiments/llama_accuracy.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import llama_front  # noqa: E402

CASES = [
    ("print the sum of the decimal digits of the input number", "98765", str(9 + 8 + 7 + 6 + 5)),
    ("print the number of characters in the input", "transformer", str(len("transformer"))),
    ("count the vowels (a, e, i, o, u) in the input", "neural computer", str(sum(c in "aeiou" for c in "neural computer"))),
    ("print the input reversed", "referee", "referee"[::-1]),
    ("print the input in uppercase", "llama", "LLAMA"),
    ("print the largest digit in the input number", "472918", "9"),
    ("print the input number plus one", "41", "42"),
    ("print YES if the input is a palindrome, otherwise NO", "racecar", "YES"),
    ("count how many times the letter l appears in the input", "hello world", "3"),
    ("print the sum of the numbers from 1 to the input number", "10", str(sum(range(11)))),
]


def main():
    rows = []
    for i, (req, arg, exp) in enumerate(CASES):
        t0 = time.time()
        code, _, _, log = llama_front.write_checked_program(req, arg, exp, f"acc{i:02d}", attempts=3)
        rounds = [r for r in log if "correct" in r]
        first = bool(rounds) and rounds[0]["round"] == 1 and rounds[0]["correct"]
        row = {"request": req, "input": arg, "expected": exp, "first_try_correct": first,
               "correct_within_3": code is not None, "rounds_used": len(log),
               "outputs": [r.get("output") for r in log], "seconds": round(time.time() - t0)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    n = len(rows)
    summary = {"n": n, "first_try_correct": sum(r["first_try_correct"] for r in rows),
               "correct_within_3_rounds": sum(r["correct_within_3"] for r in rows),
               "model": os.path.basename(llama_front.LLAMA_GGUF)}
    json.dump({"summary": summary, "rows": rows}, open("results/llama_accuracy.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
