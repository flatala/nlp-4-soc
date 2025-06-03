import json
import os
import random
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any


class FaithfulnessEvaluator:
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        result_dir: str,
        results_filename: str = "human_results.json",
        stats_filename: str = "human_stats.json",
    ) -> None:
        self.samples = samples
        self.result_dir = result_dir
        self.results_filename = results_filename
        self.stats_filename = stats_filename
        self.index = 0
        self.scores = [None] * len(samples)

        os.makedirs(result_dir, exist_ok=True)
        self._build_ui()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Faithfulness Evaluation")

        self.root.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)

        nav_frame = ttk.Frame(self.root, padding=10)
        nav_frame.grid(row=0, column=0, sticky="ew")
        self.idx_label = ttk.Label(nav_frame, text="")
        self.idx_label.pack(anchor="w")

        stmt_frame = ttk.Frame(self.root, padding=(10, 0))
        stmt_frame.grid(row=1, column=0, sticky="ew")
        self.statement_lbl = ttk.Label(
            stmt_frame,
            text="",
            wraplength=700,
            justify="left",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.statement_lbl.pack(anchor="w")

        text_frame = ttk.Frame(self.root, padding=10)
        text_frame.grid(row=2, column=0, sticky="nsew")

        self.evidence_txt = tk.Text(
            text_frame,
            height=10,
            wrap="word",
            state="disabled",
            padx=5,
            pady=5,
            borderwidth=1,
            relief="sunken",
        )
        self.evidence_txt.pack(fill="both", expand=True)

        self.expl_label = ttk.Label(text_frame, text="", wraplength=700, justify="left")
        self.expl_label.pack(anchor="w", pady=(10, 0))

        # likert scale 1‑5
        scale_frame = ttk.LabelFrame(
            self.root,
            text="Faithfulness (1 = not at all, 5 = perfect)",
            padding=10,
        )

        scale_frame.grid(row=3, column=0, sticky="ew")
        self.score_var = tk.IntVar(value=0)
        for s in range(1, 6):
            ttk.Radiobutton(scale_frame, text=str(s), variable=self.score_var, value=s).pack(
                side="left", padx=5
            )

        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.grid(row=4, column=0, sticky="ew")
        ttk.Button(btn_frame, text="◀ Prev", command=self._prev).pack(side="left")
        ttk.Button(btn_frame, text="Next ▶", command=self._next).pack(side="left")
        ttk.Button(btn_frame, text="Save & Quit", command=self._finish).pack(side="right")

        self.root.bind("<Left>", lambda _e: self._prev())
        self.root.bind("<Right>", lambda _e: self._next())

        self._refresh()
        self.root.mainloop()

    def _refresh(self) -> None:
        item = self.samples[self.index]

        self.idx_label.config(text=f"Sample {self.index + 1} / {len(self.samples)}")

        stmt = item.get("statement", "[no statement provided]")
        label = item.get("label", "")
        self.statement_lbl.config(text=f"STATEMENT ({label}): {stmt}")

        self.evidence_txt.config(state="normal")
        self.evidence_txt.delete("1.0", tk.END)
        evidences = item.get("evidences", [])
        self.evidence_txt.insert(tk.END, "EVIDENCE:\n" + "\n".join(evidences) + "\n\n")
        self.evidence_txt.config(state="disabled")

        self.expl_label.config(text=f"JUSTIFICATION:\n{item.get('explanation', '')}")
        self.score_var.set(self.scores[self.index] or 0)

    def _store_score(self) -> None:
        val = self.score_var.get()
        if val:
            self.scores[self.index] = val

    def _prev(self) -> None:
        self._store_score()
        if self.index > 0:
            self.index -= 1
            self._refresh()

    def _next(self) -> None:
        self._store_score()
        if self.index < len(self.samples) - 1:
            self.index += 1
            self._refresh()

    def _finish(self) -> None:
        self._store_score()
        for sample, score in zip(self.samples, self.scores):
            sample["human_score"] = score

        # write detailed results
        with open(os.path.join(self.result_dir, self.results_filename), "w", encoding="utf‑8") as f:
            json.dump(self.samples, f, indent=2, ensure_ascii=False)

        # write summary stats
        valid_scores = [s for s in self.scores if s]
        stats = {
            "num_scored": len(valid_scores),
            "avg_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
            "distribution": {str(i): valid_scores.count(i) for i in range(1, 6)},
        }
        with open(os.path.join(self.result_dir, self.stats_filename), "w", encoding="utf‑8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        messagebox.showinfo("Saved", f"Results saved in {self.result_dir}")
        self.root.destroy()


def evaluate_faithfulness_hil(
    data: List[Dict[str, Any]],
    sample_count: int,
    result_dir: str,
    seed: int = 42,
) -> None:
    """Sample `sample_count` random items (or all if fewer) and start the GUI."""

    random.seed(seed)
    chosen = random.sample(data, k=min(sample_count, len(data)))
    FaithfulnessEvaluator(chosen, result_dir)